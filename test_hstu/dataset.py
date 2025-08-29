"""
数据集处理和特征转换
包含数据加载、特征处理和嵌入保存功能
支持多worker数据加载和HSTU模型的特征处理需求
"""

import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集
    支持HSTU模型的特征处理需求和多worker数据加载
    
    Args:
        data_dir: 数据文件目录
        args: 全局参数
    
    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """初始化数据集"""
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
            self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
            self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
            self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

        # 初始化特征类型字典
        self.USER_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = self.feature_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = self.feature_types['item_continual']
        self.USER_ARRAY_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in self.feature_types['item_emb']}

        # 为多进程准备文件句柄
        self.data_file = None

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100', '117', '111', '118', '101', '102', '119', '120',
            '114', '112', '121', '115', '122', '116'
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            if feat_id in self.mm_emb_dict and self.mm_emb_dict[feat_id]:
                emb_shape = list(self.mm_emb_dict[feat_id].values())[0].shape[0]
                feat_default_value[feat_id] = np.zeros(emb_shape, dtype=np.float32)
            else:
                feat_default_value[feat_id] = np.zeros(32, dtype=np.float32)  # 默认维度

        return feat_default_value, feat_types, feat_statistics
    
    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file_path = self.data_dir / "seq.jsonl"
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        # 每个worker进程都有自己的文件句柄
        if self.data_file is None:
            self.data_file = open(self.data_file_path, 'rb')
        
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat_tensors: 用户序列特征张量字典
            pos_feat_tensors: 正样本特征张量字典
            neg_feat_tensors: 负样本特征张量字典
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        # 处理None值
        for i in range(len(seq_feat)):
            if seq_feat[i] is None:
                seq_feat[i] = self.feature_default_value
            if pos_feat[i] is None:
                pos_feat[i] = self.feature_default_value
            if neg_feat[i] is None:
                neg_feat[i] = self.feature_default_value

        # 将特征转换为张量格式
        seq_feat_tensors = self._process_single_sequence_features(seq_feat)
        pos_feat_tensors = self._process_single_sequence_features(pos_feat)
        neg_feat_tensors = self._process_single_sequence_features(neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat_tensors, pos_feat_tensors, neg_feat_tensors

    def fill_missing_feat(self, feat, item_id):
        """填充缺失的特征"""
        if feat is None:
            return self.feature_default_value
        return feat

    def _random_neq(self, low, high, ts):
        """随机生成不在ts中的负样本"""
        neg = np.random.randint(low, high)
        while neg in ts:
            neg = np.random.randint(low, high)
        return neg

    def _process_single_sequence_features(self, sequence_features):
        """处理单个序列的特征，转换为张量格式"""
        feature_tensors = {}
        
        # 为每个特征类型创建张量
        seq_len = len(sequence_features)
        
        # 处理稀疏特征
        for feat_id in self.ITEM_SPARSE_FEAT:
            feat_data = np.zeros(seq_len, dtype=np.int64)
            for i, feat_dict in enumerate(sequence_features):
                if isinstance(feat_dict, dict) and feat_id in feat_dict:
                    feat_data[i] = feat_dict[feat_id]
            feature_tensors[feat_id] = torch.tensor(feat_data, dtype=torch.long)
        
        # 处理数组特征
        for feat_id in self.ITEM_ARRAY_FEAT:
            feat_data = np.zeros((seq_len, 10), dtype=np.int64)  # 假设最大数组长度为10
            for i, feat_dict in enumerate(sequence_features):
                if isinstance(feat_dict, dict) and feat_id in feat_dict:
                    array_val = feat_dict[feat_id]
                    if isinstance(array_val, list):
                        feat_data[i, :len(array_val)] = array_val[:10]
            feature_tensors[feat_id] = torch.tensor(feat_data, dtype=torch.long)
        
        # 处理连续特征
        for feat_id in self.ITEM_CONTINUAL_FEAT:
            feat_data = np.zeros(seq_len, dtype=np.float32)
            for i, feat_dict in enumerate(sequence_features):
                if isinstance(feat_dict, dict) and feat_id in feat_dict:
                    feat_data[i] = float(feat_dict[feat_id])
            feature_tensors[feat_id] = torch.tensor(feat_data, dtype=torch.float32)
        
        # 处理嵌入特征
        for feat_id in self.ITEM_EMB_FEAT:
            emb_dim = self.ITEM_EMB_FEAT[feat_id]
            feat_data = np.zeros((seq_len, emb_dim), dtype=np.float32)
            for i, feat_dict in enumerate(sequence_features):
                if isinstance(feat_dict, dict) and feat_id in feat_dict:
                    emb_val = feat_dict[feat_id]
                    if isinstance(emb_val, (list, np.ndarray)):
                        feat_data[i] = emb_val[:emb_dim]
            feature_tensors[feat_id] = torch.tensor(feat_data, dtype=torch.float32)
        
        # 处理用户特征
        for feat_id in self.USER_SPARSE_FEAT:
            feat_data = np.zeros(seq_len, dtype=np.int64)
            for i, feat_dict in enumerate(sequence_features):
                if isinstance(feat_dict, dict) and feat_id in feat_dict:
                    feat_data[i] = feat_dict[feat_id]
            feature_tensors[feat_id] = torch.tensor(feat_data, dtype=torch.long)
        
        for feat_id in self.USER_ARRAY_FEAT:
            feat_data = np.zeros((seq_len, 10), dtype=np.int64)
            for i, feat_dict in enumerate(sequence_features):
                if isinstance(feat_dict, dict) and feat_id in feat_dict:
                    array_val = feat_dict[feat_id]
                    if isinstance(array_val, list):
                        feat_data[i, :len(array_val)] = array_val[:10]
            feature_tensors[feat_id] = torch.tensor(feat_data, dtype=torch.long)
        
        return feature_tensors

    def collate_fn(self, batch):
        """数据整理函数，支持新的数据格式和多worker"""
        batch_data = list(zip(*batch))
        
        # 堆叠基本序列数据
        seq = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in batch_data[0]])
        pos = torch.stack([torch.tensor(pos, dtype=torch.long) for pos in batch_data[1]])
        neg = torch.stack([torch.tensor(neg, dtype=torch.long) for neg in batch_data[2]])
        token_type = torch.stack([torch.tensor(tt, dtype=torch.long) for tt in batch_data[3]])
        next_token_type = torch.stack([torch.tensor(ntt, dtype=torch.long) for ntt in batch_data[4]])
        next_action_type = torch.stack([torch.tensor(nat, dtype=torch.long) for nat in batch_data[5]])
        
        # 合并特征张量
        seq_feat_tensors = self._merge_batch_features(batch_data[6])
        pos_feat_tensors = self._merge_batch_features(batch_data[7])
        neg_feat_tensors = self._merge_batch_features(batch_data[8])
        
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat_tensors, pos_feat_tensors, neg_feat_tensors

    def _merge_batch_features(self, batch_features):
        """合并批次特征张量"""
        if not batch_features:
            return {}
        
        merged_features = {}
        feat_keys = batch_features[0].keys()
        
        for feat_id in feat_keys:
            # 找到第一个非空张量来确定形状
            sample_tensor = None
            for feat_dict in batch_features:
                if feat_id in feat_dict and feat_dict[feat_id] is not None:
                    sample_tensor = feat_dict[feat_id]
                    break
            
            if sample_tensor is None:
                continue
            
            # 创建批次张量
            batch_size = len(batch_features)
            batch_tensor = torch.zeros((batch_size,) + sample_tensor.shape, dtype=sample_tensor.dtype)
            
            for i, feat_dict in enumerate(batch_features):
                if feat_id in feat_dict and feat_dict[feat_id] is not None:
                    batch_tensor[i] = feat_dict[feat_id]
            
            merged_features[feat_id] = batch_tensor
        
        return merged_features


class MyTestDataset(torch.utils.data.Dataset):
    """测试数据集类"""
    
    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id
        
        # 加载必要的数据
        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
            self.indexer = indexer
        
        # 初始化特征信息
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        
        # 加载测试数据
        self._load_test_data()
    
    def _init_feat_info(self):
        """初始化特征信息（与训练数据集相同）"""
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100', '117', '111', '118', '101', '102', '119', '120',
            '114', '112', '121', '115', '122', '116'
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            if feat_id in self.mm_emb_dict and self.mm_emb_dict[feat_id]:
                emb_shape = list(self.mm_emb_dict[feat_id].values())[0].shape[0]
                feat_default_value[feat_id] = np.zeros(emb_shape, dtype=np.float32)
            else:
                feat_default_value[feat_id] = np.zeros(32, dtype=np.float32)

        return feat_default_value, feat_types, feat_statistics
    
    def _load_test_data(self):
        """加载测试数据"""
        # 这里应该加载实际的测试数据
        # 暂时创建空列表，需要根据实际数据格式填充
        self.test_sequences = []
        self.test_user_ids = []
    
    def __len__(self):
        return len(self.test_sequences)
    
    def __getitem__(self, idx):
        # 返回测试数据格式
        seq = self.test_sequences[idx]
        user_id = self.test_user_ids[idx]
        
        # 处理序列特征
        seq_feat = {}  # 根据实际需要填充
        token_type = np.ones_like(seq)  # 假设都是item
        
        return seq, token_type, seq_feat, user_id
    
    def collate_fn(self, batch):
        """测试数据整理函数"""
        seqs, token_types, seq_feats, user_ids = zip(*batch)
        
        seq = torch.stack([torch.tensor(s, dtype=torch.long) for s in seqs])
        token_type = torch.stack([torch.tensor(tt, dtype=torch.long) for tt in token_types])
        
        return seq, token_type, seq_feats, list(user_ids)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                if base_path.exists():
                    for json_file in base_path.glob('*.json'):
                        with open(json_file, 'r', encoding='utf-8') as file:
                            for line in file:
                                data_dict_origin = json.loads(line.strip())
                                insert_emb = data_dict_origin['emb']
                                if isinstance(insert_emb, list):
                                    insert_emb = np.array(insert_emb, dtype=np.float32)
                                data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                                emb_dict.update(data_dict)
            except Exception as e:
                print(f"Warning: Failed to load multimodal embedding {feat_id}: {e}")
        
        if feat_id == '81':
            pkl_file = Path(mm_path, f'emb_{feat_id}_{shape}.pkl')
            if pkl_file.exists():
                try:
                    with open(pkl_file, 'rb') as f:
                        emb_dict = pickle.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load multimodal embedding {feat_id}: {e}")
        
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb with {len(emb_dict)} items')
    
    return mm_emb_dict


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def worker_init_fn(worker_id):
    """
    多进程worker初始化函数
    确保每个worker都有独立的随机状态和文件句柄
    """
    # 设置不同的随机种子
    np.random.seed(torch.initial_seed() % (2**32) + worker_id)
    
    # 获取当前worker的dataset实例
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # 重置文件句柄，确保每个worker都有独立的文件句柄
        if hasattr(dataset, 'data_file') and dataset.data_file is not None:
            dataset.data_file.close()
            dataset.data_file = None