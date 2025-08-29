"""
数据集处理和特征转换
包含数据加载、特征处理和嵌入保存功能
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
    支持HSTU模型的特征处理需求
    
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

        self.feature_default_value = json.load(open(Path(data_dir, "feature_default_value.json"), 'r'))
        self.feature_types = json.load(open(Path(data_dir, "feature_types.json"), 'r'))
        self.feat_statistics = json.load(open(Path(data_dir, "feat_statistics.json"), 'r'))

        # 初始化特征类型字典
        self.USER_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = self.feature_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = self.feature_types['item_continual']
        self.USER_ARRAY_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in self.feature_types['item_emb']}

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file_path = self.data_dir / "seq.jsonl"
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)
        # 为多进程准备，每个worker会有自己的文件句柄
        self._worker_init_fn = self._init_worker

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        # 每个worker进程都有自己的文件句柄
        if not hasattr(self, 'data_file') or self.data_file is None:
            self.data_file = open(self.data_file_path, 'rb')
        
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data
    
    def _init_worker(self, worker_id):
        """
        初始化worker进程，每个worker都有自己的文件句柄
        """
        # 确保每个worker都有自己的文件句柄
        if hasattr(self, 'data_file'):
            self.data_file = None
    
    def __del__(self):
        """
        析构函数，关闭文件句柄
        """
        if hasattr(self, 'data_file') and self.data_file is not None:
            self.data_file.close()

    def __len__(self):
        """返回数据集长度"""
        return len(self.train_offsets) - 1

    def __getitem__(self, idx):
        """获取单个数据样本"""
        start = self.train_offsets[idx]
        end = self.train_offsets[idx + 1]
        seq = self.train_data[start:end]

        # 处理序列长度
        if len(seq) > self.maxlen + 1:
            seq = seq[-self.maxlen - 1:]
        elif len(seq) < self.maxlen + 1:
            seq = np.pad(seq, (self.maxlen + 1 - len(seq), 0), 'constant')

        tokens = seq[:-1].copy()
        labels = seq[-1:].copy()

        # 创建mask
        mask = np.ones(self.maxlen, dtype=np.int64)
        mask[tokens == 0] = 0

        # 处理特征
        seq_feature_tensors = self.process_features_to_tensors([tokens])
        pos_feature_tensors = self.process_features_to_tensors([labels])

        return (torch.tensor(tokens, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),
                seq_feature_tensors,
                pos_feature_tensors)

    def process_features_to_tensors(self, sequences):
        """
        将序列特征转换为张量格式
        适配HSTU模型的特征处理需求
        """
        feature_tensors = {}
        
        # 处理稀疏特征
        for feat_id in self.ITEM_SPARSE_FEAT:
            batch_size = len(sequences)
            seq_len = len(sequences[0])
            feat_data = np.zeros((batch_size, seq_len), dtype=np.int64)
            
            for i, seq in enumerate(sequences):
                for j, token in enumerate(seq):
                    if token != 0 and token in self.item_feat_dict:
                        item_id = self.indexer_i_rev.get(token, token)
                        if feat_id in self.item_feat_dict.get(str(item_id), {}):
                            feat_data[i, j] = self.item_feat_dict[str(item_id)][feat_id]
            
            feature_tensors[feat_id] = torch.tensor(feat_data, dtype=torch.long)

        # 处理数组特征
        for feat_id in self.ITEM_ARRAY_FEAT:
            batch_size = len(sequences)
            seq_len = len(sequences[0])
            feat_data = np.zeros((batch_size, seq_len, 10), dtype=np.int64)  # 假设最大数组长度为10
            
            for i, seq in enumerate(sequences):
                for j, token in enumerate(seq):
                    if token != 0 and token in self.item_feat_dict:
                        item_id = self.indexer_i_rev.get(token, token)
                        if feat_id in self.item_feat_dict.get(str(item_id), {}):
                            array_val = self.item_feat_dict[str(item_id)][feat_id]
                            if isinstance(array_val, list):
                                feat_data[i, j, :len(array_val)] = array_val[:10]
            
            feature_tensors[feat_id] = torch.tensor(feat_data, dtype=torch.long)

        # 处理连续特征
        for feat_id in self.ITEM_CONTINUAL_FEAT:
            batch_size = len(sequences)
            seq_len = len(sequences[0])
            feat_data = np.zeros((batch_size, seq_len), dtype=np.float32)
            
            for i, seq in enumerate(sequences):
                for j, token in enumerate(seq):
                    if token != 0 and token in self.item_feat_dict:
                        item_id = self.indexer_i_rev.get(token, token)
                        if feat_id in self.item_feat_dict.get(str(item_id), {}):
                            feat_data[i, j] = float(self.item_feat_dict[str(item_id)][feat_id])
            
            feature_tensors[feat_id] = torch.tensor(feat_data, dtype=torch.float32)

        # 处理嵌入特征
        for feat_id in self.ITEM_EMB_FEAT:
            batch_size = len(sequences)
            seq_len = len(sequences[0])
            emb_dim = self.ITEM_EMB_FEAT[feat_id]
            feat_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            
            for i, seq in enumerate(sequences):
                for j, token in enumerate(seq):
                    if token != 0 and token in self.item_feat_dict:
                        item_id = self.indexer_i_rev.get(token, token)
                        if feat_id in self.item_feat_dict.get(str(item_id), {}):
                            emb_val = self.item_feat_dict[str(item_id)][feat_id]
                            if isinstance(emb_val, list):
                                feat_data[i, j] = emb_val[:emb_dim]
            
            feature_tensors[feat_id] = torch.tensor(feat_data, dtype=torch.float32)

        return feature_tensors

    def feat2tensor(self, sequences, feat_id, device):
        """
        将特征转换为张量格式（保持向后兼容）
        """
        feature_tensors = self.process_features_to_tensors(sequences)
        return feature_tensors.get(feat_id, torch.zeros((len(sequences), len(sequences[0])), dtype=torch.long)).to(device)


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
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
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

def collate_fn(batch):
    """数据整理函数"""
    tokens, labels, masks, seq_features, pos_features = zip(*batch)
    
    return (torch.stack(tokens),
            torch.stack(labels),
            torch.stack(masks),
            seq_features,
            pos_features)