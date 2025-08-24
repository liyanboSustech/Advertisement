import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """用户序列数据集"""

    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id
        self.device = args.device  # 用于预处理时的设备设置（CPU，多线程不支持GPU）

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
        self._init_feat_max_len()  # 初始化特征最大长度信息

    def _load_data_and_offsets(self):
        """加载用户序列数据和偏移量"""
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """加载单个用户数据"""
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """负采样：生成不在序列s中的随机整数"""
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def _init_feat_info(self):
        """初始化特征信息（缺省值、类型、统计量）"""
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {
            'user_sparse': ['103', '104', '105', '109'],
            'item_sparse': [
                '100', '117', '111', '118', '101', '102', '119',
                '120', '114', '112', '121', '115', '122', '116'
            ],
            'item_array': [],
            'user_array': ['106', '107', '108', '110'],
            'item_emb': self.mm_emb_ids,
            'user_continual': [],
            'item_continual': []
        }

        # 初始化特征缺省值和统计量
        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array'] + feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual'] + feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            # 多模态特征缺省值为零向量
            emb_dim = {
                "81": 32, "82": 1024, "83": 3584,
                "84": 4096, "85": 3584, "86": 3584
            }[feat_id]
            feat_default_value[feat_id] = np.zeros(emb_dim, dtype=np.float32)

        return feat_default_value, feat_types, feat_statistics

    def _init_feat_max_len(self):
        """初始化数组类型特征的最大长度（用于padding）"""
        self.feat_max_len = {}
        # 数组类型特征需要记录最大长度（可根据实际数据统计调整）
        for k in self.feature_types['item_array'] + self.feature_types['user_array']:
            self.feat_max_len[k] = 10  # 示例值，可根据数据调整

    def feat2tensor(self, feat, feat_id):
        """将单个特征转换为tensor（在__getitem__中调用）"""
        if feat_id in self.feature_types['item_array'] or feat_id in self.feature_types['user_array']:
            # 数组类型特征：padding到最大长度
            max_len = self.feat_max_len[feat_id]
            feat_np = np.zeros(max_len, dtype=np.int64)
            actual_len = min(len(feat), max_len)
            feat_np[:actual_len] = feat[:actual_len]
            return torch.from_numpy(feat_np)
        elif feat_id in self.feature_types['user_sparse'] or feat_id in self.feature_types['item_sparse']:
            # 稀疏特征：直接转换为tensor
            return torch.tensor(feat, dtype=torch.int64)
        elif feat_id in self.feature_types['item_emb']:
            # 多模态特征：转换为float32 tensor
            return torch.from_numpy(feat.astype(np.float32))
        elif feat_id in self.feature_types['user_continual'] or feat_id in self.feature_types['item_continual']:
            # 连续特征：转换为float tensor
            return torch.tensor(feat, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown feature ID: {feat_id}")

    def process_feat_dict(self, feat_dict):
        """处理单条特征字典，将所有特征转换为tensor"""
        processed = {}
        for feat_id, value in feat_dict.items():
            processed[feat_id] = self.feat2tensor(value, feat_id)
        return processed

    def fill_missing_feat(self, feat, item_id):
        """填充缺失特征"""
        if feat is None:
            feat = {}
        filled_feat = {}
        # 复制已有特征
        for k in feat.keys():
            filled_feat[k] = feat[k]
        # 填充缺失特征
        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        # 补充多模态特征
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                emb = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
                if isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                filled_feat[feat_id] = emb
        return filled_feat

    def __getitem__(self, uid):
        """获取单个样本并预处理特征为tensor"""
        user_sequence = self._load_user_data(uid)

        # 扩展用户序列（包含user和item）
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        # 初始化输出数组
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        # 特征数组（存储预处理后的tensor）
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1] if ext_user_sequence else (0, {}, 0, 0)
        idx = self.maxlen

        # 收集item ID用于负采样
        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # 填充序列（从后往前）
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt

            # 填充特征并转换为tensor
            feat = self.fill_missing_feat(feat, i)
            feat_tensor = self.process_feat_dict(feat)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            next_feat_tensor = self.process_feat_dict(next_feat)

            # 填充序列数据
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            next_action_type[idx] = next_act_type if next_act_type is not None else 0
            seq_feat[idx] = feat_tensor

            # 处理正负样本
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat_tensor
                # 负采样
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.process_feat_dict(
                    self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
                )

            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        # 填充默认值（针对未填充的位置）
        default_feat_tensor = self.process_feat_dict(self.feature_default_value)
        seq_feat = np.where(seq_feat == None, default_feat_tensor, seq_feat)
        pos_feat = np.where(pos_feat == None, default_feat_tensor, pos_feat)
        neg_feat = np.where(neg_feat == None, default_feat_tensor, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        return len(self.seq_offsets)

    @staticmethod
    def collate_fn(batch):
        """合并批次数据"""
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        
        # 转换基础数据为tensor
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))

        # 合并特征字典（按特征ID分组）
        def collate_feat(feat_list):
            batch_feat = {}
            # 初始化特征字典
            first_feat = feat_list[0][0] if len(feat_list[0]) > 0 else {}
            for k in first_feat.keys():
                batch_feat[k] = []
            # 收集每个特征的所有样本
            for sample in feat_list:
                for seq_item in sample:
                    for k, v in seq_item.items():
                        batch_feat[k].append(v.unsqueeze(0))  # 增加批次维度
            # 堆叠为tensor
            for k in batch_feat:
                batch_feat[k] = torch.cat(batch_feat[k], dim=0)
            return batch_feat

        # 处理序列特征、正样本特征、负样本特征
        batch_seq_feat = collate_feat(seq_feat)
        batch_pos_feat = collate_feat(pos_feat)
        batch_neg_feat = collate_feat(neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, batch_seq_feat, batch_pos_feat, batch_neg_feat


class MyTestDataset(MyDataset):
    """测试数据集"""

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """处理冷启动特征"""
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if isinstance(feat_value, list):
                value_list = [0 if isinstance(v, str) else v for v in feat_value]
                processed_feat[feat_id] = value_list
            elif isinstance(feat_value, str):
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """获取测试样本"""
        user_sequence = self._load_user_data(uid)
        user_id = ""

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if isinstance(u, str):
                    user_id = u
                else:
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if isinstance(u, str):
                    u = 0
                user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2))
            if i and item_feat:
                if i > self.itemnum:
                    i = 0
                item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        # 初始化输出
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        idx = self.maxlen

        # 填充序列
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq_feat[idx] = self.process_feat_dict(feat)
            seq[idx] = i
            token_type[idx] = type_
            idx -= 1
            if idx == -1:
                break

        # 填充默认值
        default_feat_tensor = self.process_feat_dict(self.feature_default_value)
        seq_feat = np.where(seq_feat == None, default_feat_tensor, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            return len(pickle.load(f))

    @staticmethod
    def collate_fn(batch):
        """测试集合并批次"""
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        
        # 合并特征
        batch_feat = {}
        first_feat = seq_feat[0][0] if len(seq_feat[0]) > 0 else {}
        for k in first_feat.keys():
            batch_feat[k] = []
        for sample in seq_feat:
            for seq_item in sample:
                for k, v in seq_item.items():
                    batch_feat[k].append(v.unsqueeze(0))
        for k in batch_feat:
            batch_feat[k] = torch.cat(batch_feat[k], dim=0)

        return seq, token_type, batch_feat, user_id


def save_emb(emb, save_path):
    """保存Embedding为二进制文件"""
    num_points = emb.shape[0]
    num_dimensions = emb.shape[1]
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """加载多模态特征"""
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
                            data = json.loads(line.strip())
                            emb = data['emb']
                            if isinstance(emb, list):
                                emb = np.array(emb, dtype=np.float32)
                            emb_dict[data['anonymous_cid']] = emb
            except Exception as e:
                print(f"Error loading {feat_id}: {e}")
        else:
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
    return mm_emb_dict