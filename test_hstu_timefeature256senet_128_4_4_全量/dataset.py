# -*- coding: utf-8 -*-
import json
import pickle
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, args):
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
        print("usernum:", self.usernum, "itemnum:", self.itemnum)

    def _load_data_and_offsets(self):
        self.data_file_path = self.data_dir / "seq.jsonl"
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)
        self._worker_init_fn = self._init_worker

    def _load_user_data(self, uid):
        if not hasattr(self, 'data_file') or self.data_file is None:
            self.data_file = open(self.data_file_path, 'rb')
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _init_worker(self, worker_id):
        if hasattr(self, 'data_file'):
            self.data_file = None

    def __del__(self):
        if hasattr(self, 'data_file') and self.data_file is not None:
            self.data_file.close()

    def _random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    # =================================================================
    #  根据图 5 重写 _add_time_features
    # =================================================================
    def _add_time_features(self, user_sequence, tau=86400):
        ts_array = np.array([r[5] for r in user_sequence], dtype=np.int64)

        # time_gap, log_gap
        prev_ts_array = np.roll(ts_array, 1)
        prev_ts_array[0] = ts_array[0]
        time_gap = ts_array - prev_ts_array
        time_gap[0] = 0
        log_gap = np.log1p(time_gap)

        # hour, weekday, month
        ts_utc8 = ts_array + 8 * 3600
        hours = (ts_utc8 % 86400) // 3600
        weekdays = ((ts_utc8 // 86400 + 4) % 7).astype(np.int32)
        months = pd.to_datetime(ts_utc8, unit='s').month.to_numpy()

        # time decay
        last_ts = ts_array[-1]
        # delta_t是时间差
        # delta_scaled是时间差经过log1p和归一化后的结果
        # delta_scaled是什么样子呢？
        # 
        delta_t = last_ts - ts_array
        delta_scaled = np.log1p(delta_t / tau)

        # 添加到 sequence
        new_sequence = []
        for idx, record in enumerate(user_sequence):
            u, i, user_feat, item_feat, action_type, ts = record
            if user_feat is None:
                user_feat = {}
            user_feat["200"] = int(hours[idx])
            user_feat["201"] = int(weekdays[idx])
            # user_feat["202"] = int(time_gap[idx])
            user_feat["203"] = float(log_gap[idx])
            user_feat["204"] = int(months[idx])
            user_feat["205"] = float(delta_scaled[idx])
            new_sequence.append((u, i, user_feat, item_feat, action_type, ts))
        return ts_array, new_sequence 

    def _features_to_tensors(self, feat_array):
        feature_tensors = {}
        all_sparse = self.feature_types['user_sparse'] + self.feature_types['item_sparse']
        all_array = self.feature_types['user_array'] + self.feature_types['item_array']
        all_continual = self.feature_types['user_continual'] + self.feature_types['item_continual']
        all_emb = self.feature_types['item_emb']

        for k in all_sparse:
            data = [item.get(k, self.feature_default_value[k]) for item in feat_array]
            feature_tensors[k] = torch.tensor(data, dtype=torch.long)

        for k in all_array:
            data_list = [item.get(k, self.feature_default_value[k]) for item in feat_array]
            feature_tensors[k] = data_list

        for k in all_continual:
            data = [item.get(k, self.feature_default_value[k]) for item in feat_array]
            feature_tensors[k] = torch.tensor(data, dtype=torch.float32)

        for k in all_emb:
            data_list = [item.get(k, self.feature_default_value[k]) for item in feat_array]
            emb_array = np.array(data_list, dtype=np.float32)
            feature_tensors[k] = torch.from_numpy(emb_array)

        return feature_tensors

    # =================================================================
    #  根据图 4 重写 __getitem__
    # =================================================================
    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)
        # 
        ts_array,user_sequence = self._add_time_features(user_sequence)
        # user_seq
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
                # insert得到的格式是 (id, feat, type, action_type)，type=2表示user
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))
        # ext_user_sequence 中每条记录格式为 (id, feat, type, action_type)，type=1表示item，type=2表示user
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = [self.feature_default_value] * (self.maxlen + 1)
        pos_feat = [self.feature_default_value] * (self.maxlen + 1)
        neg_feat = [self.feature_default_value] * (self.maxlen + 1)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen
        # ext_user_sequence的格式是 (id, feat, type, action_type)
        # type = 1 表示 item, type = 2 表示 user
        
        ts = {record[0] for record in ext_user_sequence if record[2] == 1 and record[0]}
        # -------------------------------------------------------------
        #  以下 for-loop 完全按照图 4 手写逻辑重写
        # -------------------------------------------------------------
        for record_tuple in reversed(ext_user_sequence[:-1]):
            # 1. 拆包当前 token
            i, feat_dict, type_, act_type = record_tuple
            # 2. 我们只关心 item token（type==1）
            if type_ == 2:
                continue
            # 3. 取出“紧邻的前一条 token”（倒序遍历时它就是 nxt）
            next_i, next_feat, next_type, next_act_type = nxt
            # 4. 如果前一条是 user token，则 prev_feat_dict 就是 u_feat
            next_feat = self.fill_missing_feat(next_feat, next_i)
            if next_type == 2:
                u_feat = next_feat          # 真正的 user_feat
            else:
                u_feat = {}                      # 没有可用的 user token
            # 5. 当前 item 的原始特征就是 feat_dict
            i_feat = self.fill_missing_feat(feat_dict, i)
            i_feat = self._transfer_context_features(
                u_feat, i_feat, ["200", "201", "203", "204", "205"])

            seq[idx] = i
            seq_feat[idx] = i_feat
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type

            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(
                    self.item_feat_dict.get(str(neg_id), {}), neg_id)

            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat_tensors = self._features_to_tensors(seq_feat)
        pos_feat_tensors = self._features_to_tensors(pos_feat)
        neg_feat_tensors = self._features_to_tensors(neg_feat)

        return (seq, pos, neg, token_type, next_token_type, next_action_type,
                seq_feat_tensors, pos_feat_tensors, neg_feat_tensors)

    def __len__(self):
        return len(self.seq_offsets)

    def _init_feat_info(self):
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}

        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}

        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121',
            '115', '122', '116', '200', '201', '204'
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = ['203', '205']

        fixed_vocab_size = {'200': 24, '201': 7, '204': 13}

        for feat_id in feat_types['user_sparse'] + feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = fixed_vocab_size.get(
                feat_id, len(self.indexer['f'].get(feat_id, {})))

        for feat_id in feat_types['item_array'] + feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'].get(feat_id, {}))

        for feat_id in feat_types['user_continual'] + feat_types['item_continual']:
            feat_default_value[feat_id] = 0.0

        for feat_id in feat_types['item_emb']:
            shape = EMB_SHAPE_DICT[feat_id]
            feat_default_value[feat_id] = np.zeros(shape, dtype=np.float32)
            feat_statistics[feat_id] = shape

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        filled_feat = {} if feat is None else feat.copy()
        all_feat_ids = [fid for f_list in self.feature_types.values() for fid in f_list]
        for feat_id in all_feat_ids:
            if feat_id not in filled_feat:
                filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0:
                creative_id = self.indexer_i_rev.get(item_id)
                if creative_id and creative_id in self.mm_emb_dict[feat_id]:
                    emb = self.mm_emb_dict[feat_id][creative_id]
                    if isinstance(emb, list):
                        emb = np.array(emb, dtype=np.float32)
                    filled_feat[feat_id] = emb
        return filled_feat

    # =================================================================
    #  根据图 6 增加 _transfer_context_features
    # =================================================================
    def _transfer_context_features(self, user_feat: dict, item_feat: dict, cols_to_trans: list):
        """把 user_feat 中指定列的值直接赋给 item_feat"""
        for col in cols_to_trans:
            if col in user_feat:          # 先检查
                item_feat[col] = user_feat[col]
        return item_feat

    @staticmethod
    def collate_fn(batch):
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))

        def batch_features(feat_list):
            batched_feat = {}
            if not feat_list:
                return batched_feat
            all_keys = feat_list[0].keys()
            for k in all_keys:
                if isinstance(feat_list[0][k], torch.Tensor):
                    batched_feat[k] = torch.stack([d[k] for d in feat_list])
                else:
                    max_len = 0
                    for d in feat_list:
                        for item_list in d[k]:
                            if isinstance(item_list, list):
                                max_len = max(max_len, len(item_list))
                    batch_size = len(feat_list)
                    seq_len = len(feat_list[0][k])
                    padded_batch = torch.zeros((batch_size, seq_len, max_len), dtype=torch.long)
                    for i, d in enumerate(feat_list):
                        for j, item_list in enumerate(d[k]):
                            if isinstance(item_list, list):
                                trunc_len = min(len(item_list), max_len)
                                padded_batch[i, j, :trunc_len] = torch.tensor(item_list[:trunc_len], dtype=torch.long)
                    batched_feat[k] = padded_batch
            return batched_feat

        batched_seq_feat = batch_features(seq_feat)
        batched_pos_feat = batch_features(pos_feat)
        batched_neg_feat = batch_features(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, batched_seq_feat, batched_pos_feat, batched_neg_feat


class MyTestDataset(MyDataset):
    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if isinstance(feat_value, list):
                processed_feat[feat_id] = [v if not isinstance(v, str) else 0 for v in feat_value]
            elif isinstance(feat_value, str):
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)
        ts,user_sequence = self._add_time_features(user_sequence)

        user_id = ""
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                user_id = u if isinstance(u, str) else self.indexer_u_rev.get(u, "")
                u_reid = 0 if isinstance(u, str) else u
                if user_feat:
                    ext_user_sequence.insert(0, (u_reid, self._process_cold_start_feat(user_feat), 2))
            if i and item_feat:
                i_reid = 0 if i > self.itemnum else i
                ext_user_sequence.append((i_reid, self._process_cold_start_feat(item_feat), 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = [self.feature_default_value] * (self.maxlen + 1)

        idx = self.maxlen
        for record_tuple in reversed(ext_user_sequence):
            if idx < 0:
                break
            i, feat, type_ = record_tuple
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = self.fill_missing_feat(feat, i)
            idx -= 1

        seq_feat_tensors = self._features_to_tensors(seq_feat)
        return seq, token_type, seq_feat_tensors, user_id

    def __len__(self):
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))

        def batch_features(feat_list):
            batched_feat = {}
            if not feat_list:
                return batched_feat
            all_keys = feat_list[0].keys()
            for k in all_keys:
                if isinstance(feat_list[0][k], torch.Tensor):
                    batched_feat[k] = torch.stack([d[k] for d in feat_list])
                else:
                    max_len = 0
                    for d in feat_list:
                        for item_list in d[k]:
                            if isinstance(item_list, list):
                                max_len = max(max_len, len(item_list))
                    batch_size = len(feat_list)
                    seq_len = len(feat_list[0][k])
                    padded_batch = torch.zeros((batch_size, seq_len, max_len), dtype=torch.long)
                    for i, d in enumerate(feat_list):
                        for j, item_list in enumerate(d[k]):
                            if isinstance(item_list, list):
                                trunc_len = min(len(item_list), max_len)
                                padded_batch[i, j, :trunc_len] = torch.tensor(item_list[:trunc_len], dtype=torch.long)
                    batched_feat[k] = padded_batch
            return batched_feat

        batched_seq_feat = batch_features(seq_feat)
        return seq, token_type, batched_seq_feat, user_id


def save_emb(emb, save_path):
    num_points, num_dimensions = emb.shape
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)

def load_mm_emb(mm_path, feat_ids):
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
                print(f"transfer error: {e}")
        else: # feat_id == '81'
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict