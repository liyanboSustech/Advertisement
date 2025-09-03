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
                        # -*- coding: utf-8 -*-
import json
import pickle
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, args, dataset_type='train'):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type  # 'train', 'valid', or 'test'
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
        
        # 如果是训练模式，进行训练集/验证集划分
        if dataset_type in ['train', 'valid']:
            self._split_train_valid(args.valid_ratio if has