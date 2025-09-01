from pathlib import Path
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from dataset import TemporalFeatureExtractor, save_emb


class TemporalSENet(torch.nn.Module):
    """融合时间特征的 SENet 模块"""
    def __init__(self, hidden_units, reduction_ratio=16):
        super().__init__()
        self.hidden_units = hidden_units
        self.reduction_ratio = reduction_ratio

        # 离散时间特征嵌入
        self.temporal_embeddings = torch.nn.ModuleDict({
            'hour_emb': torch.nn.Embedding(24, hidden_units),
            'day_of_week_emb': torch.nn.Embedding(7, hidden_units),
            'day_of_month_emb': torch.nn.Embedding(32, hidden_units),
            'month_emb': torch.nn.Embedding(13, hidden_units),
            'is_weekend_emb': torch.nn.Embedding(2, hidden_units),
        })

        # 连续时间特征映射
        self.cyclic_transform = torch.nn.Linear(6, hidden_units)
        self.relative_transform = torch.nn.Linear(3, hidden_units)

    # 其余实现与原文件一致，此处省略 …
    def forward(self, feature_list, temporal_features_dict=None):
        if not feature_list:
            return torch.zeros(1, 1, self.hidden_units)

        batch_size, seq_len = feature_list[0].shape[:2]
        device = feature_list[0].device

        if temporal_features_dict is not None:
            temporal_emb_list = []
            discrete_features = ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend']
            for feat_name in discrete_features:
                if feat_name in temporal_features_dict:
                    feat_value = temporal_features_dict[feat_name]
                    feat_tensor = torch.tensor([feat_value], device=device, dtype=torch.long)
                    feat_tensor = feat_tensor.expand(batch_size, seq_len)
                    temporal_emb_list.append(self.temporal_embeddings[f'{feat_name}_emb'](feat_tensor))

            cyclic_names = ['time_of_day_sin', 'time_of_day_cos', 'day_of_week_sin',
                            'day_of_week_cos', 'month_sin', 'month_cos']
            if all(feat in temporal_features_dict for feat in cyclic_names):
                values = [temporal_features_dict[feat] for feat in cyclic_names]
                cyclic_tensor = torch.tensor(values, device=device, dtype=torch.float)
                cyclic_tensor = cyclic_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                temporal_emb_list.append(self.cyclic_transform(cyclic_tensor))

            relative_names = ['time_since_last', 'time_since_last_log', 'time_since_last_hours']
            if all(feat in temporal_features_dict for feat in relative_names):
                values = [temporal_features_dict[feat] for feat in relative_names]
                relative_tensor = torch.tensor(values, device=device, dtype=torch.float)
                relative_tensor = relative_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                temporal_emb_list.append(self.relative_transform(relative_tensor))

            all_features = feature_list + temporal_emb_list
        else:
            all_features = feature_list

        if len(all_features) == 1:
            return all_features[0]

        stacked = torch.stack(all_features, dim=2)
        num_features = len(all_features)
        squeezed = stacked.mean(dim=1, keepdim=True).mean(dim=-1)

        reduced_dim = max(1, num_features // self.reduction_ratio)
        if not hasattr(self, f'fc1_{num_features}'):
            setattr(self, f'fc1_{num_features}', torch.nn.Linear(num_features, reduced_dim).to(device))
            setattr(self, f'fc2_{num_features}', torch.nn.Linear(reduced_dim, num_features).to(device))
        fc1 = getattr(self, f'fc1_{num_features}')
        fc2 = getattr(self, f'fc2_{num_features}')

        excitation = torch.sigmoid(fc2(torch.relu(fc1(squeezed)))).unsqueeze(-1)
        weighted = stacked * excitation.expand_as(stacked)
        fused = weighted.sum(dim=2)
        return fused


class RoPE(torch.nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.shape[-2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class HSTU(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        self.pointwise_proj = torch.nn.Linear(hidden_units, 4 * hidden_units)
        self.rope = RoPE(self.head_dim)
        self.layer_norm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        self.pointwise_transform = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.size()
        residual = x
        proj = self.pointwise_proj(x)
        U, V, Q, K = torch.chunk(proj, 4, dim=-1)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        Q, K = self.rope(Q, K, L)

        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))
        attn_weights = torch.sigmoid(scores)
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
        attn_out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, L, self.hidden_units)
        attn_out = self.layer_norm(attn_out)

        gated = F.silu(U) * attn_out
        out = self.dropout(self.pointwise_transform(gated))
        return residual + out


class BaselineModel(torch.nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen
        self.temp = args.temperature

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        self.continual_transform = torch.nn.ModuleDict()

        # ✅MOD 去掉内部再次提取相对时间的逻辑，完全依赖 dataset 的 220-222
        self.temporal_extractor = TemporalFeatureExtractor()
        self.item_senet = TemporalSENet(args.hidden_units)
        self.user_senet = TemporalSENet(args.hidden_units)
        self.final_senet = TemporalSENet(args.hidden_units)

        self._init_feat_info(feat_statistics, feat_types)
        self.userdnn = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.itemdnn = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.hstu_layers = torch.nn.ModuleList([
            HSTU(args.hidden_units, args.num_heads, args.dropout_rate)
            for _ in range(args.num_blocks)
        ])

        # 初始化 embedding 层
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)
        for k in self.USER_CONTINUAL_FEAT + self.ITEM_CONTINUAL_FEAT:
            self.continual_transform[k] = torch.nn.Linear(1, args.hidden_units)

    def _init_feat_info(self, feat_statistics, feat_types):
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        self.ITEM_EMB_FEAT = {k: feat_statistics[k] for k in feat_types['item_emb']}

    def feat2emb(self, seq, feature_tensors, mask=None, include_user=False, timestamps=None):
        """把离散/连续/embedding 特征全部映射到 hidden_units；时间特征由 dataset 提供"""
        seq = seq.to(self.dev)

        # ✅MOD 不再调用 extract_relative_time_features，直接用 220-222 等
        temporal_features = None
        if timestamps is not None and len(timestamps) > 0:
            # 这里 timestamps 是 list[list[int]]，如 dataset 返回
            current_ts = timestamps[-1] if isinstance(timestamps[-1], int) else timestamps[-1][-1]
            time_feats = self.temporal_extractor.extract_time_features(current_ts)
            temporal_features = time_feats[0] if isinstance(time_feats, dict) else {}

        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_emb = self.user_emb(user_mask * seq)
            item_emb = self.item_emb(item_mask * seq)
            item_feat_list, user_feat_list = [item_emb], [user_emb]
        else:
            item_emb = self.item_emb(seq)
            item_feat_list = [item_emb]

        # 处理 item 特征
        for k in self.ITEM_SPARSE_FEAT:
            item_feat_list.append(self.sparse_emb[k](feature_tensors[k].to(self.dev)))
        for k in self.ITEM_ARRAY_FEAT:
            item_feat_list.append(self.sparse_emb[k](feature_tensors[k].to(self.dev)).sum(2))
        for k in self.ITEM_CONTINUAL_FEAT:
            item_feat_list.append(self.continual_transform[k](feature_tensors[k].to(self.dev).unsqueeze(-1)))
        for k in self.ITEM_EMB_FEAT:
            item_feat_list.append(self.emb_transform[k](feature_tensors[k].to(self.dev)))

        all_item_emb = self.item_senet(item_feat_list, temporal_features)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))

        if include_user:
            for k in self.USER_SPARSE_FEAT:
                user_feat_list.append(self.sparse_emb[k](feature_tensors[k].to(self.dev)))
            for k in self.USER_ARRAY_FEAT:
                user_feat_list.append(self.sparse_emb[k](feature_tensors[k].to(self.dev)).sum(2))
            for k in self.USER_CONTINUAL_FEAT:
                user_feat_list.append(self.continual_transform[k](feature_tensors[k].to(self.dev).unsqueeze(-1)))

            all_user_emb = self.user_senet(user_feat_list, temporal_features)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            final_list = [all_item_emb, all_user_emb]
            seqs_emb = self.final_senet(final_list, temporal_features)
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature_tensors, timestamps=None):
        seqs = self.feat2emb(log_seqs, seq_feature_tensors, mask=mask, include_user=True, timestamps=timestamps)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)

        attn_mask = torch.tril(torch.ones((self.maxlen + 1, self.maxlen + 1), dtype=torch.bool, device=self.dev))
        attn_mask = attn_mask.unsqueeze(0).expand(log_seqs.shape[0], -1, -1)
        mask_pad = (mask != 0).to(self.dev)
        attn_mask = attn_mask & mask_pad.unsqueeze(1)

        for layer in self.hstu_layers:
            seqs = layer(seqs, attn_mask=attn_mask)
        return self.last_layernorm(seqs)

    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type,
                seq_feat, pos_feat, neg_feat, timestamps=None):
        log_feats = self.log2feats(user_item, mask, seq_feat, timestamps=timestamps)
        loss_mask = (next_mask == 1).to(self.dev)
        
        # 为正负样本也传递时间戳信息
        pos_embs = self.feat2emb(pos_seqs, pos_feat, include_user=False, timestamps=timestamps)
        neg_embs = self.feat2emb(neg_seqs, neg_feat, include_user=False, timestamps=timestamps)
        
        return log_feats, pos_embs, neg_embs, loss_mask
    
    def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask, writer):
        hidden_size = neg_embs.size(-1)
        
        seq_embs_normalized = F.normalize(seq_embs, p=2, dim=-1)
        pos_embs_normalized = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs_normalized = F.normalize(neg_embs, p=2, dim=-1)
        
        pos_logits = (seq_embs_normalized * pos_embs_normalized).sum(dim=-1).unsqueeze(-1)
        writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item())
        
        neg_embedding_all = neg_embs_normalized.view(-1, hidden_size)
        neg_logits = torch.matmul(seq_embs_normalized, neg_embedding_all.transpose(-1, -2))
        writer.add_scalar("Model/nce_neg_logits", neg_logits.mean().item())
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        
        masked_logits = logits[loss_mask.bool()] / self.temp
        labels = torch.zeros(masked_logits.size(0), device=masked_logits.device, dtype=torch.long)
        
        loss = F.cross_entropy(masked_logits, labels)
        return loss

    def predict(self, log_seqs, seq_feature, mask, timestamps=None):
        log_feats = self.log2feats(log_seqs, mask, seq_feature, timestamps=timestamps)
        final_feat = log_feats[:, -1, :]
        return F.normalize(final_feat, p=2, dim=-1)

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024, timestamps=None):
        all_embs = []
        for start in range(0, len(item_ids), batch_size):
            end = min(start + batch_size, len(item_ids))
            item_seq = torch.tensor(item_ids[start:end], device=self.dev).unsqueeze(0)

            batch_feat = feat_dict[start:end]
            batch_ts = timestamps[start:end] if timestamps else None

            feat_tensors = {}
            keys = set(k for d in batch_feat for k in d.keys())
            for k in keys:
                if any(isinstance(d.get(k), list) for d in batch_feat):
                    max_len = max(len(d.get(k, [])) for d in batch_feat)
                    data = np.zeros((len(batch_feat), max_len), dtype=np.int64)
                    for i, d in enumerate(batch_feat):
                        val = d.get(k, [])
                        data[i, :len(val)] = val
                    feat_tensors[k] = torch.from_numpy(data).unsqueeze(0)
                elif any(isinstance(d.get(k), np.ndarray) for d in batch_feat):
                    data = np.stack([d.get(k) for d in batch_feat])
                    feat_tensors[k] = torch.from_numpy(data).float().unsqueeze(0)
                else:
                    data = [d.get(k, 0) for d in batch_feat]
                    feat_tensors[k] = torch.tensor(data, dtype=torch.long).unsqueeze(0)

            embs = self.feat2emb(item_seq, feat_tensors, include_user=False, timestamps=batch_ts).squeeze(0)
            embs = F.normalize(embs, p=2, dim=-1)
            all_embs.append(embs.detach().cpu().numpy().astype(np.float32))

        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))