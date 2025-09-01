from pathlib import Path
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataset import TemporalFeatureExtractor

from dataset import save_emb
class TemporalSENet(torch.nn.Module):
    """融合时间特征的SENet模块"""
    def __init__(self, hidden_units, reduction_ratio=16):
        super(TemporalSENet, self).__init__()
        self.hidden_units = hidden_units
        self.reduction_ratio = reduction_ratio
        
        # 时间特征嵌入层
        self.temporal_embeddings = torch.nn.ModuleDict({
            'hour_emb': torch.nn.Embedding(24, hidden_units),
            'day_of_week_emb': torch.nn.Embedding(7, hidden_units),
            'day_of_month_emb': torch.nn.Embedding(32, hidden_units),
            'month_emb': torch.nn.Embedding(13, hidden_units),  # 1-12 + padding
            'is_weekend_emb': torch.nn.Embedding(2, hidden_units),
        })
        
        # 连续时间特征变换
        self.cyclic_transform = torch.nn.Linear(6, hidden_units)  # 6个sin/cos特征
        self.relative_transform = torch.nn.Linear(3, hidden_units)  # 3个相对时间特征
        
    def forward(self, feature_list, temporal_features_dict=None):
        """
        feature_list: 原有特征列表
        temporal_features_dict: 时间特征字典
        """
        if not feature_list:
            return torch.zeros(1, 1, self.hidden_units)
            
        batch_size, seq_len = feature_list[0].shape[:2]
        device = feature_list[0].device
        
        # 如果有时间特征，添加到特征列表
        if temporal_features_dict is not None:
            temporal_emb_list = []
            
            # 离散时间特征嵌入
            discrete_features = ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend']
            for feat_name in discrete_features:
                if feat_name in temporal_features_dict:
                    feat_value = temporal_features_dict[feat_name]
                    feat_tensor = torch.tensor([feat_value], device=device, dtype=torch.long)
                    feat_tensor = feat_tensor.expand(batch_size, seq_len)
                    temporal_emb_list.append(self.temporal_embeddings[f'{feat_name}_emb'](feat_tensor))
            
            # 周期性特征
            cyclic_feature_names = ['time_of_day_sin', 'time_of_day_cos', 'day_of_week_sin', 
                                  'day_of_week_cos', 'month_sin', 'month_cos']
            if all(feat in temporal_features_dict for feat in cyclic_feature_names):
                cyclic_values = [temporal_features_dict[feat] for feat in cyclic_feature_names]
                cyclic_tensor = torch.tensor(cyclic_values, device=device, dtype=torch.float)
                cyclic_tensor = cyclic_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                temporal_emb_list.append(self.cyclic_transform(cyclic_tensor))
            
            # 相对时间特征
            relative_feature_names = ['time_since_last', 'time_since_last_log', 'time_since_last_hours']
            if all(feat in temporal_features_dict for feat in relative_feature_names):
                relative_values = [temporal_features_dict[feat] for feat in relative_feature_names]
                relative_tensor = torch.tensor(relative_values, device=device, dtype=torch.float)
                relative_tensor = relative_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                temporal_emb_list.append(self.relative_transform(relative_tensor))
            
            # 合并所有特征
            all_features = feature_list + temporal_emb_list
        else:
            all_features = feature_list
        
        if len(all_features) == 1:
            return all_features[0]
            
        # Stack所有特征
        stacked_features = torch.stack(all_features, dim=2)  # [B, L, num_features, H]
        num_features = len(all_features)
        
        # SENet注意力机制
        # Squeeze: 全局平均池化
        squeezed = stacked_features.mean(dim=1, keepdim=True)  # [B, 1, num_features, H]
        squeezed = squeezed.mean(dim=-1)  # [B, 1, num_features]
        
        # Excitation: 学习特征权重
        reduced_dim = max(1, num_features // self.reduction_ratio)
        
        # 动态创建FC层（避免参数不在同一设备的问题）
        if not hasattr(self, f'fc1_{num_features}'):
            setattr(self, f'fc1_{num_features}', torch.nn.Linear(num_features, reduced_dim).to(device))
            setattr(self, f'fc2_{num_features}', torch.nn.Linear(reduced_dim, num_features).to(device))
        
        fc1 = getattr(self, f'fc1_{num_features}')
        fc2 = getattr(self, f'fc2_{num_features}')
        
        excitation = F.relu(fc1(squeezed))  # [B, 1, reduced_dim]
        excitation = torch.sigmoid(fc2(excitation))  # [B, 1, num_features]
        excitation = excitation.unsqueeze(-1)  # [B, 1, num_features, 1]
        
        # Scale: 应用权重
        weighted_features = stacked_features * excitation.expand_as(stacked_features)
        
        # 融合：加权求和
        fused_features = weighted_features.sum(dim=2)  # [B, L, H]
        
        return fused_features


class RoPE(torch.nn.Module):
    """Rotary Position Embedding implementation"""
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super(RoPE, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute theta values
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin tables
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
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
    """Hierarchical Sequential Transduction Unit (HSTU) Layer"""
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(HSTU, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        
        # Pointwise Projection (Equation 1 in paper)
        self.pointwise_proj = torch.nn.Linear(hidden_units, 4 * hidden_units)
        
        # RoPE for positional encoding
        self.rope = RoPE(self.head_dim)
        
        # Layer normalization
        self.layer_norm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        
        # Pointwise Transformation (Equation 3 in paper)
        self.pointwise_transform = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.size()
        residual = x
        
        # Pointwise Projection
        proj = self.pointwise_proj(x)  # [B, L, 4*D]
        U, V, Q, K = torch.chunk(proj, 4, dim=-1)  # Each: [B, L, D]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        Q_rope, K_rope = self.rope(Q, K, seq_len)
        
        # Spatial Aggregation
        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(Q_rope, K_rope.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))
        
        # Pointwise aggregated attention
        attn_weights = torch.sigmoid(scores)
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        
        # Apply layer normalization after spatial aggregation
        attn_output = self.layer_norm(attn_output)
        
        # Element-wise gating with U
        U = U.view(batch_size, seq_len, self.hidden_units)
        gated_output = F.silu(U) * attn_output
        
        # Pointwise Transformation
        output = self.pointwise_transform(gated_output)
        output = self.dropout(output)
        
        # Residual connection
        output = residual + output
        
        return output


class BaselineModel(torch.nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super(BaselineModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.temp = args.temperature

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        self.continual_transform = torch.nn.ModuleDict()
        
        # HSTU layers
        self.hstu_layers = torch.nn.ModuleList()
        
        # 时间特征提取器
        self.temporal_extractor = TemporalFeatureExtractor()
        
        # SENet模块用于特征融合
        self.item_senet = TemporalSENet(args.hidden_units)
        self.user_senet = TemporalSENet(args.hidden_units)
        self.final_senet = TemporalSENet(args.hidden_units)
        
        self._init_feat_info(feat_statistics, feat_types)

        # 计算融合后的特征维度（不再需要concat所有特征）
        # 因为SENet会将所有特征融合到hidden_units维度
        self.userdnn = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.itemdnn = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # Initialize HSTU layers
        for _ in range(args.num_blocks):
            self.hstu_layers.append(HSTU(args.hidden_units, args.num_heads, args.dropout_rate))

        # Initialize sparse embeddings
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)
        
        # Initialize continual feature transforms
        for k in self.ITEM_CONTINUAL_FEAT:
            self.continual_transform[k] = torch.nn.Linear(1, args.hidden_units)
        for k in self.USER_CONTINUAL_FEAT:
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
        """融合时间特征的嵌入方法"""
        seq = seq.to(self.dev)
        
        # 提取时间特征
        temporal_features = None
        if timestamps is not None:
            # 只处理当前序列位置的时间特征（通常是最后一个位置）
            current_ts = timestamps[-1] if isinstance(timestamps, list) else timestamps
            if current_ts is not None:
                time_feats = self.temporal_extractor.extract_time_features(current_ts)
                if timestamps is not None:
                    temporal_features = None
                    if timestamps is not None and len(timestamps) > 0:
                        current_ts = timestamps[-1]
                        if current_ts is not None:
                            time_feats = self.temporal_extractor.extract_time_features([current_ts])
                            relative_feats = self.temporal_extractor.extract_relative_time_features(timestamps)
                            temporal_features = {}
                            temporal_features.update(time_feats[0])
                            temporal_features.update(relative_feats[len(timestamps) - 1])
                else:
                    relative_feats = torch.zeros(seq.shape[0], self.hidden_units).to(seq.device)

                # 合并时间特征
                temporal_features = {}
                if 0 in time_feats:
                    temporal_features.update(time_feats[0])
                if len(relative_feats) > 0:
                    last_idx = len(relative_feats) - 1
                    temporal_features.update(relative_feats[last_idx])
        
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # 处理item特征
        feature_groups = [
            (self.ITEM_SPARSE_FEAT, item_feat_list, 'sparse'),
            (self.ITEM_ARRAY_FEAT, item_feat_list, 'array'),
            (self.ITEM_CONTINUAL_FEAT, item_feat_list, 'continual'),
        ]

        for feat_dict, feat_list, feat_type in feature_groups:
            for k in feat_dict:
                tensor_feature = feature_tensors[k].to(self.dev)
                if feat_type == 'sparse':
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type == 'array':
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type == 'continual':
                    continual_emb = self.continual_transform[k](tensor_feature.unsqueeze(-1))
                    feat_list.append(continual_emb)

        # 处理embedding特征
        for k in self.ITEM_EMB_FEAT:
            tensor_feature = feature_tensors[k].to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # 使用SENet融合item特征
        all_item_emb = self.item_senet(item_feat_list, temporal_features)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))

        if include_user:
            # 处理用户特征
            user_feature_groups = [
                (self.USER_SPARSE_FEAT, user_feat_list, 'sparse'),
                (self.USER_ARRAY_FEAT, user_feat_list, 'array'),
                (self.USER_CONTINUAL_FEAT, user_feat_list, 'continual'),
            ]
            
            for feat_dict, feat_list, feat_type in user_feature_groups:
                for k in feat_dict:
                    tensor_feature = feature_tensors[k].to(self.dev)
                    if feat_type == 'sparse':
                        feat_list.append(self.sparse_emb[k](tensor_feature))
                    elif feat_type == 'array':
                        feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                    elif feat_type == 'continual':
                        continual_emb = self.continual_transform[k](tensor_feature.unsqueeze(-1))
                        feat_list.append(continual_emb)
            
            # 使用SENet融合用户特征
            all_user_emb = self.user_senet(user_feat_list, temporal_features)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            
            # 最终融合item和user特征
            final_feat_list = [all_item_emb, all_user_emb]
            seqs_emb = self.final_senet(final_feat_list, temporal_features)
        else:
            seqs_emb = all_item_emb

        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature_tensors, timestamps=None):
        """添加时间戳参数"""
        seqs = self.feat2emb(log_seqs, seq_feature_tensors, mask=mask, include_user=True, timestamps=timestamps)
        seqs *= self.item_emb.embedding_dim**0.5
        
        seqs = self.emb_dropout(seqs)

        # Create causal attention mask
        attention_mask = torch.tril(torch.ones((self.maxlen + 1, self.maxlen + 1), dtype=torch.bool, device=self.dev))
        attention_mask = attention_mask.unsqueeze(0).expand(log_seqs.shape[0], -1, -1)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask & attention_mask_pad.unsqueeze(1)

        # Pass through HSTU layers
        for hstu_layer in self.hstu_layers:
            seqs = hstu_layer(seqs, attn_mask=attention_mask)

        return self.last_layernorm(seqs)

    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feat, pos_feat, neg_feat, timestamps=None):
        """添加时间戳参数到forward方法"""
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
        """添加时间戳参数到predict方法"""
        log_feats = self.log2feats(log_seqs, mask, seq_feature, timestamps=timestamps)
        final_feat = log_feats[:, -1, :]
        final_feat = F.normalize(final_feat, p=2, dim=-1)
        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024, timestamps=None):
        """添加时间戳参数到save_item_emb方法"""
        all_embs = []
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            
            batch_feat_list = feat_dict[start_idx:end_idx]
            batch_timestamps = timestamps[start_idx:end_idx] if timestamps else None
            
            item_feat_tensors = {}
            all_keys = set(k for d in batch_feat_list for k in d.keys())
            
            for k in all_keys:
                if any(isinstance(d.get(k), list) for d in batch_feat_list):
                    max_len = max(len(d.get(k,[])) for d in batch_feat_list)
                    data = np.zeros((len(batch_feat_list), max_len), dtype=np.int64)
                    for i, d in enumerate(batch_feat_list):
                        val = d.get(k, [])
                        data[i, :len(val)] = val
                    item_feat_tensors[k] = torch.from_numpy(data).unsqueeze(0)
                elif any(isinstance(d.get(k), np.ndarray) for d in batch_feat_list):
                    data = np.stack([d.get(k) for d in batch_feat_list])
                    item_feat_tensors[k] = torch.from_numpy(data).float().unsqueeze(0)
                else:
                    data = [d.get(k, 0) for d in batch_feat_list]
                    item_feat_tensors[k] = torch.tensor(data, dtype=torch.long).unsqueeze(0)

            batch_emb = self.feat2emb(item_seq, item_feat_tensors, include_user=False, timestamps=batch_timestamps).squeeze(0)
            batch_emb = F.normalize(batch_emb, p=2, dim=-1)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))


# # 数据预处理辅助函数
# def preprocess_temporal_batch(batch_data):
#     """
#     预处理包含时间戳的批次数据
#     batch_data: 格式为 [user_id, item_id, user_features, item_features, label, timestamp] 的列表
#     """
#     timestamps = []
#     processed_features = []
    
#     for record in batch_data:
#         user_id, item_id, user_features, item_features, label, timestamp = record
#         timestamps.append(timestamp)
        
#         # 提取时间特征
#         if timestamp is not None:
#             extractor = TemporalFeatureExtractor()
#             time_features = extractor.extract_time_features([timestamp])[0]
            
#             # 如果是序列的一部分，也提取相对时间特征
#             if len(timestamps) > 1:
#                 relative_features = extractor.extract_relative_time_features(timestamps)
#                 time_features.update(relative_features[len(timestamps)-1])
#         else:
#             time_features = {
#                 'hour': 0, 'day_of_week': 0, 'day_of_month': 0,
#                 'month': 0, 'is_weekend': 0,
#                 'time_of_day_sin': 0.0, 'time_of_day_cos': 1.0,
#                 'day_of_week_sin': 0.0, 'day_of_week_cos': 1.0,
#                 'month_sin': 0.0, 'month_cos': 1.0,
#                 'time_since_last': 0.0, 'time_since_last_log': 0.0,
#                 'time_since_last_hours': 0.0
#             }
        
#         # 将时间特征合并到item_features
#         if item_features is None:
#             item_features = {}
        
#         # 添加时间特征，使用temporal_前缀避免冲突
#         enhanced_features = item_features.copy()
#         for k, v in time_features.items():
#             enhanced_features[f'temporal_{k}'] = v
            
#         processed_features.append([user_id, item_id, user_features, enhanced_features, label, timestamp])
    
#     return processed_features, timestamps


# # 训练时的使用示例
# class TemporalTrainer:
#     """支持时间特征的训练器示例"""
    
#     def __init__(self, model, args):
#         self.model = model
#         self.args = args
    
#     def train_step(self, batch_data):
#         """单步训练，支持时间特征"""
#         # 预处理时间特征
#         processed_batch, timestamps = preprocess_temporal_batch(batch_data)
        
#         # 提取序列数据
#         user_items = []
#         pos_seqs = []
#         neg_seqs = []
#         masks = []
        
#         # 构建特征张量（这里需要根据你的实际数据加载逻辑调整）
#         seq_feat = {}
#         pos_feat = {}
#         neg_feat = {}
        
#         # ... 数据构建逻辑 ...
        
#         # 调用模型forward，传入时间戳
#         log_feats, pos_embs, neg_embs, loss_mask = self.model(
#             user_items, pos_seqs, neg_seqs, masks, 
#             next_mask=None, next_action_type=None,
#             seq_feat=seq_feat, pos_feat=pos_feat, neg_feat=neg_feat,
#             timestamps=timestamps  # 新增时间戳参数
#         )
        
#         return log_feats, pos_embs, neg_embs, loss_mask

