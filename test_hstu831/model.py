from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb


# class RotaryPositionalEncoding(torch.nn.Module):
#     """
#     Rotary Positional Encoding (RoPE) implementation
#     """
#     def __init__(self, dim, max_seq_len=512):
#         super().__init__()
#         self.dim = dim
#         self.max_seq_len = max_seq_len
        
#         # Create frequency tensor
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer('inv_freq', inv_freq)
        
#         # Create position tensor
#         position = torch.arange(max_seq_len, dtype=torch.float)
#         self.register_buffer('position', position)
        
#     def forward(self, x):
#         """
#         Apply rotary encoding to input tensor
        
#         Args:
#             x: input tensor of shape [batch_size, seq_len, dim]
        
#         Returns:
#             rotary encoded tensor
#         """
#         seq_len = x.size(1)
        
#         # Calculate frequencies for each position
#         freqs = torch.outer(self.position[:seq_len], self.inv_freq)
        
#         # Create rotation matrix
#         cos_freqs = torch.cos(freqs)  # [seq_len, dim//2]
#         sin_freqs = torch.sin(freqs)  # [seq_len, dim//2]
        
#         # Apply rotation to input
#         x_rot = self.rotate_half(x)
        
#         # Apply rotary encoding
#         x = x * cos_freqs.unsqueeze(0) + x_rot * sin_freqs.unsqueeze(0)
        
#         return x
    
#     def rotate_half(self, x):
#         """
#         Rotate half of the dimensions for rotary encoding
#         """
#         x1, x2 = x.chunk(2, dim=-1)
#         return torch.cat([-x2, x1], dim=-1)
class RotaryPositionalEncoding(torch.nn.Module):
    """
    修正维度匹配问题的 Rotary Positional Encoding (RoPE) 实现
    """
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim  # 这里的 dim 是单头注意力维度（head_dim）
        self.max_seq_len = max_seq_len
        
        # 1. 生成频率张量（原始逻辑不变，频率维度为 dim//2）
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 2. 生成位置张量（原始逻辑不变）
        position = torch.arange(max_seq_len, dtype=torch.float)
        self.register_buffer('position', position)
        
    def forward(self, x):
        """
        应用旋转编码到输入张量
        Args:
            x: 输入张量，形状为 [batch_size, num_heads, seq_len, head_dim]
        Returns:
            旋转编码后的张量（与输入形状一致）
        """
        seq_len = x.size(2)  # 注意：输入形状变为 [B, H, S, D]，seq_len 在第 2 维
        
        # 3. 计算每个位置的频率（形状：[seq_len, dim//2]）
        freqs = torch.outer(self.position[:seq_len], self.inv_freq)
        
        # 4. 生成旋转矩阵的 cos/sin 分量（形状：[seq_len, dim//2]）
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        
        # 5. 关键修正：将 cos/sin 扩展到与输入最后一维（head_dim）匹配
        # 因为输入最后一维会被拆分为两半，所以需要将 cos/sin 重复 2 次（dim//2 * 2 = dim）
        cos_freqs = cos_freqs.unsqueeze(-1).repeat(1, 1, 2)  # [seq_len, dim//2, 2] → 展平后 [seq_len, dim]
        sin_freqs = sin_freqs.unsqueeze(-1).repeat(1, 1, 2)
        cos_freqs = cos_freqs.view(seq_len, self.dim)  # [seq_len, dim]
        sin_freqs = sin_freqs.view(seq_len, self.dim)  # [seq_len, dim]
        
        # 6. 对输入进行半维度旋转（原始逻辑不变）
        x_rot = self.rotate_half(x)
        
        # 7. 应用旋转编码：注意输入形状为 [B, H, S, D]，需调整 cos/sin 的维度
        # 给 cos/sin 添加 batch 和 head 维度的 singleton 维度，确保广播兼容
        x = x * cos_freqs.unsqueeze(0).unsqueeze(0)  # [B,H,S,D] * [1,1,S,D] → 广播后匹配
        x_rot = x_rot * sin_freqs.unsqueeze(0).unsqueeze(0)
        x = x + x_rot
        
        return x
    
    def rotate_half(self, x):
        """
        将输入的最后一维拆分为两半，并旋转后一半（修正输入形状适配）
        输入 x 形状：[batch_size, num_heads, seq_len, head_dim]
        输出形状：与输入一致
        """
        x1, x2 = x.chunk(2, dim=-1)  # 最后一维拆分为两半（各为 head_dim//2）
        return torch.cat([-x2, x1], dim=-1)  # 旋转后合并，恢复为 head_dim 维度


class HSTUAttention(torch.nn.Module):
    """
    Hierarchical Spatial-Temporal Unit (HSTU) Attention mechanism
    Based on ICML 2024 paper "Actions Speak Louder than Words"
    """
    def __init__(self, hidden_units, num_heads, dropout_rate, max_seq_len=512):
        super(HSTUAttention, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len
        
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)
        
        # Rotary Positional Encoding
        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)
        
        # Hierarchical temporal aggregation
        self.temporal_scale = 4  # Number of temporal scales
        self.temporal_weights = torch.nn.Parameter(torch.ones(self.temporal_scale) / self.temporal_scale)
        
        # Spatial feature fusion
        self.spatial_gate = torch.nn.Linear(hidden_units, hidden_units)
        self.spatial_norm = torch.nn.LayerNorm(hidden_units)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def hierarchical_temporal_aggregation(self, x):
        """
        Apply hierarchical temporal aggregation at multiple scales
        Input x shape: [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        temporal_features = []
        
        for scale in range(self.temporal_scale):
            # Calculate pooling size for this scale
            pool_size = max(1, seq_len // (2 ** scale))
            
            # Apply adaptive pooling to get fixed-size representation
            if scale == 0:
                # Original scale
                scale_feat = x
            else:
                # Downsampled scale - reshape for pooling
                # Reshape to [batch_size * num_heads, head_dim, seq_len] for pooling
                x_reshaped = x.view(batch_size * num_heads, head_dim, seq_len)
                scale_feat = F.adaptive_avg_pool1d(x_reshaped, pool_size)
                scale_feat = scale_feat.view(batch_size, num_heads, head_dim, pool_size).transpose(-2, -1)
            
            # Upsample back to original length if needed
            if scale_feat.size(2) < seq_len:
                # Reshape for interpolation: [batch_size * num_heads, head_dim, current_seq_len]
                scale_feat_reshaped = scale_feat.transpose(-2, -1).contiguous().view(batch_size * num_heads, head_dim, -1)
                scale_feat_interp = F.interpolate(scale_feat_reshaped, size=seq_len, mode='linear', align_corners=False)
                scale_feat = scale_feat_interp.view(batch_size, num_heads, head_dim, seq_len).transpose(-2, -1)
            
            temporal_features.append(scale_feat)
        
        # Weighted aggregation across temporal scales
        temporal_features = torch.stack(temporal_features, dim=0)  # [temporal_scale, batch_size, num_heads, seq_len, head_dim]
        weights = F.softmax(self.temporal_weights, dim=0).view(-1, 1, 1, 1, 1)
        aggregated = (temporal_features * weights).sum(dim=0)
        
        return aggregated
    
    def spatial_feature_fusion(self, query, key, value):
        """
        Apply spatial feature fusion with gating mechanism
        Input shapes: [batch_size, num_heads, seq_len, head_dim]
        """
        # Compute spatial attention scores
        spatial_scores = torch.matmul(query, key.transpose(-2, -1))
        spatial_scores = spatial_scores / (self.head_dim ** 0.5)
        
        # Apply spatial gating - need to reshape for gate computation
        batch_size, num_heads, seq_len, head_dim = query.shape
        query_flat = query.view(batch_size, seq_len, num_heads * head_dim)
        gate = torch.sigmoid(self.spatial_gate(query_flat))
        gate = gate.view(batch_size, num_heads, seq_len, head_dim)
        
        # Apply gate to spatial scores
        spatial_scores = spatial_scores * gate.mean(dim=-1, keepdim=True)
        
        return spatial_scores
    
    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()
        
        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply Rotary Positional Encoding
        Q = self.rope(Q)
        K = self.rope(K)
        
        # Hierarchical temporal aggregation (now handles 4D tensors correctly)
        Q = self.hierarchical_temporal_aggregation(Q)
        K = self.hierarchical_temporal_aggregation(K)
        V = self.hierarchical_temporal_aggregation(V)
        
        # Compute attention with spatial fusion
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use efficient scaled dot product attention if available
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, 
                dropout_p=self.dropout_rate if self.training else 0.0, 
                attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None
            )
        else:
            # Manual attention computation
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        
        # Apply spatial normalization
        attn_output = self.spatial_norm(attn_output)
        
        # Final projection
        output = self.out_linear(attn_output)
        
        return output, None


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        output = self.out_linear(attn_output)
        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        return outputs


class BaselineModel(torch.nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super(BaselineModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.temp = args.temperature

        # Embeddings
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        
        # Replace absolute positional encoding with Rotary Positional Encoding
        self.rope = RotaryPositionalEncoding(args.hidden_units, max_seq_len=2 * args.maxlen + 1)
        
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()  # Will use HSTUAttention
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self._init_feat_info(feat_statistics, feat_types)

        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(self.USER_CONTINUAL_FEAT)
        itemdim = args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT)) + len(self.ITEM_CONTINUAL_FEAT) + args.hidden_units * len(self.ITEM_EMB_FEAT)

        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # Use HSTU Attention layers instead of FlashMultiHeadAttention
        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(HSTUAttention(args.hidden_units, args.num_heads, args.dropout_rate, max_seq_len=2 * args.maxlen + 1))
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

        # Feature embeddings
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

    def _init_feat_info(self, feat_statistics, feat_types):
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        self.ITEM_EMB_FEAT = {k: feat_statistics[k] for k in feat_types['item_emb']}

    def feat2emb(self, seq, feature_tensors, mask=None, include_user=False):
        seq = seq.to(self.dev)
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

        feature_groups = [
            (self.ITEM_SPARSE_FEAT, item_feat_list, 'sparse'),
            (self.ITEM_ARRAY_FEAT, item_feat_list, 'array'),
            (self.ITEM_CONTINUAL_FEAT, item_feat_list, 'continual'),
        ]
        if include_user:
            feature_groups.extend([
                (self.USER_SPARSE_FEAT, user_feat_list, 'sparse'),
                (self.USER_ARRAY_FEAT, user_feat_list, 'array'),
                (self.USER_CONTINUAL_FEAT, user_feat_list, 'continual'),
            ])

        for feat_dict, feat_list, feat_type in feature_groups:
            for k in feat_dict:
                tensor_feature = feature_tensors[k].to(self.dev)
                if feat_type == 'sparse':
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type == 'array':
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type == 'continual':
                    feat_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            tensor_feature = feature_tensors[k].to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature_tensors):
        seqs = self.feat2emb(log_seqs, seq_feature_tensors, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim**0.5
        
        # Apply Rotary Positional Encoding instead of absolute positional encoding
        # Note: HSTUAttention handles RoPE internally, so we don't need to apply it here
        
        seqs = self.emb_dropout(seqs)

        attention_mask = torch.tril(torch.ones((self.maxlen + 1, self.maxlen + 1), dtype=torch.bool, device=self.dev))
        attention_mask = attention_mask.unsqueeze(0).expand(log_seqs.shape[0], -1, -1)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        return self.last_layernorm(seqs)

    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feat, pos_feat, neg_feat):
        log_feats = self.log2feats(user_item, mask, seq_feat)
        loss_mask = (next_mask == 1).to(self.dev)
        pos_embs = self.feat2emb(pos_seqs, pos_feat, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feat, include_user=False)
        return log_feats, pos_embs, neg_embs, loss_mask
    
    def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask,writer):
        hidden_size = neg_embs.size(-1)
        
        seq_embs_normalized = F.normalize(seq_embs, p=2, dim=-1)
        pos_embs_normalized = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs_normalized = F.normalize(neg_embs, p=2, dim=-1)
        
        pos_logits = (seq_embs_normalized * pos_embs_normalized).sum(dim=-1).unsqueeze(-1)
        writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item())
    # 
        neg_embedding_all = neg_embs_normalized.view(-1, hidden_size)
        neg_logits = torch.matmul(seq_embs_normalized, neg_embedding_all.transpose(-1, -2))
        writer.add_scalar("Model/nce_neg_logits", neg_logits.mean().item())
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        
        masked_logits = logits[loss_mask.bool()] / self.temp
        labels = torch.zeros(masked_logits.size(0), device=masked_logits.device, dtype=torch.long)
        
        loss = F.cross_entropy(masked_logits, labels)
        return loss

    def predict(self, log_seqs, seq_feature, mask):
        log_feats = self.log2feats(log_seqs, mask, seq_feature)
        final_feat = log_feats[:, -1, :]
        final_feat = F.normalize(final_feat, p=2, dim=-1) # L2 Normalization for consistency
        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        all_embs = []
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            
            batch_feat_list = feat_dict[start_idx:end_idx]
            
            item_feat_tensors = {}
            all_keys = set(k for d in batch_feat_list for k in d.keys())
            
            for k in all_keys:
                if any(isinstance(d.get(k), list) for d in batch_feat_list): # Array feature
                    max_len = max(len(d.get(k,[])) for d in batch_feat_list)
                    data = np.zeros((len(batch_feat_list), max_len), dtype=np.int64)
                    for i, d in enumerate(batch_feat_list):
                        val = d.get(k, [])
                        data[i, :len(val)] = val
                    item_feat_tensors[k] = torch.from_numpy(data).unsqueeze(0)
                elif any(isinstance(d.get(k), np.ndarray) for d in batch_feat_list): # Emb feature
                    data = np.stack([d.get(k) for d in batch_feat_list])
                    item_feat_tensors[k] = torch.from_numpy(data).float().unsqueeze(0)
                else: # Sparse feature
                    data = [d.get(k, 0) for d in batch_feat_list]
                    item_feat_tensors[k] = torch.tensor(data, dtype=torch.long).unsqueeze(0)

            batch_emb = self.feat2emb(item_seq, item_feat_tensors, include_user=False).squeeze(0)
            batch_emb = F.normalize(batch_emb, p=2, dim=-1) # L2 Normalization for consistency
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))