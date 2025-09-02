# model.py
# 改动一览（搜索 # === MOD === 可见）：
# 1. HSTU 内部 attention dropout 彻底删除
# 2. embedding 层 dropout 与 FFN dropout 分离：
#    - emb_dropout_rate（默认 0.3）用于 embedding / sparse / user / item
#    - ffn_dropout_rate（默认 0.1）仅用于 HSTU 内部 FFN
# 3. 新增 SENetFusion 做 user & item 加权融合
# 4. 融合后、进 Transformer 前加 fusion_dropout
# 5. 融合后再加 RoPE（已在 HSTU 内部完成）

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataset import save_emb

# ---------- RoPE ----------
class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def rotate_half(self, x):
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


# ---------- HSTU（attention dropout 已删除） ----------
class HSTU(nn.Module):
    def __init__(self, hidden_units, num_heads, ffn_dropout_rate):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.ffn_dropout_rate = ffn_dropout_rate
        assert hidden_units % num_heads == 0

        self.pointwise_proj = nn.Linear(hidden_units, 4 * hidden_units)
        self.rope = RoPE(self.head_dim)
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.pointwise_transform = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(p=ffn_dropout_rate)  # 仅 FFN 用

    def forward(self, x, attn_mask=None):
        B, L, _ = x.size()
        residual = x
        proj = self.pointwise_proj(x)
        U, V, Q, K = torch.chunk(proj, 4, dim=-1)

        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        Q_rope, K_rope = self.rope(Q, K, L)
        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(Q_rope, K_rope.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

        attn_weights = torch.sigmoid(scores)
        # === MOD: attention dropout 直接删掉 ===
        # attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)

        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.hidden_units)
        attn_out = self.layer_norm(attn_out)

        U = U.view(B, L, self.hidden_units)
        gated = F.silu(U) * attn_out
        out = self.pointwise_transform(gated)
        out = self.dropout(out)
        return residual + out


# ---------- SENet 融合 ----------
class SENetFusion(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, item_emb, user_emb):
        gate = self.fc(item_emb + user_emb)
        return item_emb * gate + user_emb * (1 - gate)


# ---------- BaselineModel ----------
class BaselineModel(nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen
        self.temp = args.temperature

        # === MOD: 两套 dropout ===
        self.emb_dropout_rate = getattr(args, 'emb_dropout_rate', 0.3)
        self.ffn_dropout_rate = getattr(args, 'ffn_dropout_rate', 0.1)
        self.fusion_dropout_rate = getattr(args, 'fusion_dropout_rate', 0.5)

        self.emb_dropout = nn.Dropout(self.emb_dropout_rate)
        self.fusion_dropout = nn.Dropout(self.fusion_dropout_rate)

        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)

        self.sparse_emb = nn.ModuleDict()
        self.emb_transform = nn.ModuleDict()
        self.hstu_layers = nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + \
                  len(self.USER_CONTINUAL_FEAT)
        itemdim = args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT)) + \
                  len(self.ITEM_CONTINUAL_FEAT) + args.hidden_units * len(self.ITEM_EMB_FEAT)

        self.userdnn = nn.Linear(userdim, args.hidden_units)
        self.itemdnn = nn.Linear(itemdim, args.hidden_units)
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        # === MOD: 仅 FFN dropout ===
        for _ in range(args.num_blocks):
            self.hstu_layers.append(
                HSTU(args.hidden_units, args.num_heads, self.ffn_dropout_rate)
            )

        # === MOD: SENet 融合 ===
        self.fusion_layer = SENetFusion(input_dim=args.hidden_units)

        # 其余 embedding 初始化
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.USER_SPARSE_FEAT[k], args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.ITEM_SPARSE_FEAT[k], args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.ITEM_ARRAY_FEAT[k], args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.USER_ARRAY_FEAT[k], args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

    # ---------- 其余函数 ----------
    def _init_feat_info(self, feat_statistics, feat_types):
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] + 1 for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_contiual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] + 1 for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] + 1 for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] + 1 for k in feat_types['item_array']}
        self.ITEM_EMB_FEAT = {k: feat_statistics[k] for k in feat_types['item_emb']}

    def feat2emb(self, seq, feature_tensors, mask=None, include_user=False):
        seq = seq.to(self.dev)
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_emb = self.user_emb(user_mask * seq)
            item_emb = self.item_emb(item_mask * seq)
            item_feat_list = [item_emb]
            user_feat_list = [user_emb]
        else:
            item_emb = self.item_emb(seq)
            item_feat_list = [item_emb]

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
                tensor = feature_tensors[k].to(self.dev)
                if feat_type == 'sparse':
                    feat_list.append(self.sparse_emb[k](tensor))
                elif feat_type == 'array':
                    feat_list.append(self.sparse_emb[k](tensor).sum(2))
                elif feat_type == 'continual':
                    feat_list.append(tensor.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            tensor = feature_tensors[k].to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor))

        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = F.relu(self.itemdnn(all_item_emb))

        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = F.relu(self.userdnn(all_user_emb))
            return all_item_emb, all_user_emb
        return all_item_emb, None

    def log2feats(self, log_seqs, mask, seq_feature_tensors):
        # === MOD: 分别拿 item & user 表征 ===
        item_emb, user_emb = self.feat2emb(log_seqs, seq_feature_tensors, mask=mask, include_user=True)

        # === MOD: SENet 融合 + 融合后 dropout ===
        if user_emb is not None:
            seqs = self.fusion_layer(item_emb, user_emb)
        else:
            seqs = item_emb
        seqs = self.fusion_dropout(seqs)

        # 加 scale & embedding dropout
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)

        # === RoPE & mask ===
        attn_mask = torch.tril(torch.ones((self.maxlen + 1, self.maxlen + 1),
                                          dtype=torch.bool, device=self.dev))
        attn_mask = attn_mask.unsqueeze(0).expand(log_seqs.size(0), -1, -1)
        if mask is not None:
            attn_mask &= (mask != 0).unsqueeze(1)

        for layer in self.hstu_layers:
            seqs = layer(seqs, attn_mask=attn_mask)
        return self.last_layernorm(seqs)

    # === 下方函数与原版完全一致 ===
    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type,
                seq_feat, pos_feat, neg_feat):
        log_feats = self.log2feats(user_item, mask, seq_feat)
        loss_mask = (next_mask == 1).to(self.dev)
        pos_embs = self.feat2emb(pos_seqs, pos_feat, include_user=False)[0]
        neg_embs = self.feat2emb(neg_seqs, neg_feat, include_user=False)[0]
        return log_feats, pos_embs, neg_embs, loss_mask

    def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask, writer):
        hidden_size = neg_embs.size(-1)
        seq_embs = F.normalize(seq_embs, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)

        pos_logits = (seq_embs * pos_embs).sum(dim=-1, keepdim=True)
        neg_logits = torch.matmul(seq_embs, neg_embs.view(-1, hidden_size).T)
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[loss_mask.bool()] / self.temp
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)

    def predict(self, log_seqs, seq_feature, mask):
        final_feat = self.log2feats(log_seqs, mask, seq_feature)[:, -1, :]
        return F.normalize(final_feat, p=2, dim=-1)

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        all_embs = []
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)

            batch_feat_list = feat_dict[start_idx:end_idx]
            item_feat_tensors = {}
            all_keys = set(k for d in batch_feat_list for k in d.keys())

            for k in all_keys:
                if any(isinstance(d.get(k), list) for d in batch_feat_list):
                    max_len = max(len(d.get(k, [])) for d in batch_feat_list)
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

            batch_emb = self.feat2emb(item_seq, item_feat_tensors, include_user=False)[0]
            batch_emb = F.normalize(batch_emb, p=2, dim=-1)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))