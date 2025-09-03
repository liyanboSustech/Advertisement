# model.py
# 优化版本 - 主要改进：
# 1. 添加混合精度训练支持
# 2. 修改LayerNorm顺序（Pre-LN架构）
# 3. 支持in-batch negatives
# 4. 添加验证指标计算

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataset import save_emb


class RoPE(nn.Module):
    """旋转位置编码(Rotary Position Embedding)"""
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算cos和sin值
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def rotate_half(self, x):
        """将向量的后半部分取负号并与前半部分交换"""
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


class HSTU(nn.Module):
    """混合门控注意力单元 - Pre-LN架构"""
    def __init__(self, hidden_units, num_heads, ffn_dropout_rate):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.ffn_dropout_rate = ffn_dropout_rate
        
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        # Pre-LN: LayerNorm在attention和FFN之前
        self.attn_layer_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn_layer_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        
        # 4倍投影：U(门控), V(值), Q(查询), K(键)
        self.pointwise_proj = nn.Linear(hidden_units, 4 * hidden_units)
        self.rope = RoPE(self.head_dim)
        self.pointwise_transform = nn.Linear(hidden_units, hidden_units)
        
        # 仅FFN使用dropout，attention部分不使用
        self.ffn_dropout = nn.Dropout(p=ffn_dropout_rate)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.size()
        
        # === Attention Block with Pre-LN ===
        residual = x
        x_norm = self.attn_layer_norm(x)  # Pre-LN
        
        # 投影到4个子空间
        proj = self.pointwise_proj(x_norm)
        U, V, Q, K = torch.chunk(proj, 4, dim=-1)

        # 重塑为多头格式
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用RoPE位置编码
        Q_rope, K_rope = self.rope(Q, K, L)
        
        # 计算attention scores
        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(Q_rope, K_rope.transpose(-2, -1)) * scale
        
        # 应用因果mask
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

        # 使用sigmoid激活（区别于传统softmax）
        attn_weights = torch.sigmoid(scores)
        
        # 加权求和
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.hidden_units)

        # 门控机制
        U = U.view(B, L, self.hidden_units)
        gated = F.silu(U) * attn_out  # SiLU激活函数
        
        # 第一个残差连接
        x = residual + gated
        
        # === FFN Block with Pre-LN ===
        residual = x
        x_norm = self.ffn_layer_norm(x)  # Pre-LN
        
        # FFN变换（这里使用dropout）
        out = self.pointwise_transform(x_norm)
        out = self.ffn_dropout(out)
        
        # 第二个残差连接
        return residual + out


class SENetFusion(nn.Module):
    """SENet风格的特征融合模块"""
    def __init__(self, input_dim, reduction=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, item_emb, user_emb):
        # 通过item+user特征计算注意力权重
        combined = item_emb + user_emb
        attention_weights = self.attention(combined)
        
        # 加权融合：item * weight + user * (1-weight)
        fused = item_emb * attention_weights + user_emb * (1 - attention_weights)
        return fused


class BaselineModel(nn.Module):
    """主模型类 - 支持混合精度训练和验证指标"""
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen
        self.temp = args.temperature

        # 分层dropout配置
        self.emb_dropout_rate = getattr(args, 'emb_dropout_rate', 0.3)
        self.fusion_dropout_rate = getattr(args, 'fusion_dropout_rate', 0.2)
        self.ffn_dropout_rate = getattr(args, 'ffn_dropout_rate', 0.1)

        # 创建不同的dropout层
        self.emb_dropout = nn.Dropout(self.emb_dropout_rate)
        self.fusion_dropout = nn.Dropout(self.fusion_dropout_rate)

        # 基础embedding层
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)

        # 特征相关的模块
        self.sparse_emb = nn.ModuleDict()
        self.emb_transform = nn.ModuleDict()
        
        # 初始化特征信息
        self._init_feat_info(feat_statistics, feat_types)

        # 计算用户和物品特征维度
        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + \
                  len(self.USER_CONTINUAL_FEAT)
        itemdim = args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT)) + \
                  len(self.ITEM_CONTINUAL_FEAT) + args.hidden_units * len(self.ITEM_EMB_FEAT)

        # 特征变换层
        self.userdnn = nn.Linear(userdim, args.hidden_units)
        self.itemdnn = nn.Linear(itemdim, args.hidden_units)
        
        # Transformer层
        self.hstu_layers = nn.ModuleList([
            HSTU(args.hidden_units, args.num_heads, self.ffn_dropout_rate)
            for _ in range(args.num_blocks)
        ])

        # SENet融合层
        self.fusion_layer = SENetFusion(input_dim=args.hidden_units)
        
        # 最终层归一化（Pre-LN架构需要最后的LN）
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        # 初始化所有特征embedding
        self._init_feature_embeddings(args)

    def _init_feat_info(self, feat_statistics, feat_types):
        """初始化特征信息字典"""
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] + 1 for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] + 1 for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] + 1 for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] + 1 for k in feat_types['item_array']}
        self.ITEM_EMB_FEAT = {k: feat_statistics[k] for k in feat_types['item_emb']}

    def _init_feature_embeddings(self, args):
        """初始化所有特征embedding层"""
        # 用户稀疏特征
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = nn.Embedding(
                self.USER_SPARSE_FEAT[k], args.hidden_units, padding_idx=0
            )
        
        # 物品稀疏特征
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = nn.Embedding(
                self.ITEM_SPARSE_FEAT[k], args.hidden_units, padding_idx=0
            )
        
        # 数组特征
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = nn.Embedding(
                self.ITEM_ARRAY_FEAT[k], args.hidden_units, padding_idx=0
            )
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = nn.Embedding(
                self.USER_ARRAY_FEAT[k], args.hidden_units, padding_idx=0
            )
        
        # 预训练embedding特征
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

    def feat2emb(self, seq, feature_tensors, mask=None, include_user=False):
        """将特征转换为embedding表示"""
        seq = seq.to(self.dev)
        
        if include_user:
            # 分离用户和物品
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_emb = self.user_emb(user_mask * seq)
            item_emb = self.item_emb(item_mask * seq)
            item_feat_list = [item_emb]
            user_feat_list = [user_emb]
        else:
            item_emb = self.item_emb(seq)
            item_feat_list = [item_emb]

        # 处理各类特征
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

        # 处理每组特征
        for feat_dict, feat_list, feat_type in feature_groups:
            for k in feat_dict:
                tensor = feature_tensors[k].to(self.dev)
                if feat_type == 'sparse':
                    feat_list.append(self.sparse_emb[k](tensor))
                elif feat_type == 'array':
                    feat_list.append(self.sparse_emb[k](tensor).sum(2))
                elif feat_type == 'continual':
                    feat_list.append(tensor.unsqueeze(2))

        # 处理预训练embedding特征
        for k in self.ITEM_EMB_FEAT:
            tensor = feature_tensors[k].to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor))

        # 拼接并变换
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = F.relu(self.itemdnn(all_item_emb))

        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = F.relu(self.userdnn(all_user_emb))
            return all_item_emb, all_user_emb
        
        return all_item_emb, None

    def log2feats(self, log_seqs, mask, seq_feature_tensors):
        """序列特征处理的核心流程"""
        # 1. 获取item和user的基础embedding
        item_emb, user_emb = self.feat2emb(
            log_seqs, seq_feature_tensors, mask=mask, include_user=True
        )

        # 2. SENet融合（如果有用户信息）
        if user_emb is not None:
            seqs = self.fusion_layer(item_emb, user_emb)
        else:
            seqs = item_emb
        
        # 3. 融合后dropout（在进Transformer前）
        seqs = self.fusion_dropout(seqs)

        # 4. 添加位置缩放 + embedding dropout
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)

        attention_mask = torch.tril(
            torch.ones((self.maxlen + 1, self.maxlen + 1), dtype=torch.bool, device=self.dev))
        attention_mask = attention_mask.unsqueeze(0).expand(log_seqs.shape[0], -1, -1)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask & attention_mask_pad.unsqueeze(1)

        for hstu_layer in self.hstu_layers:
            seqs = hstu_layer(seqs, attn_mask=attention_mask)

        # Pre-LN架构需要最后的LayerNorm
        return self.last_layernorm(seqs)

    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type,
                seq_feat, pos_feat, neg_feat):
        """前向传播"""
        log_feats = self.log2feats(user_item, mask, seq_feat)
        loss_mask = (next_mask == 1).to(self.dev)
        pos_embs, _ = self.feat2emb(pos_seqs, pos_feat, include_user=False)
        neg_embs, _ = self.feat2emb(neg_seqs, neg_feat, include_user=False)
        return log_feats, pos_embs, neg_embs, loss_mask

    def compute_infonce_loss_with_inbatch(self, seq_embs, pos_embs, neg_embs, loss_mask, writer):
        """计算InfoNCE损失 - 支持in-batch negatives"""
        batch_size, seq_len, hidden_size = seq_embs.size()
        
        # L2归一化
        seq_embs_normalized = F.normalize(seq_embs, p=2, dim=-1)
        pos_embs_normalized = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs_normalized = F.normalize(neg_embs, p=2, dim=-1)
        
        # 获取有效位置的embeddings
        valid_seq_embs = seq_embs_normalized[loss_mask.bool()]  # [N, hidden_size]
        valid_pos_embs = pos_embs_normalized[loss_mask.bool()]  # [N, hidden_size]
        
        if valid_seq_embs.size(0) == 0:
            return torch.tensor(0.0, device=seq_embs.device, requires_grad=True)
        
        # 正样本相似度
        pos_logits = torch.sum(valid_seq_embs * valid_pos_embs, dim=-1, keepdim=True)  # [N, 1]
        writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item())
        
        # 构建in-batch negatives + 原始negatives
        # 1. In-batch negatives: 同一batch内其他样本的正样本作为当前样本的负样本
        inbatch_neg_logits = torch.matmul(valid_seq_embs, valid_pos_embs.T)  # [N, N]
        # 移除对角线（自己不能作为自己的负样本）
        mask_diag = ~torch.eye(inbatch_neg_logits.size(0), dtype=torch.bool, device=inbatch_neg_logits.device)
        inbatch_neg_logits = inbatch_neg_logits[mask_diag].view(inbatch_neg_logits.size(0), -1)
        
        # 2. 原始random negatives
        valid_neg_embs = neg_embs_normalized[loss_mask.bool()]  # [N, hidden_size] 
        random_neg_logits = torch.matmul(valid_seq_embs, valid_neg_embs.T)  # [N, N]
        
        writer.add_scalar("Model/nce_inbatch_neg_logits", inbatch_neg_logits.mean().item())
        writer.add_scalar("Model/nce_random_neg_logits", random_neg_logits.mean().item())
        
        # 拼接所有logits: [pos, in-batch_negs, random_negs]
        all_logits = torch.cat([pos_logits, inbatch_neg_logits, random_neg_logits], dim=-1)  # [N, 1+batch_size-1+batch_size]
        
        # 应用温度
        all_logits = all_logits / self.temp
        
        # 标签都是0（正样本在第一个位置）
        labels = torch.zeros(all_logits.size(0), device=all_logits.device, dtype=torch.long)
        
        loss = F.cross_entropy(all_logits, labels)
        return loss

    def predict(self, log_seqs, seq_feature, mask):
        """预测函数"""
        final_feat = self.log2feats(log_seqs, mask, seq_feature)[:, -1, :]
        return F.normalize(final_feat, p=2, dim=-1)

    def evaluate_metrics(self, valid_loader, k=10):
        """计算验证集上的HitRate@k和NDCG@k"""
        self.eval()
        hit_rates = []
        ndcgs = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Evaluating"):
                seq, pos, neg, token_type, next_token_type, next_action_type, \
                    seq_feat, pos_feat, neg_feat = batch
                
                # 获取序列表示
                seq_embs = self.log2feats(seq, token_type, seq_feat)[:, -1, :]  # [B, H]
                seq_embs = F.normalize(seq_embs, p=2, dim=-1)
                
                # 获取候选物品表示
                pos_embs, _ = self.feat2emb(pos, pos_feat, include_user=False)
                neg_embs, _ = self.feat2emb(neg, neg_feat, include_user=False)
                
                # 构建候选集：正样本 + 负样本
                batch_size = seq_embs.size(0)
                for i in range(batch_size):
                    # 获取有效的候选物品（next_token_type[i, :] == 1）
                    valid_mask = (next_token_type[i, :] == 1) & (pos[i, :] != 0)
                    if not valid_mask.any():
                        continue
                        
                    # 取第一个有效位置作为目标
                    target_idx = valid_mask.nonzero(as_tuple=True)[0][0].item()
                    
                    # 正样本
                    pos_emb = F.normalize(pos_embs[i, target_idx, :].unsqueeze(0), p=2, dim=-1)
                    
                    # 负样本（随机选择k-1个）
                    neg_candidates = F.normalize(neg_embs[i, :valid_mask.sum(), :], p=2, dim=-1)
                    if neg_candidates.size(0) >= k-1:
                        neg_selected = neg_candidates[:k-1]
                    else:
                        # 如果负样本不够，重复采样
                        repeat_times = (k-1) // neg_candidates.size(0) + 1
                        neg_repeated = neg_candidates.repeat(repeat_times, 1)
                        neg_selected = neg_repeated[:k-1]
                    
                    # 构建候选集：[pos, neg1, neg2, ..., neg_{k-1}]
                    candidates = torch.cat([pos_emb, neg_selected], dim=0)  # [k, H]
                    
                    # 计算相似度得分
                    query = seq_embs[i, :].unsqueeze(0)  # [1, H]
                    scores = torch.matmul(query, candidates.T).squeeze(0)  # [k]
                    
                    # 计算排名
                    _, indices = torch.sort(scores, descending=True)
                    rank = (indices == 0).nonzero(as_tuple=True)[0].item() + 1  # 正样本的排名
                    
                    # HitRate@k
                    hit_rates.append(1.0 if rank <= k else 0.0)
                    
                    # NDCG@k
                    if rank <= k:
                        ndcgs.append(1.0 / math.log2(rank + 1))
                    else:
                        ndcgs.append(0.0)
        
        self.train()
        return np.mean(hit_rates), np.mean(ndcgs)

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """保存物品embedding"""
        all_embs = []
        
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)

            # 准备特征数据
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

            # 计算embedding
            batch_emb, _ = self.feat2emb(item_seq, item_feat_tensors, include_user=False)
            batch_emb = batch_emb.squeeze(0)
            batch_emb = F.normalize(batch_emb, p=2, dim=-1)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 保存结果
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))