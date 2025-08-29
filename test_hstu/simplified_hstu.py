"""
HSTU模型实现
包含简化的HSTU模型架构
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from hstu_components import (
    EmbeddingModule,
    InputFeaturesPreprocessorModule,
    OutputPostprocessorModule,
)


class SequentialEncoderWithLearnedSimilarityModule(torch.nn.Module):
    """带学习相似度的序列编码器基类"""
    def __init__(self, similarity_module):
        super().__init__()
        self._similarity_module = similarity_module

    def debug_str(self) -> str:
        pass

    def similarity_fn(self, query_embeddings: torch.Tensor, item_ids: torch.Tensor, 
                     item_embeddings: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if item_embeddings is None:
            item_embeddings = self.get_item_embeddings(item_ids)
        return self._similarity_module(
            query_embeddings=query_embeddings,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            **kwargs,
        )


class HSTUAttention(torch.nn.Module):
    """HSTU注意力模块"""
    def __init__(self, embedding_dim: int, linear_dim: int, attention_dim: int, 
                 num_heads: int, dropout_rate: float):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.linear_dim = linear_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # QKV投影
        self.qkv_proj = torch.nn.Linear(
            embedding_dim,
            (linear_dim * 2 + attention_dim * 2) * num_heads
        )
        
        # 输出投影
        self.output_proj = torch.nn.Linear(linear_dim * num_heads, embedding_dim)
        
        # 层归一化
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.size()
        
        # 层归一化
        x_norm = self.layer_norm(x)
        
        # QKV投影
        qkv = self.qkv_proj(x_norm)
        
        # 分割Q, K, V
        qkv = qkv.view(B, N, self.num_heads, 2 * self.linear_dim + 2 * self.attention_dim)
        q, k, v = torch.split(qkv, [self.attention_dim, self.attention_dim, self.linear_dim], dim=-1)
        
        # 重塑为多头注意力格式
        q = q.view(B, N, self.num_heads, self.attention_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.attention_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.linear_dim).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attention_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        
        # 应用SiLU激活和归一化
        scores = F.silu(scores) / N
        
        # Softmax注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力到值
        attn_output = torch.matmul(attn_weights, v)
        
        # 转置和重塑
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, -1)
        
        # 输出投影
        output = self.output_proj(attn_output)
        output = self.dropout(output)
        
        # 残差连接
        return x + output


class HSTU(SequentialEncoderWithLearnedSimilarityModule):
    """
    简化的HSTU实现
    适配现有代码架构
    """
    def __init__(self, max_sequence_len: int, max_output_len: int, embedding_dim: int,
                 num_blocks: int, num_heads: int, attention_dim: int, linear_dim: int,
                 linear_dropout_rate: float = 0.0, attn_dropout_rate: float = 0.0,
                 embedding_module: EmbeddingModule = None, similarity_module = None,
                 input_features_preproc_module: InputFeaturesPreprocessorModule = None,
                 output_postproc_module: OutputPostprocessorModule = None,
                 enable_relative_attention_bias: bool = True, verbose: bool = True):
        super().__init__(similarity_module)

        self._embedding_dim = embedding_dim
        self._item_embedding_dim = embedding_module.item_embedding_dim if embedding_module else embedding_dim
        self._max_sequence_length = max_sequence_len
        self._embedding_module = embedding_module
        self._input_features_preproc = input_features_preproc_module
        self._output_postproc = output_postproc_module
        self._num_blocks = num_blocks
        self._num_heads = num_heads
        self._attention_dim = attention_dim
        self._linear_dim = linear_dim
        self._linear_dropout_rate = linear_dropout_rate
        self._verbose = verbose

        # 创建HSTU块
        self.blocks = torch.nn.ModuleList([
            HSTUAttention(
                embedding_dim=embedding_dim,
                linear_dim=linear_dim,
                attention_dim=attention_dim,
                num_heads=num_heads,
                dropout_rate=linear_dropout_rate,
            )
            for _ in range(num_blocks)
        ])
        
        # 创建因果掩码
        self.register_buffer(
            "_causal_mask",
            torch.triu(
                torch.ones(max_sequence_len, max_sequence_len, dtype=torch.bool),
                diagonal=1,
            ),
        )
        
        self.reset_params()

    def reset_params(self) -> None:
        """初始化参数"""
        for name, params in self.named_parameters():
            if params.requires_grad:
                if 'weight' in name:
                    if len(params.size()) >= 2:
                        torch.nn.init.xavier_uniform_(params.data)
                    else:
                        torch.nn.init.uniform_(params.data, -0.1, 0.1)
                elif 'bias' in name:
                    torch.nn.init.constant_(params.data, 0)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """获取物品嵌入"""
        if self._embedding_module is not None:
            return self._embedding_module.get_item_embeddings(item_ids)
        else:
            raise NotImplementedError("嵌入模块未提供")

    def debug_str(self) -> str:
        return f"HSTU-b{self._num_blocks}-h{self._num_heads}-dqk{self._attention_dim}-dv{self._linear_dim}"

    def generate_user_embeddings(self, past_lengths: torch.Tensor, past_ids: torch.Tensor,
                                past_embeddings: torch.Tensor, past_payloads: Dict[str, torch.Tensor],
                                **kwargs) -> torch.Tensor:
        """生成用户嵌入"""
        # 应用输入预处理
        if self._input_features_preproc is not None:
            past_lengths, user_embeddings, valid_mask = self._input_features_preproc(
                past_lengths=past_lengths,
                past_ids=past_ids,
                past_embeddings=past_embeddings,
                past_payloads=past_payloads,
            )
        else:
            user_embeddings = past_embeddings
            valid_mask = (past_ids != 0).unsqueeze(-1).float()

        # 应用因果掩码
        batch_size, seq_len = past_ids.size()
        causal_mask = self._causal_mask[:seq_len, :seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        # 应用有效掩码
        padding_mask = valid_mask.squeeze(-1).bool()
        attention_mask = causal_mask & padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
        
        # 通过HSTU块处理
        for block in self.blocks:
            user_embeddings = block(user_embeddings, attention_mask)
        
        # 应用输出后处理
        if self._output_postproc is not None:
            user_embeddings = self._output_postproc(user_embeddings)
        
        return user_embeddings

    def forward(self, past_lengths: torch.Tensor, past_ids: torch.Tensor,
                past_embeddings: torch.Tensor, past_payloads: Dict[str, torch.Tensor],
                **kwargs) -> torch.Tensor:
        """前向传播"""
        return self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            **kwargs
        )

    def encode(self, past_lengths: torch.Tensor, past_ids: torch.Tensor,
               past_embeddings: torch.Tensor, past_payloads: Dict[str, torch.Tensor],
               **kwargs) -> torch.Tensor:
        """编码序列获取当前状态嵌入"""
        encoded_embeddings = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            **kwargs
        )
        
        # 获取最后一个非填充嵌入
        current_embeddings = []
        for i, length in enumerate(past_lengths):
            if length > 0:
                current_embeddings.append(encoded_embeddings[i, length - 1])
            else:
                current_embeddings.append(torch.zeros(self._embedding_dim, device=past_embeddings.device))
        
        return torch.stack(current_embeddings, dim=0)