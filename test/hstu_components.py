"""
HSTU模型核心组件
包含HSTU模型所需的基础组件类
"""

import abc
import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class EmbeddingModule(torch.nn.Module):
    """嵌入模块基类"""
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    """本地嵌入模块"""
    def __init__(self, num_items: int, item_embedding_dim: int):
        super().__init__()
        self._item_embedding_dim = item_embedding_dim
        self._item_emb = torch.nn.Embedding(num_items + 1, item_embedding_dim, padding_idx=0)
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        torch.nn.init.normal_(self._item_emb.weight.data, mean=0.0, std=0.02)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class InputFeaturesPreprocessorModule(torch.nn.Module):
    """输入特征预处理器基类"""
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(self, past_lengths: torch.Tensor, past_ids: torch.Tensor, 
                past_embeddings: torch.Tensor, past_payloads: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class LearnablePositionalEmbeddingInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):
    """可学习位置嵌入输入特征预处理器"""
    def __init__(self, max_sequence_len: int, embedding_dim: int, dropout_rate: float):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._pos_emb = torch.nn.Embedding(max_sequence_len, self._embedding_dim)
        self._dropout_rate = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.reset_state()

    def debug_str(self) -> str:
        return f"posi_d{self._dropout_rate}"

    def reset_state(self) -> None:
        torch.nn.init.normal_(self._pos_emb.weight.data, mean=0.0, std=math.sqrt(1.0 / self._embedding_dim))

    def forward(self, past_lengths: torch.Tensor, past_ids: torch.Tensor, 
                past_embeddings: torch.Tensor, past_payloads: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class OutputPostprocessorModule(torch.nn.Module):
    """输出后处理器基类"""
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(self, output_embeddings: torch.Tensor) -> torch.Tensor:
        pass


class L2NormEmbeddingPostprocessor(OutputPostprocessorModule):
    """L2归一化嵌入后处理器"""
    def __init__(self, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._eps = eps

    def debug_str(self) -> str:
        return "l2"

    def forward(self, output_embeddings: torch.Tensor) -> torch.Tensor:
        output_embeddings = output_embeddings[..., : self._embedding_dim]
        return output_embeddings / torch.clamp(
            torch.linalg.norm(output_embeddings, ord=None, dim=-1, keepdim=True),
            min=self._eps,
        )