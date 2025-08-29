"""
点积相似度模块
为HSTU模型提供相似度计算功能
"""

import torch
from typing import Optional


class DotProductSimilarity(torch.nn.Module):
    """点积相似度模块"""
    
    def forward(self, query_embeddings: torch.Tensor, item_embeddings: torch.Tensor,
                item_ids: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        计算查询嵌入和物品嵌入之间的点积相似度
        
        Args:
            query_embeddings: (B, query_embedding_dim) 查询嵌入
            item_embeddings: (B, num_items, item_embedding_dim) 或 (1, num_items, item_embedding_dim) 物品嵌入
            item_ids: 可选的物品ID
            **kwargs: 其他参数
            
        Returns:
            similarity_scores: (B, num_items) 相似度分数
        """
        if len(item_embeddings.size()) == 3:
            # item_embeddings: (B, num_items, embedding_dim) 或 (1, num_items, embedding_dim)
            similarity = torch.matmul(
                query_embeddings.unsqueeze(1),  # (B, 1, embedding_dim)
                item_embeddings.transpose(-1, -2)  # (B, embedding_dim, num_items) 或 (1, embedding_dim, num_items)
            ).squeeze(1)  # (B, num_items)
        else:
            # item_embeddings: (B, embedding_dim)
            similarity = torch.sum(query_embeddings * item_embeddings, dim=-1, keepdim=True)
        
        return similarity