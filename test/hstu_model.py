import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from dataset import save_emb
from dot_product_similarity import DotProductSimilarity
from hstu_components import (
    LocalEmbeddingModule,
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    L2NormEmbeddingPostprocessor,
)
from simplified_hstu import HSTU


class HSTUModel(torch.nn.Module):
    """
    HSTU (Hierarchical Sequential Transduction Unit) model adapted for the existing codebase.
    
    This model replaces the BaselineModel with the HSTU architecture from the generative-recommenders repository.
    It maintains compatibility with the existing dataset and training pipeline.
    
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表
        args: 全局参数
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super(HSTUModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen
        self.temperature = getattr(args, 'infonce_temp', 0.1)
        
        # HSTU specific parameters
        self.num_blocks = getattr(args, 'num_blocks', 2)
        self.num_heads = getattr(args, 'num_heads', 1)
        self.attention_dim = getattr(args, 'attention_dim', 64)
        self.linear_dim = getattr(args, 'linear_dim', 64)
        self.dropout_rate = getattr(args, 'dropout_rate', 0.2)
        
        # Initialize feature information
        self._init_feat_info(feat_statistics, feat_types)
        
        # Create embedding module for HSTU
        self.embedding_module = LocalEmbeddingModule(
            num_items=item_num,
            item_embedding_dim=args.hidden_units,
        )
        
        # Create input features preprocessor
        self.input_preprocessor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=args.maxlen,
            embedding_dim=args.hidden_units,
            dropout_rate=self.dropout_rate,
        )
        
        # Create output postprocessor
        self.output_postprocessor = L2NormEmbeddingPostprocessor(
            embedding_dim=args.hidden_units,
        )
        
        # Create similarity module
        self.similarity_module = DotProductSimilarity()
        
        # Create HSTU model
        self.hstu = HSTU(
            max_sequence_len=args.maxlen,
            max_output_len=1,  # For next item prediction
            embedding_dim=args.hidden_units,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            attention_dim=self.attention_dim,
            linear_dim=self.linear_dim,
            linear_dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.dropout_rate,
            embedding_module=self.embedding_module,
            similarity_module=self.similarity_module,
            input_features_preproc_module=self.input_preprocessor,
            output_postproc_module=self.output_postprocessor,
            enable_relative_attention_bias=True,
            verbose=False,
        )
        
        # Feature processing modules (to handle additional features)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        
        # Initialize feature embeddings
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(
                self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0
            )
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(
                self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0
            )
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(
                self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0
            )
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(
                self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0
            )
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(
                self.ITEM_EMB_FEAT[k], args.hidden_units
            )

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息按特征类型分组产生不同的字典
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}

    def feat2emb(self, seq, feature_tensors, mask=None, include_user=False):
        """
        Convert features to embeddings, compatible with existing dataset format
        
        Args:
            seq: 序列ID
            feature_tensors: 预处理好的特征tensor字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        
        # Get base item embeddings
        item_embeddings = self.embedding_module.get_item_embeddings(seq)
        
        # Process additional features
        item_feat_list = [item_embeddings]
        user_feat_list = []
        
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            user_embedding = self.embedding_module.get_item_embeddings(user_mask * seq)
            user_feat_list = [user_embedding]
        
        # Batch-process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend([
                (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
            ])

        # Process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                if k in feature_tensors:
                    tensor_feature = feature_tensors[k].to(self.dev)
                    
                    if feat_type.endswith('sparse'):
                        feat_list.append(self.sparse_emb[k](tensor_feature))
                    elif feat_type.endswith('array'):
                        feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                    elif feat_type.endswith('continual'):
                        feat_list.append(tensor_feature.unsqueeze(2))

        # Process embedding features
        for k in self.ITEM_EMB_FEAT:
            if k in feature_tensors:
                tensor_feature = feature_tensors[k].to(self.dev)
                item_feat_list.append(self.emb_transform[k](tensor_feature))

        # Combine features
        all_item_emb = torch.cat(item_feat_list, dim=2)
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
            
        return seqs_emb

    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, 
                seq_feature_tensors, pos_feature_tensors, neg_feature_tensors):
        """
        训练时调用，计算序列和正负样本的embeddings
        
        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码
            next_mask: 下一个token类型掩码
            next_action_type: 下一个token动作类型
            seq_feature_tensors: 序列特征tensor字典
            pos_feature_tensors: 正样本特征tensor字典
            neg_feature_tensors: 负样本特征tensor字典

        Returns:
            seq_embs: 序列embeddings
            pos_embs: 正样本embeddings
            neg_embs: 负样本embeddings
            loss_mask: 损失掩码
        """
        batch_size = user_item.shape[0]
        seq_len = user_item.shape[1]
        
        # Process sequence features
        seq_embeddings = self.feat2emb(user_item, seq_feature_tensors, mask=mask, include_user=True)
        
        # Create past_lengths tensor (actual sequence lengths)
        past_lengths = (mask != 0).sum(dim=1).long()
        
        # Create past_payloads dictionary
        past_payloads = {}
        if 'timestamps' in seq_feature_tensors:
            past_payloads['timestamps'] = seq_feature_tensors['timestamps']
        if 'ratings' in seq_feature_tensors:
            past_payloads['ratings'] = seq_feature_tensors['ratings']
        
        # Forward through HSTU
        seq_embs = self.hstu(
            past_lengths=past_lengths,
            past_ids=user_item,
            past_embeddings=seq_embeddings,
            past_payloads=past_payloads,
        )
        
        # Process positive and negative samples
        pos_embs = self.feat2emb(pos_seqs, pos_feature_tensors, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature_tensors, include_user=False)
        
        # Create loss mask
        loss_mask = (next_mask == 1).to(self.dev)
        
        return seq_embs, pos_embs, neg_embs, loss_mask

    def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask, writer=None):
        """
        计算InfoNCE损失函数
        """
        # 只保留有效的位置（根据loss_mask）
        valid_indices = loss_mask.bool()
        
        # 提取有效的embeddings
        seq_embs_valid = seq_embs[valid_indices]
        pos_embs_valid = pos_embs[valid_indices]
        neg_embs_valid = neg_embs[valid_indices]
        
        # 归一化embeddings
        seq_embs_valid = F.normalize(seq_embs_valid, p=2, dim=-1)
        pos_embs_valid = F.normalize(pos_embs_valid, p=2, dim=-1)
        neg_embs_valid = F.normalize(neg_embs_valid, p=2, dim=-1)
        
        # 计算正样本的相似度
        pos_sim = torch.sum(seq_embs_valid * pos_embs_valid, dim=-1)
        
        # 计算负样本的相似度
        neg_sim = torch.matmul(seq_embs_valid, neg_embs_valid.transpose(-1, -2))
        
        # 为了避免负样本中包含正样本，将对角线设为很小的值
        neg_sim = neg_sim - torch.eye(neg_sim.size(0), device=neg_sim.device) * 1e9
        
        # 拼接正负样本的相似度
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # 应用温度参数
        logits = logits / self.temperature
        
        # 创建标签（正样本的索引是0）
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        # 记录到TensorBoard
        if writer is not None:
            writer.add_scalar("Model/nce_pos_logits", pos_sim.mean().item())
            writer.add_scalar("Model/nce_neg_logits", neg_sim.mean().item())
            writer.add_scalar("Model/temperature", self.temperature)
        
        return loss

    def predict(self, log_seqs, seq_feature_tensors, mask):
        """
        计算用户序列的表征
        """
        batch_size = log_seqs.shape[0]
        
        # Process sequence features
        seq_embeddings = self.feat2emb(log_seqs, seq_feature_tensors, mask=mask, include_user=True)
        
        # Create past_lengths tensor
        past_lengths = (mask != 0).sum(dim=1).long()
        
        # Create past_payloads dictionary
        past_payloads = {}
        if 'timestamps' in seq_feature_tensors:
            past_payloads['timestamps'] = seq_feature_tensors['timestamps']
        if 'ratings' in seq_feature_tensors:
            past_payloads['ratings'] = seq_feature_tensors['ratings']
        
        # Forward through HSTU
        log_feats = self.hstu(
            past_lengths=past_lengths,
            past_ids=log_seqs,
            past_embeddings=seq_embeddings,
            past_payloads=past_payloads,
        )
        
        # Get the last token embedding
        final_feat = log_feats[:, -1, :]
        
        # 归一化embeddings，与训练时保持一致
        final_feat = F.normalize(final_feat, p=2, dim=-1)
        
        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)

            # Convert features to tensors
            feature_tensors = {}
            for feat_type, feat_dict_type in [
                ('item_sparse', self.ITEM_SPARSE_FEAT),
                ('item_array', self.ITEM_ARRAY_FEAT),
            ]:
                for feat_id in feat_dict_type:
                    from dataset import MyDataset
                    dummy_dataset = MyDataset.__new__(MyDataset)
                    dummy_dataset.ITEM_SPARSE_FEAT = self.ITEM_SPARSE_FEAT
                    dummy_dataset.ITEM_ARRAY_FEAT = self.ITEM_ARRAY_FEAT
                    dummy_dataset.ITEM_CONTINUAL_FEAT = self.ITEM_CONTINUAL_FEAT
                    dummy_dataset.ITEM_EMB_FEAT = self.ITEM_EMB_FEAT
                    dummy_dataset.feature_types = {
                        'item_sparse': list(self.ITEM_SPARSE_FEAT.keys()),
                        'item_array': list(self.ITEM_ARRAY_FEAT.keys()),
                        'item_emb': list(self.ITEM_EMB_FEAT.keys()),
                        'item_continual': []
                    }
                    feature_tensors[feat_id] = dummy_dataset.feat2tensor([batch_feat], feat_id, self.dev)

            # Handle embedding features
            for k in self.ITEM_EMB_FEAT:
                batch_size_feat = len(batch_feat)
                seq_len = 1
                emb_dim = self.ITEM_EMB_FEAT[k]
                
                batch_emb_data = np.zeros((batch_size_feat, seq_len, emb_dim), dtype=np.float32)
                for i, item in enumerate(batch_feat):
                    if k in item:
                        batch_emb_data[i, 0] = item[k]
                
                feature_tensors[k] = torch.from_numpy(batch_emb_data).to(self.dev)

            batch_emb = self.feat2emb(item_seq, feature_tensors, include_user=False).squeeze(0)
            
            # 归一化embeddings，与训练时保持一致
            batch_emb = F.normalize(batch_emb, p=2, dim=-1)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))