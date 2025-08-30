def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask):
<<<<<<< HEAD
=======
        """
        seq_embs: [B, D] sequence embeddings
        pos_embs: [B, D] positive embeddings
        neg_embs: [B, D] negative embeddings
        loss_mask: [B] mask to filter valid samples
        """
>>>>>>> ff6b965 (update gemini)
        hidden_size = neg_embs.size(-1)

        # L2 normalization
        seq_embs = seq_embs / seq_embs.norm(dim=-1, keepdim=True)
        pos_embs = pos_embs / pos_embs.norm(dim=-1, keepdim=True)
        neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)

        # Positive logits (cosine similarity)
        pos_logits = F.cosine_similarity(seq_embs, pos_embs, dim=-1).unsqueeze(-1)

        if self.writer is not None:
            self.writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item())

        # Negative logits: reshape negatives [B, N, D] -> [B, N*D]
        neg_embedding_all = neg_embs.reshape(-1, hidden_size)  # [B*N, D]
        neg_logits = torch.matmul(seq_embs, neg_embedding_all.transpose(0, 1))  # [B, B*N]

        if self.writer is not None:
            self.writer.add_scalar("Model/nce_neg_logits", neg_logits.mean().item())

        # Concatenate positive and negative logits
        logits = torch.cat([pos_logits, neg_logits], dim=-1)  # [B, 1+N*B]

        # Apply mask and temperature scaling
        logits = logits[loss_mask.bool()] / self.temp

        # Labels: 0 means the positive sample is always at index 0
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss