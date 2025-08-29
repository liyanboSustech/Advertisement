import argparse
import json
import os
import time
from pathlib import Path
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from dataset import MyDataset
from model import BaselineModel


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    # 模型参数
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=0.005, type=float)  # 新增 weight decay
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    # 多模态特征ID
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature for InfoNCE loss')
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, 
                       choices=[str(s) for s in range(81, 87)])
    return parser.parse_args()


# def compute_infonce_loss(seq_embs, pos_embs, neg_embs, temperature, writer=None):
#     """
#     计算InfoNCE损失 - 修复版本
    
#     Args:
#         seq_embs: 序列embeddings, shape: [valid_batch_size, hidden_size]
#         pos_embs: 正样本embeddings, shape: [valid_batch_size, hidden_size] 
#         neg_embs: 负样本embeddings, shape: [valid_batch_size, hidden_size]
#         temperature: 温度参数
#         writer: tensorboard writer
#     """
#     # 标准化embedding向量
#     seq_embs = F.normalize(seq_embs, dim=-1)
#     pos_embs = F.normalize(pos_embs, dim=-1)
#     neg_embs = F.normalize(neg_embs, dim=-1)
    
#     # 计算正样本相似度 [valid_batch_size, 1]
#     pos_logits = torch.sum(seq_embs * pos_embs, dim=-1, keepdim=True)
    
#     # 计算负样本相似度 [valid_batch_size, 1]
#     neg_logits = torch.sum(seq_embs * neg_embs, dim=-1, keepdim=True)
    
#     if writer is not None:
#         writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item())
#         writer.add_scalar("Model/nce_neg_logits", neg_logits.mean().item())
    
#     # 合并logits并应用temperature [valid_batch_size, 2]
#     logits = torch.cat([pos_logits, neg_logits], dim=-1) / temperature
    
#     # 标签全为0，表示正样本在第0个位置
#     labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
    
#     # 计算交叉熵损失
#     loss = F.cross_entropy(logits, labels)
    
#     return loss

def compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, temperature):
        """你提供的InfoNCE损失函数"""
        hidden_size = neg_embs.size(-1)
        seq_embs = seq_embs / seq_embs.norm(dim=-1, keepdim=True)
        pos_embs = pos_embs / pos_embs.norm(dim=-1, keepdim=True)
        pos_logits = F.cosine_similarity(seq_embs, pos_embs, dim=-1).unsqueeze(-1)
        writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item())
        neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)

        neg_embedding_all = neg_embs.reshape(-1, hidden_size)
        neg_logits = torch.matmul(seq_embs, neg_embedding_all.transpose(-1, -2))
        writer.add_scalar("Model/nce_neg_logits", neg_logits.mean().item())
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[loss_mask.bool()] / temperature
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels)
        return loss



def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """余弦退火学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == '__main__':
    # 初始化目录
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    data_path = os.environ.get('TRAIN_DATA_PATH')

    # 加载参数和数据集
    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    # 数据加载器（可用多线程）
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=6, collate_fn=dataset.collate_fn,
        worker_init_fn=dataset._worker_init_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,  # 验证集不需要shuffle
        num_workers=6, collate_fn=dataset.collate_fn,
        worker_init_fn=dataset._worker_init_fn
    )

    # 初始化模型
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # 初始化参数
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    # 加载预训练模型
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except Exception as e:
            print(f"Failed loading state_dict: {e}")
            raise

    # 使用AdamW优化器和余弦退火调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr * 0.5,  # 减小学习率
        weight_decay=args.weight_decay, 
        betas=(0.9, 0.999),
        eps=1e-6  # 增加数值稳定性
    )
    # 计算总训练步数
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = total_steps // 10  # 10% warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 训练循环
    best_val_loss = float('inf')
    global_step = 0
    print("Start training")
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            
            # 转移到设备
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            token_type = token_type.to(args.device)
            next_token_type = next_token_type.to(args.device)
            
            # 前向传播
            pos_logits, neg_logits, seq_embs, pos_embs, neg_embs, loss_mask = model(
                seq, pos, neg, token_type, next_token_type, next_action_type,
                seq_feat, pos_feat, neg_feat
            )
            
            # 检查是否有有效位置
            if loss_mask.sum() > 0:
                # 调用你的compute_infonce_loss函数
                loss = compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, args.temperature)

                # L2正则化
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                    
            else:
                print(f"Warning: No valid positions in batch {step}, skipping...")
                continue
            
            # 日志和优化
            log_json = json.dumps({
                'global_step': global_step, 'loss': loss.item(), 
                'epoch': epoch, 'time': time.time(), 'lr': scheduler.get_last_lr()[0]
            })
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

        # 验证循环 - 修正逻辑错误
        model.eval()
        valid_losses = 0.0
        valid_batches_count = 0  # 用于计算有效批次数量

        with torch.no_grad():
            for batch in tqdm(valid_loader, total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                token_type = token_type.to(args.device)
                next_token_type = next_token_type.to(args.device)

                # 前向传播
                pos_logits, neg_logits, seq_embs, pos_embs, neg_embs, loss_mask = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type,
                    seq_feat, pos_feat, neg_feat
                )
                
                # 检查有效位置
                if loss_mask.sum() > 0:
                    # 使用你的损失函数
                    loss = compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, args.temperature)
                    valid_losses += loss.item()
                    valid_batches_count += 1
                else:
                    print(f"Warning: No valid positions in validation batch, skipping...")

        # 计算平均验证损失
        if valid_batches_count > 0:
            avg_valid_loss = valid_losses / valid_batches_count
            writer.add_scalar('Loss/valid', avg_valid_loss, global_step)
            print(f"Epoch {epoch} valid loss: {avg_valid_loss:.4f}")
        else:
            avg_valid_loss = float('inf')
            print(f"Epoch {epoch}: No valid validation batches")

        # 保存模型
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={avg_valid_loss:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Training done")
    writer.close()
    log_file.close()