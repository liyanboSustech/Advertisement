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
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    # 模型参数
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)  # 新增 weight decay
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    # 多模态特征ID
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature for InfoNCE loss')
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, 
                       choices=[str(s) for s in range(81, 87)])
    return parser.parse_args()

# 

def compute_infonce_loss(seq_embs, pos_embs, neg_embs, temperature, writer=None):
    """
    计算InfoNCE损失 - 修复版本
    
    Args:
        seq_embs: 序列embeddings, shape: [valid_batch_size, hidden_size]
        pos_embs: 正样本embeddings, shape: [valid_batch_size, hidden_size] 
        neg_embs: 负样本embeddings, shape: [valid_batch_size, hidden_size]
        temperature: 温度参数
        writer: tensorboard writer
    """
    # 标准化embedding向量
    seq_embs = F.normalize(seq_embs, dim=-1)
    pos_embs = F.normalize(pos_embs, dim=-1)
    neg_embs = F.normalize(neg_embs, dim=-1)
    
    # 计算正样本相似度 [valid_batch_size, 1]
    pos_logits = torch.sum(seq_embs * pos_embs, dim=-1, keepdim=True)
    
    # 计算负样本相似度 [valid_batch_size, 1]
    neg_logits = torch.sum(seq_embs * neg_embs, dim=-1, keepdim=True)
    
    if writer is not None:
        writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item())
        writer.add_scalar("Model/nce_neg_logits", neg_logits.mean().item())
    
    # 合并logits并应用temperature [valid_batch_size, 2]
    logits = torch.cat([pos_logits, neg_logits], dim=-1) / temperature
    
    # 标签全为0，表示正样本在第0个位置
    labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
    
    # 计算交叉熵损失
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
        valid_dataset, batch_size=args.batch_size, shuffle=True, 
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
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
        
        # 训练批次
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
            
            # 提取有效位置的embeddings用于损失计算
            valid_positions = loss_mask.bool()
            
            if valid_positions.sum() > 0:
                # 从forward返回的embeddings中提取有效位置
                valid_seq_embs = seq_embs[valid_positions]
                valid_pos_embs = pos_embs[valid_positions] 
                valid_neg_embs = neg_embs[valid_positions]
                
                # 计算InfoNCE损失
                loss = compute_infonce_loss(
                    valid_seq_embs, valid_pos_embs, valid_neg_embs, 
                    args.temperature, writer
                )
                
                # L2正则化 - 只在有有效损失时添加
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
            scheduler.step()  # 更新学习率
            global_step += 1

        # 验证
        model.eval()
        valid_loss_sum = 0.0
        valid_batches = 0

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
                
                # 提取有效位置的embeddings用于损失计算
                valid_positions = loss_mask.bool()
                
                if valid_positions.sum() > 0:
                    valid_seq_embs = seq_embs[valid_positions]
                    valid_pos_embs = pos_embs[valid_positions]
                    valid_neg_embs = neg_embs[valid_positions]
                    
                    # 计算损失（验证时不需要writer）
                    loss = compute_infonce_loss(
                        valid_seq_embs, valid_pos_embs, valid_neg_embs,
                        args.temperature, writer=None
                    )
                    
                    valid_loss_sum += loss.item()
                    valid_batches += 1
                else:
                    print(f"Warning: No valid positions in validation batch, skipping...")

        # 计算平均验证损失
        if valid_batches > 0:
            valid_loss = valid_loss_sum / valid_batches
            writer.add_scalar('Loss/valid', valid_loss, global_step)
            print(f"Epoch {epoch} valid loss: {valid_loss:.4f}")

            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"epoch_{epoch}.valid_loss_{valid_loss:.4f}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")
        else:
            print(f"Epoch {epoch}: No valid batches processed")

    print("Training done")
    writer.close()
    log_file.close()