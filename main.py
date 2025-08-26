import argparse
import json
import os
import time
from pathlib import Path

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
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    # 多模态特征ID
    parser.add_argument('--temperature', default=0.07, type=float ,help='Temperature for InfoNCE loss')
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, 
                       choices=[str(s) for s in range(81, 87)])
    return parser.parse_args()

def compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, temperature, writer=None):
    """计算InfoNCE损失"""
    # 标准化embedding向量
    seq_embs = seq_embs / seq_embs.norm(dim=-1, keepdim=True)
    pos_embs = pos_embs / pos_embs.norm(dim=-1, keepdim=True)
    neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)
    
    # 计算正样本相似度
    pos_logits = F.cosine_similarity(seq_embs, pos_embs, dim=-1).unsqueeze(-1)
    if writer is not None:
        writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item())
    
    # 计算负样本相似度
    hidden_size = neg_embs.size(-1)
    neg_embedding_all = neg_embs.reshape(-1, hidden_size)
    neg_logits = torch.matmul(seq_embs, neg_embedding_all.transpose(-1, -2))
    if writer is not None:
        writer.add_scalar("Model/nce_neg_logits", neg_logits.mean().item())
    
    # 合并logits并应用temperature
    logits = torch.cat([pos_logits, neg_logits], dim=-1)
    
    # loss_mask应该已经被应用在seq_embs, pos_embs, neg_embs上了
    # 所以这里直接使用temperature
    logits = logits / temperature
    labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
    loss = F.cross_entropy(logits, labels)
    return loss


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
        num_workers=4, collate_fn=dataset.collate_fn,
        worker_init_fn=dataset._worker_init_fn  # 初始化worker进程
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, collate_fn=dataset.collate_fn,
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

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # 训练循环
    best_val_loss = float('inf')
    global_step = 0
    print("Start training")
    # 这样写总共会生成几个pt文件？
    # 每个epoch都会保存一次模型，所以总共会生成args.num_epochs个pt文件
    # 每个模型文件在何时生成？
    # 在每个epoch结束时生成
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
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type,
                seq_feat, pos_feat, neg_feat
            )
            
            # 提取embeddings用于损失计算
            log_feats = model.log2feats(seq, token_type, seq_feat)
            loss_mask = (next_token_type == 1).to(args.device)
            
            # 只计算有效位置的损失
            valid_positions = loss_mask.bool()
            if valid_positions.sum() > 0:  # 确保有有效位置
                seq_embs = log_feats[valid_positions]
                pos_embs = model.feat2emb(pos, pos_feat, include_user=False)[valid_positions]
                neg_embs = model.feat2emb(neg, neg_feat, include_user=False)[valid_positions]
                
                # 计算损失
                loss = compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, args.temperature, writer)
            else:
                # 如果没有有效位置，使用一个小的虚拟损失
                loss = torch.tensor(0.1, device=args.device, requires_grad=True)

            # L2正则化
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)

            # 日志和优化
            log_json = json.dumps({
                'global_step': global_step, 'loss': loss.item(), 
                'epoch': epoch, 'time': time.time()
            })
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)
            writer.add_scalar('Loss/train', loss.item(), global_step)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        # 验证
        model.eval()
        valid_loss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_loader, total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                token_type = token_type.to(args.device)
                next_token_type = next_token_type.to(args.device)
                
                pos_logits, neg_logits = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type,
                    seq_feat, pos_feat, neg_feat
                )
                
                # 提取embeddings用于损失计算
                log_feats = model.log2feats(seq, token_type, seq_feat)
                loss_mask = (next_token_type == 1).to(args.device)
                
                # 只计算有效位置的损失
                valid_positions = loss_mask.bool()
                if valid_positions.sum() > 0:  # 确保有有效位置
                    seq_embs = log_feats[valid_positions]
                    pos_embs = model.feat2emb(pos, pos_feat, include_user=False)[valid_positions]
                    neg_embs = model.feat2emb(neg, neg_feat, include_user=False)[valid_positions]
                    
                    # 计算损失
                    loss = compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, args.temperature)
                else:
                    # 如果没有有效位置，使用一个小的虚拟损失
                    loss = torch.tensor(0.1, device=args.device)
                valid_loss_sum += loss.item()
        
        valid_loss = valid_loss_sum / len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss, global_step)
        print(f"Epoch {epoch} valid loss: {valid_loss:.4f}")

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Training done")
    writer.close()
    log_file.close()
    