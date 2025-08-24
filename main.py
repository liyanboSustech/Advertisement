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
    parser.add_argument('--hidden_units', default=256, type=int)
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
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, 
                       choices=[str(s) for s in range(81, 87)])
    return parser.parse_args()

def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask):
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
    logits = logits[loss_mask.bool()] / self.temp
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
    
    # 数据加载器（启用多线程）
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, collate_fn=dataset.collate_fn  # num_workers>0启用多线程
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, collate_fn=dataset.collate_fn
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

    # 优化器和损失函数
    # 这里如何用你提供的 compute_infonce_loss 方法呢
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean') 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

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
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type,
                seq_feat, pos_feat, neg_feat
            )
            
            # 计算损失
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            indices = (next_token_type == 1).nonzero().t()  # 过滤有效样本
            loss = bce_criterion(pos_logits[indices[0], indices[1]], pos_labels[indices[0], indices[1]])
            loss += bce_criterion(neg_logits[indices[0], indices[1]], neg_labels[indices[0], indices[1]])

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
            # L2正则化
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
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
                
                pos_labels = torch.ones(pos_logits.shape, device=args.device)
                neg_labels = torch.zeros(neg_logits.shape, device=args.device)
                indices = (next_token_type == 1).nonzero().t()
                loss = bce_criterion(pos_logits[indices[0], indices[1]], pos_labels[indices[0], indices[1]])
                loss += bce_criterion(neg_logits[indices[0], indices[1]], neg_labels[indices[0], indices[1]])
                valid_loss_sum += loss.item()
        
        valid_loss = valid_loss_sum / len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss, global_step)
        print(f"Epoch {epoch} valid loss: {valid_loss:.4f}")

        # 保存模型
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), 
                           f"epoch={epoch}.valid_loss={valid_loss:.4f}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")

    print("Training done")
    writer.close()
    log_file.close()