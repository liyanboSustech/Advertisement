import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import MyDataset, MyTestDataset
from model import BaselineModel


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='AdamW权重衰减系数')
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
    parser.add_argument('--temperature', default=0.07, type=float ,help='Temperature for InfoNCE loss')
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, 
                       choices=[str(s) for s in range(81, 87)])
    return parser.parse_args()

def compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, temperature=0.07):
    """计算InfoNCE损失（使用全局负样本）"""
    # 标准化embedding向量（确保余弦相似度有效）
    seq_embs = F.normalize(seq_embs, dim=-1)
    pos_embs = F.normalize(pos_embs, dim=-1)
    neg_embs = F.normalize(neg_embs, dim=-1)
    
    # 计算正样本相似度
    pos_logits = (seq_embs * pos_embs).sum(dim=-1, keepdim=True)  # [B, L, 1]
    # 计算负样本相似度（全局负样本）
    neg_logits = (seq_embs * neg_embs).sum(dim=-1, keepdim=True)  # [B, L, 1]
    
    # 合并正负样本logits
    logits = torch.cat([pos_logits, neg_logits], dim=-1)  # [B, L, 2]
    
    # 应用损失掩码（只计算有效位置）
    valid_logits = logits[loss_mask.bool()]  # [valid_len, 2]
    
    # 计算交叉熵损失（正样本标签为0）
    labels = torch.zeros(valid_logits.size(0), device=valid_logits.device, dtype=torch.long)
    loss = F.cross_entropy(valid_logits / temperature, labels)
    
    # 记录日志
    writer.add_scalar("Loss/pos_logits_mean", pos_logits.mean().item())
    writer.add_scalar("Loss/neg_logits_mean", neg_logits.mean().item())
    return loss

def train_epoch(model, train_loader, optimizer, scheduler, args):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch in tqdm(train_loader, desc="Training", total=num_batches):
        # 解包批次数据
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
        
        # 移至设备
        seq = seq.to(args.device)
        pos = pos.to(args.device)
        neg = neg.to(args.device)
        token_type = token_type.to(args.device)
        next_token_type = next_token_type.to(args.device)
        next_action_type = next_action_type.to(args.device)
        
        # 前向传播
        pos_logits, neg_logits, seq_embs, pos_embs, neg_embs, loss_mask = model(
            seq, pos, neg, token_type, next_token_type, next_action_type, 
            seq_feat, pos_feat, neg_feat
        )
        
        # 计算损失
        loss = compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, args.temperature)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 记录批次损失
        global_step = scheduler.last_epoch * num_batches + (len(total_loss) - 1)
        writer.add_scalar("Loss/batch_loss", loss.item(), global_step)
    
    # 每个epoch结束后更新学习率
    scheduler.step()
    
    avg_loss = total_loss / num_batches
    writer.add_scalar("Loss/epoch_loss", avg_loss, scheduler.last_epoch)
    writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], scheduler.last_epoch)
    
    return avg_loss

def evaluate(model, test_loader, args):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    num_batches = len(test_loader)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", total=num_batches):
            # 解包批次数据
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            
            # 移至设备
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            token_type = token_type.to(args.device)
            next_token_type = next_token_type.to(args.device)
            next_action_type = next_action_type.to(args.device)
            
            # 前向传播
            pos_logits, neg_logits, seq_embs, pos_embs, neg_embs, loss_mask = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, 
                seq_feat, pos_feat, neg_feat
            )
            
            # 计算损失
            loss = compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, args.temperature)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss

def main():
    global writer
    args = get_args()
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 加载数据集
    print(f"加载训练数据...")
    train_dataset = MyDataset(args.data_dir, args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"加载测试数据...")
    test_dataset = MyTestDataset(args.data_dir, args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    print(f"初始化模型...")
    model = BaselineModel(
        user_num=train_dataset.usernum,
        item_num=train_dataset.itemnum,
        feat_statistics=train_dataset.feat_statistics,
        feat_types=train_dataset.feature_types,
        args=args
    ).to(args.device)
    
    # 加载预训练模型（如果有）
    if args.state_dict_path and os.path.exists(args.state_dict_path):
        print(f"加载预训练模型: {args.state_dict_path}")
        model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device))
    
    # 如果只是推理，则直接保存物品嵌入
    if args.inference_only:
        print("生成物品嵌入...")
        model.save_item_emb(
            item_ids=list(range(1, train_dataset.itemnum + 1)),
            retrieval_ids=list(range(1, train_dataset.itemnum + 1)),
            feat_dict=train_dataset.item_feat_dict,
            save_path=args.save_dir
        )
        print(f"物品嵌入已保存至 {args.save_dir}")
        return
    
    # 配置优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6  # 最小学习率
    )
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, args)
        print(f"训练损失: {train_loss:.4f}")
        
        # 评估
        test_loss = evaluate(model, test_loader, args)
        print(f"测试损失: {test_loss:.4f}")
        
        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            save_path = os.path.join(args.save_dir, f"best_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"保存最佳模型至 {save_path}")
    
    # 训练结束后保存最终模型
    final_save_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"训练结束，最终模型保存至 {final_save_path}")
    
    # 生成物品嵌入
    print("生成最终物品嵌入...")
    model.save_item_emb(
        item_ids=list(range(1, train_dataset.itemnum + 1)),
        retrieval_ids=list(range(1, train_dataset.itemnum + 1)),
        feat_dict=train_dataset.item_feat_dict,
        save_path=args.save_dir
    )
    print(f"物品嵌入已保存至 {args.save_dir}")
    
    writer.close()

if __name__ == "__main__":
    main()