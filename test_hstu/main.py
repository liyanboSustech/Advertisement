import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import HSTUModel
import torch.nn.functional as F

def compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, writer):
    """
    计算InfoNCE损失函数x

    Args:
        seq_embs: 序列embeddings [batch_size, hidden_size]
        pos_embs: 正样本embeddings [batch_size, hidden_size]
        neg_embs: 负样本embeddings [batch_size, num_negatives, hidden_size]
        loss_mask: 损失掩码 [batch_size]

    Returns:
        loss: InfoNCE损失值
   
        计算InfoNCE损失函数
        
        Args:
            seq_embs: 序列embeddings [batch_size, hidden_size]
            pos_embs: 正样本embeddings [batch_size, hidden_size] 
            neg_embs: 负样本embeddings [batch_size, num_negatives, hidden_size]
            loss_mask: 损失掩码 [batch_size]
        
        Returns:
            loss: InfoNCE损失值
    """
    hidden_size = neg_embs.size(-1)
    # 归一化序列embeddings
    seq_embs = seq_embs / seq_embs.norm(dim=-1, keepdim=True)
    # 归一化正样本embeddings
    pos_embs = pos_embs / pos_embs.norm(dim=-1, keepdim=True)
    # 计算正样本logits (余弦相似度)
    pos_logits = F.cosine_similarity(seq_embs, pos_embs, dim=-1).unsqueeze(-1)
    # 记录正样本logits到tensorboard
    writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item())
    # 归一化负样本embeddings
    neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)
    # 将负样本reshape为矩阵乘法格式
    neg_embedding_all = neg_embs.reshape(-1, hidden_size)
    # 计算负样本logits (批量矩阵乘法)
    neg_logits = torch.matmul(seq_embs, neg_embedding_all.transpose(-1, -2))
    # 记录负样本logits到tensorboard
    writer.add_scalar("Model/nce_neg_logits", neg_logits.mean().item())
    # 拼接正负样本logits
    logits = torch.cat([pos_logits, neg_logits], dim=-1)
    # 应用温度参数和损失掩码
    logits = logits[loss_mask.bool()] / args.temperature
    # 创建标签 (正样本的索引总是0)
    labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
    # 计算交叉熵损失
    loss = F.cross_entropy(logits, labels)
    
    return loss


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--T_max', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--temperature', default=0.07, type=float)
    parser.add_argument('--infonce_temp', default=0.1, type=float)
    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = HSTUModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    # Note: HSTUModel doesn't have pos_emb, item_emb, or user_emb attributes
    # Position embeddings are handled internally by the input_preprocessor module

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    print(f"Cosine annealing: T_max={args.T_max}, initial_lr={args.lr}")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        
        print(f"Epoch {epoch}: Starting training")
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat_tensors, pos_feat_tensors, neg_feat_tensors = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            token_type = token_type.to(args.device)
            next_token_type = next_token_type.to(args.device)
            next_action_type = next_action_type.to(args.device)
            
            # 特征张量已经在dataset中处理好了，直接传递给模型
            seq_embs, pos_embs, neg_embs, loss_mask = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, 
                seq_feat_tensors, pos_feat_tensors, neg_feat_tensors
            )
            optimizer.zero_grad()
            loss = model.compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, args.temperature, writer)

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)

            global_step += 1

            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        valid_loss_sum = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat_tensors, pos_feat_tensors, neg_feat_tensors = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            token_type = token_type.to(args.device)
            next_token_type = next_token_type.to(args.device)
            next_action_type = next_action_type.to(args.device)
            
            seq_embs, pos_embs, neg_embs, loss_mask = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, 
                seq_feat_tensors, pos_feat_tensors, neg_feat_tensors
            )
            loss = model.compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, args.temperature, writer)
            valid_loss_sum += loss.item()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
