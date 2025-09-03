import argparse
import json
import os
import time
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--valid_ratio', default=0.1, type=float, help='Validation set ratio')

    # Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    
    # Dropout parameters
    parser.add_argument('--emb_dropout_rate', default=0.2, type=float, help='Dropout rate for embeddings')
    parser.add_argument('--fusion_dropout_rate', default=0.1, type=float, help='Dropout rate for fusion layer')
    parser.add_argument('--ffn_dropout_rate', default=0.05, type=float, help='Dropout rate for FFN layers')
    
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # InfoNCE Loss and Optimizer Params
    parser.add_argument('--temperature', default=0.035, type=float, help='Temperature for InfoNCE loss')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay for AdamW optimizer')
    parser.add_argument('--use_inbatch_negatives', action='store_true', help='Use in-batch negatives')

    # Mixed precision training
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')

    # Validation params
    parser.add_argument('--eval_every', default=1, type=int, help='Evaluate every N epochs')
    parser.add_argument('--eval_k', default=10, type=int, help='K for HitRate@K and NDCG@K')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()
    return args


def create_data_loaders(data_path, args):
    """创建训练和验证数据加载器"""
    train_dataset = MyDataset(data_path, args, dataset_type='train')
    valid_dataset = MyDataset(data_path, args, dataset_type='valid')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        worker_init_fn=train_dataset._worker_init_fn,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=valid_dataset.collate_fn,
        worker_init_fn=valid_dataset._worker_init_fn,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    return train_loader, valid_loader, train_dataset


def train_epoch(model, train_loader, optimizer, scaler, args, writer, epoch, global_step):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        seq, pos, neg, token_type, next_token_type, next_action_type, \
            seq_feat, pos_feat, neg_feat = batch

        optimizer.zero_grad()
        
        if args.use_amp:
            with autocast():
                log_feats, pos_embs, neg_embs, loss_mask = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type,
                    seq_feat, pos_feat, neg_feat
                )
                
                if args.use_inbatch_negatives:
                    loss = model.compute_infonce_loss_with_inbatch(
                        log_feats, pos_embs, neg_embs, loss_mask, writer)
                else:
                    loss = model.compute_infonce_loss(
                        log_feats, pos_embs, neg_embs, loss_mask, writer)
                
                # L2 regularization
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            log_feats, pos_embs, neg_embs, loss_mask = model(
                seq, pos, neg, token_type, next_token_type, next_action_type,
                seq_feat, pos_feat, neg_feat
            )
            
            if args.use_inbatch_negatives:
                loss = model.compute_infonce_loss_with_inbatch(
                    log_feats, pos_embs, neg_embs, loss_mask, writer)
            else:
                loss = model.compute_infonce_loss(
                    log_feats, pos_embs, neg_embs, loss_mask, writer)
            
            # L2 regularization
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 记录日志
        log_json = json.dumps({
            'global_step': global_step,
            'loss': loss.item(),
            'epoch': epoch,
            'time': time.time(),
            'lr': optimizer.param_groups[0]['lr']
        })
        
        writer.add_scalar('Loss/train', loss.item(), global_step)
        global_step += 1
        
        # 可以选择性地减少日志输出频率
        if step % 100 == 0:
            print(log_json)
    
    avg_loss = total_loss / num_batches
    return avg_loss, global_step


def validate_model(model, valid_loader, args):
    """验证模型并计算指标"""
    hit_rate, ndcg = model.evaluate_metrics(valid_loader, k=args.eval_k)
    return hit_rate, ndcg


if __name__ == '__main__':
    # 设置日志路径
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))

    data_path = os.environ.get('TRAIN_DATA_PATH')
    args = get_args()
    
    print("=== Model Configuration ===")
    print(f"Embedding dropout rate: {args.emb_dropout_rate}")
    print(f"Fusion dropout rate: {args.fusion_dropout_rate}")
    print(f"FFN dropout rate: {args.ffn_dropout_rate}")
    print(f"Use mixed precision: {args.use_amp}")
    print(f"Use in-batch negatives: {args.use_inbatch_negatives}")
    print(f"Validation ratio: {args.valid_ratio}")
    
    # 创建数据加载器
    train_loader, valid_loader, train_dataset = create_data_loaders(data_path, args)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_loader.dataset)}")
    
    usernum, itemnum = train_dataset.usernum, train_dataset.itemnum
    feat_statistics, feat_types = train_dataset.feat_statistics, train_dataset.feature_types

    # 初始化模型
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # 参数初始化
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    # 加载预训练模型
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except Exception as e:
            print(f'Failed loading state_dicts: {e}')
            raise RuntimeError('Failed loading state_dicts, please check file path!')

    # 初始化优化器和混合精度scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                  betas=(0.9, 0.98), weight_decay=args.weight_decay)
    scaler = GradScaler() if args.use_amp else None

    # 学习率调度表
    lr_schedule = {
        1: 0.001,
        2: 0.001,
        3: 0.0005,
        4: 0.0005,
        5: 0.00025,
    }

    best_hit_rate = 0.0
    best_ndcg = 0.0
    best_epoch = 0
    global_step = 0
    t0 = time.time()

    print("Start training")

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break

        # 设置学习率
        lr = lr_schedule.get(epoch, 0.00025)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Epoch {epoch} using lr = {lr}")

        # 训练
        avg_train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scaler, args, writer, epoch, global_step
        )
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")

        # 验证
        if epoch % args.eval_every == 0:
            print("Evaluating...")
            hit_rate, ndcg = validate_model(model, valid_loader, args)
            
            writer.add_scalar(f'Metrics/HitRate@{args.eval_k}', hit_rate, epoch)
            writer.add_scalar(f'Metrics/NDCG@{args.eval_k}', ndcg, epoch)
            
            print(f"Epoch {epoch} - HitRate@{args.eval_k}: {hit_rate:.4f}, NDCG@{args.eval_k}: {ndcg:.4f}")
            
            # 保存最佳模型
            if hit_rate > best_hit_rate:
                best_hit_rate = hit_rate
                best_ndcg = ndcg
                best_epoch = epoch
                
                # 保存最佳模型
                best_model_path = Path(os.environ.get('TRAIN_CKPT_PATH'), 'best_model')
                best_model_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_model_path / "model.pt")
                
                # 保存指标信息
                metrics_info = {
                    'epoch': epoch,
                    'hit_rate': hit_rate,
                    'ndcg': ndcg,
                    'train_loss': avg_train_loss
                }
                with open(best_model_path / "metrics.json", 'w') as f:
                    json.dump(metrics_info, f, indent=2)
                
                print(f"New best model saved! HitRate@{args.eval_k}: {hit_rate:.4f}")

        # 保存定期checkpoint
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), 
                       f"epoch_{epoch}_step_{global_step}_loss_{avg_train_loss:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        
        # 记录epoch日志
        epoch_log = json.dumps({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'lr': lr,
            'time': time.time() - t0
        })
        log_file.write(epoch_log + '\n')
        log_file.flush()

    # 训练结束总结
    print("=" * 50)
    print("Training completed!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best HitRate@{args.eval_k}: {best_hit_rate:.4f}")
    print(f"Best NDCG@{args.eval_k}: {best_ndcg:.4f}")
    print(f"Total training time: {time.time() - t0:.2f} seconds")

    writer.close()
    log_file.close()