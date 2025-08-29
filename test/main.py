import argparse
import json
import os
import time
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


def compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask,temperature, writer):
    """
    seq_embs: [B, D] sequence embeddings
    pos_embs: [B, D] positive embeddings
    neg_embs: [B, N, D] negative embeddings
    loss_mask: [B] mask to filter valid samples
    """
    hidden_size = neg_embs.size(-1)

    # L2 normalization
    seq_embs = seq_embs / seq_embs.norm(dim=-1, keepdim=True)
    pos_embs = pos_embs / pos_embs.norm(dim=-1, keepdim=True)
    neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)

    # Positive logits (cosine similarity)
    pos_logits = F.cosine_similarity(seq_embs, pos_embs, dim=-1).unsqueeze(-1)

    print("seq_embs shape:", seq_embs.shape)
    print("pos_embs shape:", pos_embs.shape)
    print("neg_embs shape:", neg_embs.shape)

    writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item())

    # Negative logits: reshape negatives [B, N, D] -> [B, N*D]
    neg_embedding_all = neg_embs.reshape(-1, hidden_size)  # [B*N, D]
    neg_logits = torch.matmul(seq_embs, neg_embedding_all.transpose(-1, -2))  # [B, B*N]

    writer.add_scalar("Model/nce_neg_logits", neg_logits.mean().item())

    # Concatenate positive and negative logits
    # print出loss_mask的维度，方便调试
    print("loss_mask shape:", loss_mask.shape)
    logits = torch.cat([pos_logits, neg_logits], dim=-1)  # [B, 1+N*B]
    # # loss_mask的维度是[B]，表示每个样本是否有效
    # # logits的维度是[B, 1+N*B]，表示每个样本的正负样本的相似度
    # # 这里的操作是将loss_mask扩展为与logits相同的维度，然后进行掩码操作
    # # 这样可以确保只有有效样本的logits被用于计算损失

        
    # # Apply mask and temperature scaling
    # logits = logits[loss_mask.bool()] / temperature
``
    # # Labels: 0 means the positive sample is always at index 0
    # labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)

    # # Cross-entropy loss
    # loss = F.cross_entropy(logits, labels)
    logits = logits[loss_mask.bool()] / temperature

    # Ensure logits is always 2-D [B, C]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # Labels: 0 means the positive sample is always at index 0
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    print("Before cross_entropy:")
    print("logits shape:", logits.shape)
    print("labels shape:", labels.shape)
    print("logits dtype:", logits.dtype)
    print("labels dtype:", labels.dtype)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    # Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # InfoNCE Loss and Optimizer Params
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature for InfoNCE loss')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay for AdamW optimizer')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
<<<<<<< HEAD
=======
    
    # InfoNCE loss parameters
    parser.add_argument('--temp', default=0.07, type=float, help='Temperature parameter for InfoNCE loss')
>>>>>>> 28df5a1 (update)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    
    Path(os.environ.get('TRAIN_LOG_PATH', 'logs')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH', 'tf_events')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH', 'logs'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH', 'tf_events'))
    
    data_path = os.environ.get('TRAIN_DATA_PATH')
    dataset = MyDataset(data_path, args)
    
    train_dataset, valid_dataset = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, collate_fn=dataset.collate_fn
    )

    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

<<<<<<< HEAD
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
=======
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types,args).to(args.device)
>>>>>>> 28df5a1 (update)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass 

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except Exception as e:
            print(f'Failed loading state_dicts: {e}')
            raise RuntimeError('Failed loading state_dicts, please check file path!')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)

    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only: break
        
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            
            optimizer.zero_grad()
            
            log_feats, pos_embs, neg_embs, loss_mask = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
<<<<<<< HEAD
            
            loss = model.compute_infonce_loss(log_feats, pos_embs, neg_embs, loss_mask)
            
            # L2 regularization on item embeddings
=======
            optimizer.zero_grad()
            loss = compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, args.temp, writer)

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

>>>>>>> 28df5a1 (update)
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)

            loss.backward()
            optimizer.step()

            log_json = json.dumps({'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()})
            log_file.write(log_json + '\n')
            log_file.flush()
            if step % 100 == 0:
                print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
        
        # Validation loop
        model.eval()
        valid_loss_sum = 0
<<<<<<< HEAD
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validating"):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                
                log_feats, pos_embs, neg_embs, loss_mask = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                
                loss = model.compute_infonce_loss(log_feats, pos_embs, neg_embs, loss_mask)
                valid_loss_sum += loss.item()
        
        valid_loss = valid_loss_sum / len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss, global_step)
        print(f"Epoch {epoch} validation loss: {valid_loss:.4f}")
=======
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            seq_embs, pos_embs, neg_embs, loss_mask = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            loss = compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, args.temp, writer)
            valid_loss_sum += loss.item()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
>>>>>>> 28df5a1 (update)

        # Step the scheduler
        scheduler.step()

        # Save checkpoint
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH', 'checkpoints'), f"epoch{epoch}_step{global_step}_vloss{valid_loss:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()