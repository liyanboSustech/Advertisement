import argparse
import json
import os
import time
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
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
    print("=========================================================")
    print("before:", seq_embs.shape, pos_embs.shape, neg_embs.shape, loss_mask.shape)
    # L2 normalization
    seq_embs = seq_embs / seq_embs.norm(dim=-1, keepdim=True)
    pos_embs = pos_embs / pos_embs.norm(dim=-1, keepdim=True)
    pos_logits = F.cosine_similarity(seq_embs, pos_embs, dim=-1).unsqueeze(-1)
    writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item())
    # 
    neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)
    # Negative logits: reshape negatives [B, N, D] -> [B, N*D]
    # N是
    print("=========================================================")
    print("after:", seq_embs.shape, pos_embs.shape, neg_embs.shape, loss_mask.shape)
    neg_embedding_all = neg_embs.reshape(-1, hidden_size)  # [B*N, D]
    neg_logits = torch.matmul(seq_embs, neg_embedding_all.transpose(-1, -2))  # [B, B*N]

    writer.add_scalar("Model/nce_neg_logits", neg_logits.mean().item())

    logits = torch.cat([pos_logits, neg_logits], dim=-1)  # [B, 1+N*B]
    print(loss_mask.shape, logits.shape)
    logits = logits[loss_mask.bool()] / temperature
    # labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    # /2025-08-30 11:11:31.154 RuntimeError: Expected floating point type for target with class probabilities, got Long
    # Labels: 0 means the positive sample is always at index 0
    # 如何修改？
    labels = torch.zeros(logits.size(0), dtype=torch.int64, device=logits.device)
    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=128, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    
    # InfoNCE loss parameters
    parser.add_argument('--temp', default=0.07, type=float, help='Temperature parameter for InfoNCE loss')

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
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types,args).to(args.device)

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

    # Set TensorBoard writer for model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            seq_embs, pos_embs, neg_embs, loss_mask = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
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

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss_sum = 0
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

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
