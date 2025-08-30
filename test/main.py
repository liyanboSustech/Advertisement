import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


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

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

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
            
            loss = model.compute_infonce_loss(log_feats, pos_embs, neg_embs, loss_mask)
            
            # L2 regularization on item embeddings
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

        # Step the scheduler
        scheduler.step()

        # Save checkpoint
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH', 'checkpoints'), f"epoch{epoch}_step{global_step}_vloss{valid_loss:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()