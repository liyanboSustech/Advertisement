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
from model import HSTUModel  # Changed from BaselineModel to HSTUModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # InfoNCE Loss and Optimizer Params
    parser.add_argument('--temperature', default=0.03, type=float, help='Temperature for InfoNCE loss')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay for AdamW optimizer')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    
    # HSTU specific params
    parser.add_argument('--action_aware', action='store_true', help='Enable action-aware modeling in HSTU')
    parser.add_argument('--hierarchical_levels', default=2, type=int, help='Number of hierarchical levels')
    parser.add_argument('--window_size', default=64, type=int, help='Local attention window size')
    parser.add_argument('--use_rope', action='store_true', help='Use RoPE position encoding')
    parser.add_argument('--rope_base', default=10000, type=int, help='RoPE base frequency')

    args = parser.parse_args()
    return args


def train_epoch(model, train_loader, optimizer, args, writer, log_file, global_step):
    """
    Training loop for one epoch with HSTU enhancements
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
        
        # Move action types to device
        next_action_type = next_action_type.to(args.device)
        
        # Forward pass with action awareness
        log_feats, pos_embs, neg_embs, loss_mask = model(
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
        )
        
        # Compute loss
        loss = model.compute_infonce_loss(log_feats, pos_embs, neg_embs, loss_mask)
        
        # L2 regularization on item embeddings
        for param in model.item_emb.parameters():
            loss += args.l2_emb * torch.norm(param)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Logging
        total_loss += loss.item()
        num_batches += 1
        
        log_json = json.dumps({
            'global_step': global_step + step,
            'loss': loss.item(),
            'time': time.time()
        })
        log_file.write(log_json + '\n')
        log_file.flush()
        print(log_json)

        writer.add_scalar('Loss/train', loss.item(), global_step + step)
    
    return total_loss / num_batches, global_step + len(train_loader)


def validate_epoch(model, valid_loader, args):
    """
    Validation loop for one epoch
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating"):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            
            # Move action types to device
            next_action_type = next_action_type.to(args.device)
            
            # Forward pass
            log_feats, pos_embs, neg_embs, loss_mask = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            
            # Compute loss
            loss = model.compute_infonce_loss(log_feats, pos_embs, neg_embs, loss_mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


if __name__ == '__main__':
    # Setup directories
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    
    # Load data
    data_path = os.environ.get('TRAIN_DATA_PATH')
    args = get_args()
    dataset = MyDataset(data_path, args)
    
    # Split dataset
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, collate_fn=dataset.collate_fn
    )
    
    # Model initialization
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    print(f"Initializing HSTU model with {usernum} users and {itemnum} items")
    print(f"Using RoPE: {args.use_rope}, Window size: {args.window_size}")
    model = HSTUModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.xavier_normal_(param.data)
        elif 'bias' in name:
            torch.nn.init.zeros_(param.data)

    # Zero out padding embeddings
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    # Load pretrained weights if specified
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
            print(f"Loaded model from epoch {epoch_start_idx - 1}")
        except Exception as e:
            print(f'Failed loading state_dicts: {e}')
            raise RuntimeError('Failed loading state_dicts, please check file path!')

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.98), 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=0
    )
    
    # Training metrics
    best_val_loss = float('inf')
    global_step = 0
    
    print(f"Starting HSTU training for {args.num_epochs} epochs")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break
            
        print(f"\n=== Epoch {epoch}/{args.num_epochs} ===")
        
        # Training
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, args, writer, log_file, global_step
        )
        
        # Validation
        val_loss = validate_epoch(model, valid_loader, args)
        
        # Logging
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/valid_epoch', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
    print("Done")
    writer.close()
    log_file.close()