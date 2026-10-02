import os
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SoccerNet.utils import getListGames
from model_av_gate import AVGateNet, extract_gate_features
from sweep_audio_boost import parse_labels_v2, temporal_nms, TOLERANCE_SECONDS, NMS_WINDOW

def load_dataset_from_log(csv_log_path, soccernet_dir="soccernet_data"):
    """
    Loads window logs grouped by (match, half), parses GT labels,
    and extracts feature vectors and ground truth binary targets.
    """
    if not os.path.exists(csv_log_path):
        raise FileNotFoundError(f"{csv_log_path} not found.")

    matches = {}
    with open(csv_log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row['match']
            h = int(row['half'])
            key = (m, h)
            if key not in matches:
                matches[key] = []
            matches[key].append({
                'start': float(row['window_start']),
                'end': float(row['window_end']),
                'raw_foul': float(row.get('raw_foul_prob', 0)),
                'raw_sp': float(row.get('raw_sp_prob', 0)),
                'audio': float(row['audio_prob'])
            })

    unique_games = sorted(list(set(m[0] for m in matches.keys())))
    gts_by_match = {g: parse_labels_v2(os.path.join(soccernet_dir, g, 'Labels-v2.json')) for g in unique_games}

    halves_data = []
    for (game, half), windows in matches.items():
        gt_f, gt_sp = gts_by_match[game]
        half_gt_f = [sec for h, sec in gt_f if h == half]
        half_gt_sp = [sec for h, sec in gt_sp if h == half]
        
        feats = extract_gate_features(windows)
        raw_vf = torch.tensor([w['raw_foul'] for w in windows], dtype=torch.float32)
        raw_vsp = torch.tensor([w['raw_sp'] for w in windows], dtype=torch.float32)
        
        # Binary target: 1 if window overlaps a ground truth event
        target_f = []
        target_sp = []
        for w in windows:
            hit_f = any(w['start'] - TOLERANCE_SECONDS <= g <= w['end'] + TOLERANCE_SECONDS for g in half_gt_f)
            hit_sp = any(w['start'] - TOLERANCE_SECONDS <= g <= w['end'] + TOLERANCE_SECONDS for g in half_gt_sp)
            target_f.append(1.0 if hit_f else 0.0)
            target_sp.append(1.0 if hit_sp else 0.0)
            
        halves_data.append({
            'game': game,
            'half': half,
            'windows': windows,
            'feats': feats,
            'raw_vf': raw_vf,
            'raw_vsp': raw_vsp,
            'target_f': torch.tensor(target_f, dtype=torch.float32),
            'target_sp': torch.tensor(target_sp, dtype=torch.float32),
            'gt_f': half_gt_f,
            'gt_sp': half_gt_sp
        })
        
    return halves_data

def evaluate_gate_on_halves(model, halves_data, foul_thresh=0.85, sp_thresh=0.85, device='cpu'):
    """
    Runs inference through the gate and evaluates event-level Precision, Recall, and F1.
    """
    model.eval()
    
    tp_f, fp_f, fn_f = 0, 0, 0
    tp_sp, fp_sp, fn_sp = 0, 0, 0
    
    with torch.no_grad():
        for d in halves_data:
            feats = d['feats'].to(device)
            raw_vf = d['raw_vf'].to(device)
            raw_vsp = d['raw_vsp'].to(device)
            
            p_f, p_sp = model(feats, raw_vf, raw_vsp)
            p_f = p_f.cpu().numpy()
            p_sp = p_sp.cpu().numpy()
            
            # FOUL DETECTIONS
            dets_f = []
            for i, w in enumerate(d['windows']):
                if p_f[i] > foul_thresh:
                    dets_f.append((w['start'], w['end'], float(p_f[i]), 'Foul'))
                    
            kept_f = temporal_nms(dets_f, NMS_WINDOW)
            matched_gt_f = set()
            for p_start, p_end, p_prob, _ in kept_f:
                hit = [g for g in d['gt_f'] if p_start - TOLERANCE_SECONDS <= g <= p_end + TOLERANCE_SECONDS]
                if hit:
                    tp_f += 1
                    for g in hit: matched_gt_f.add(g)
                else:
                    fp_f += 1
            fn_f += len(d['gt_f']) - len(matched_gt_f)
            
            # SETPIECE DETECTIONS
            dets_sp = []
            for i, w in enumerate(d['windows']):
                if p_sp[i] > sp_thresh:
                    dets_sp.append((w['start'], w['end'], float(p_sp[i]), 'SetPiece'))
                    
            kept_sp = temporal_nms(dets_sp, NMS_WINDOW)
            matched_gt_sp = set()
            for p_start, p_end, p_prob, _ in kept_sp:
                hit = [g for g in d['gt_sp'] if p_start - TOLERANCE_SECONDS <= g <= p_end + TOLERANCE_SECONDS]
                if hit:
                    tp_sp += 1
                    for g in hit: matched_gt_sp.add(g)
                else:
                    fp_sp += 1
            fn_sp += len(d['gt_sp']) - len(matched_gt_sp)
            
    p_prec = tp_f / (tp_f + fp_f) if tp_f + fp_f > 0 else 0
    p_rec = tp_f / (tp_f + fn_f) if tp_f + fn_f > 0 else 0
    f1_foul = 2 * p_prec * p_rec / (p_prec + p_rec) if p_prec + p_rec > 0 else 0
    
    sp_prec = tp_sp / (tp_sp + fp_sp) if tp_sp + fp_sp > 0 else 0
    sp_rec = tp_sp / (tp_sp + fn_sp) if tp_sp + fn_sp > 0 else 0
    f1_sp = 2 * sp_prec * sp_rec / (sp_prec + sp_rec) if sp_prec + sp_rec > 0 else 0
    
    return {
        'foul': {'tp': tp_f, 'fp': fp_f, 'fn': fn_f, 'prec': p_prec, 'rec': p_rec, 'f1': f1_foul},
        'sp': {'tp': tp_sp, 'fp': fp_sp, 'fn': fn_sp, 'prec': sp_prec, 'rec': sp_rec, 'f1': f1_sp}
    }

def train_gate(train_csv="soccernet_valid_log.csv", test_csv="soccernet_eval_v3_log.csv", epochs=80, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading training data from {train_csv}...")
    train_halves = load_dataset_from_log(train_csv)
    print(f"Loaded {len(train_halves)} halves for training.")
    
    test_halves = None
    if os.path.exists(test_csv):
        print(f"Loading test data from {test_csv} for validation tracking...")
        test_halves = load_dataset_from_log(test_csv)
        print(f"Loaded {len(test_halves)} test halves.")
        
    model = AVGateNet(in_features=15, hidden_dim=32, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    os.makedirs("checkpoints", exist_ok=True)
    best_combined_f1 = 0.0
    best_foul_f1 = 0.0
    
    print("\nStarting AVGateNet Training...")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        # Permute halves for stochasticity
        perm = torch.randperm(len(train_halves))
        for idx in perm:
            d = train_halves[idx]
            feats = d['feats'].to(device)
            raw_vf = d['raw_vf'].to(device)
            raw_vsp = d['raw_vsp'].to(device)
            target_f = d['target_f'].to(device)
            target_sp = d['target_sp'].to(device)
            
            optimizer.zero_grad()
            logit_f, logit_sp = model(feats, raw_vf, raw_vsp, return_logits=True)
            
            # Loss on window predictions with balanced pos_weight
            loss_f = F.binary_cross_entropy_with_logits(logit_f, target_f, pos_weight=torch.tensor([2.0], device=device))
            loss_sp = F.binary_cross_entropy_with_logits(logit_sp, target_sp, pos_weight=torch.tensor([2.0], device=device))
            loss = loss_f + loss_sp
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_halves)
        
        # Evaluate on test set every 5 epochs or last epoch
        if epoch % 5 == 0 or epoch == epochs:
            eval_metrics = evaluate_gate_on_halves(model, test_halves if test_halves else train_halves,
                                                   foul_thresh=0.82, sp_thresh=0.85, device=device)
            f_m = eval_metrics['foul']
            sp_m = eval_metrics['sp']
            comb_f1 = (f_m['f1'] + sp_m['f1']) / 2.0
            
            print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | "
                  f"Foul F1: {f_m['f1']*100:.2f}% (P: {f_m['prec']*100:.2f}%, R: {f_m['rec']*100:.2f}%) | "
                  f"SP F1: {sp_m['f1']*100:.2f}% (P: {sp_m['prec']*100:.2f}%, R: {sp_m['rec']*100:.2f}%) | "
                  f"Comb: {comb_f1*100:.2f}%")
                  
            if comb_f1 > best_combined_f1:
                best_combined_f1 = comb_f1
                best_foul_f1 = f_m['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'foul_f1': f_m['f1'],
                    'sp_f1': sp_m['f1'],
                    'combined_f1': comb_f1,
                    'eval_metrics': eval_metrics
                }, "checkpoints/av_gate_best.pth")
                print(f"  --> Saved new best AVGate model to checkpoints/av_gate_best.pth (Comb F1: {comb_f1*100:.2f}%)")

    print("\nTraining completed.")
    print(f"Best Combined F1: {best_combined_f1*100:.2f}% | Best Foul F1: {best_foul_f1*100:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, default="soccernet_valid_15s_log.csv")
    parser.add_argument("--test-csv", type=str, default="soccernet_eval_15s_test_log.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="checkpoints/av_gate_best.pth")
    args = parser.parse_args()
    
    train_gate(train_csv=args.train_csv, test_csv=args.test_csv, epochs=args.epochs, lr=args.lr)
