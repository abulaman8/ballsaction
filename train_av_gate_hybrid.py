import os
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SoccerNet.utils import getListGames
from model_av_gate import AVGateNet, extract_gate_features

TOLERANCE_SECONDS = 10.0 # 20s center-point window (|t_pred - t_gt| <= 10.0s)
NMS_WINDOW = 15.0

def temporal_nms(detections, nms_window=15.0):
    if not detections:
        return []
    detections.sort(key=lambda x: x[2], reverse=True)
    kept = []
    suppressed = set()
    for i, det in enumerate(detections):
        if i in suppressed:
            continue
        kept.append(det)
        t_center = (det[0] + det[1]) / 2.0
        for j in range(i + 1, len(detections)):
            if j in suppressed:
                continue
            other_center = (detections[j][0] + detections[j][1]) / 2.0
            if abs(t_center - other_center) < nms_window:
                suppressed.add(j)
    kept.sort(key=lambda x: x[0])
    return kept

def parse_labels_v2(json_path):
    if not os.path.exists(json_path):
        return [], []
    with open(json_path, 'r') as f:
        data = json.load(f)
    gt_fouls = []
    gt_sps = []
    for ann in data.get('annotations', []):
        label = ann.get('label', '')
        time_str = ann.get('gameTime', '')
        if not time_str:
            continue
        parts = time_str.split(' - ')
        if len(parts) != 2:
            continue
        half = int(parts[0])
        mm, ss = map(int, parts[1].split(':'))
        sec = mm * 60 + ss
        
        if label in ['Foul', 'Yellow card', 'Red card', 'Yellow->red card']:
            gt_fouls.append((half, sec))
        elif label in ['Direct free-kick', 'Penalty']:
            gt_sps.append((half, sec))
            
    return gt_fouls, gt_sps

def load_dataset_from_log(csv_log_path, soccernet_dir="soccernet_data"):
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
                'audio': float(row.get('audio_prob', 0))
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
        
        target_f = []
        target_sp = []
        for w in windows:
            w_center = (w['start'] + w['end']) / 2.0
            hit_f = any(abs(w_center - g) <= TOLERANCE_SECONDS for g in half_gt_f)
            hit_sp = any(abs(w_center - g) <= TOLERANCE_SECONDS for g in half_gt_sp)
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

def evaluate_gate_on_halves(model, halves_data, foul_thresh=0.80, sp_thresh=0.75, device='cpu'):
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
            
            # Foul detections
            dets_f = []
            for i, w in enumerate(d['windows']):
                if p_f[i] > foul_thresh:
                    dets_f.append((w['start'], w['end'], float(p_f[i]), 'Foul'))
            kept_f = temporal_nms(dets_f, NMS_WINDOW)
            matched_gt_f = set()
            for p_start, p_end, _, _ in kept_f:
                p_center = (p_start + p_end) / 2.0
                hit = [g for g in d['gt_f'] if abs(p_center - g) <= TOLERANCE_SECONDS]
                if hit:
                    tp_f += 1
                    for g in hit:
                        matched_gt_f.add(g)
                else:
                    fp_f += 1
            fn_f += len(d['gt_f']) - len(matched_gt_f)
            
            # Set-piece detections
            dets_sp = []
            for i, w in enumerate(d['windows']):
                if p_sp[i] > sp_thresh:
                    dets_sp.append((w['start'], w['end'], float(p_sp[i]), 'SetPiece'))
            kept_sp = temporal_nms(dets_sp, NMS_WINDOW)
            matched_gt_sp = set()
            for p_start, p_end, _, _ in kept_sp:
                p_center = (p_start + p_end) / 2.0
                hit = [g for g in d['gt_sp'] if abs(p_center - g) <= TOLERANCE_SECONDS]
                if hit:
                    tp_sp += 1
                    for g in hit:
                        matched_gt_sp.add(g)
                else:
                    fp_sp += 1
            fn_sp += len(d['gt_sp']) - len(matched_gt_sp)
            
    p_prec = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0
    p_rec = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
    f1_foul = 2 * p_prec * p_rec / (p_prec + p_rec) if (p_prec + p_rec) > 0 else 0.0
    
    sp_prec = tp_sp / (tp_sp + fp_sp) if (tp_sp + fp_sp) > 0 else 0.0
    sp_rec = tp_sp / (tp_sp + fn_sp) if (tp_sp + fn_sp) > 0 else 0.0
    f1_sp = 2 * sp_prec * sp_rec / (sp_prec + sp_rec) if (sp_prec + sp_rec) > 0 else 0.0
    
    return {
        'foul': {'tp': tp_f, 'fp': fp_f, 'fn': fn_f, 'prec': p_prec, 'rec': p_rec, 'f1': f1_foul},
        'sp': {'tp': tp_sp, 'fp': fp_sp, 'fn': fn_sp, 'prec': sp_prec, 'rec': sp_rec, 'f1': f1_sp},
        'comb_f1': (f1_foul + f1_sp) / 2.0
    }

def train_gate(train_csv="soccernet_valid_hybrid_log.csv", epochs=80, lr=1e-3, save_path="checkpoints/av_gate_hybrid_best.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading data from {train_csv}...")
    all_halves = load_dataset_from_log(train_csv)
    print(f"Loaded {len(all_halves)} halves total.")
    
    unique_games = sorted(list(set(d['game'] for d in all_halves)))
    val_game_count = max(1, len(unique_games) // 3)
    val_games = set(unique_games[-val_game_count:])
    train_games = set(unique_games[:-val_game_count])
    
    train_halves = [d for d in all_halves if d['game'] in train_games]
    val_halves = [d for d in all_halves if d['game'] in val_games]
    print(f"Gate Split: {len(train_halves)} halves train ({len(train_games)} games), {len(val_halves)} halves val ({len(val_games)} games).")
    
    model = AVGateNet(in_features=15, hidden_dim=32, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    best_val_combined_f1 = 0.0
    
    print("\nStarting AVGateNet Hybrid Training...")
    print("=" * 75)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
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
            
            loss_f = F.binary_cross_entropy_with_logits(logit_f, target_f, pos_weight=torch.tensor([3.5], device=device))
            loss_sp = F.binary_cross_entropy_with_logits(logit_sp, target_sp, pos_weight=torch.tensor([12.0], device=device))
            loss = loss_f + loss_sp
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_halves)
        
        if epoch % 5 == 0 or epoch == epochs:
            val_metrics = evaluate_gate_on_halves(model, val_halves, foul_thresh=0.80, sp_thresh=0.75, device=device)
            f_m = val_metrics['foul']
            sp_m = val_metrics['sp']
            comb_f1 = val_metrics['comb_f1']
            
            print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | "
                  f"Val Foul F1: {f_m['f1']*100:.2f}% (P: {f_m['prec']*100:.2f}%, R: {f_m['rec']*100:.2f}%) | "
                  f"Val SP F1: {sp_m['f1']*100:.2f}% (P: {sp_m['prec']*100:.2f}%, R: {sp_m['rec']*100:.2f}%) | "
                  f"Comb: {comb_f1*100:.2f}%")
                  
            if comb_f1 > best_val_combined_f1:
                best_val_combined_f1 = comb_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_foul_f1': f_m['f1'],
                    'val_sp_f1': sp_m['f1'],
                    'val_combined_f1': comb_f1,
                    'val_metrics': val_metrics
                }, save_path)
                print(f"  --> Saved new best AVGate model to {save_path} (Comb F1: {comb_f1*100:.2f}%)")
                
    print("\nAVGateNet Hybrid Training Completed.")
    print(f"Best Validation Combined F1: {best_val_combined_f1*100:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, default="soccernet_valid_hybrid_log.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="checkpoints/av_gate_hybrid_best.pth")
    args = parser.parse_args()
    
    train_gate(train_csv=args.train_csv, epochs=args.epochs, lr=args.lr, save_path=args.save_path)
