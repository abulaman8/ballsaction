import os
import csv
import json
from SoccerNet.utils import getListGames

TOLERANCE_SECONDS = 15.0
NMS_WINDOW = 15.0

def temporal_nms(detections, nms_window=15.0):
    if not detections: return []
    detections.sort(key=lambda x: x[2], reverse=True)
    kept = []
    suppressed = set()
    for i, det in enumerate(detections):
        if i in suppressed: continue
        kept.append(det)
        t_center = (det[0] + det[1]) / 2
        for j in range(i+1, len(detections)):
            if j in suppressed: continue
            other_center = (detections[j][0] + detections[j][1]) / 2
            if abs(t_center - other_center) < nms_window:
                suppressed.add(j)
    kept.sort(key=lambda x: x[0])
    return kept

def parse_labels_v2(label_path):
    if not os.path.exists(label_path): return [], []
    with open(label_path, 'r') as f: data = json.load(f)
    gt_fouls = []
    gt_sps = []
    for ann in data.get('annotations', []):
        label = ann.get('label', '')
        time_str = ann.get('gameTime', '')
        if not time_str: continue
        parts = time_str.split(' - ')
        if len(parts) != 2: continue
        half = int(parts[0])
        mm, ss = map(int, parts[1].split(':'))
        sec = mm * 60 + ss
        if label in ['Foul', 'Yellow card', 'Red card', 'Yellow->red card']:
            gt_fouls.append((half, sec))
        elif label in ['Direct free-kick', 'Penalty']:
            gt_sps.append((half, sec))
    return gt_fouls, gt_sps

def evaluate_thresholds(csv_log_path="soccernet_eval_v3_log.csv", soccernet_dir="soccernet_data"):
    if not os.path.exists(csv_log_path):
        print(f"Error: {csv_log_path} not found. Run soccernet_eval_v3.py first.")
        return

    # Read window log
    matches = {}
    with open(csv_log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row['match']
            h = int(row['half'])
            key = (m, h)
            if key not in matches:
                matches[key] = []
            
            raw_foul = float(row.get('raw_foul_prob', row.get('foul_prob', 0)))
            raw_sp = float(row.get('raw_sp_prob', row.get('sp_prob', 0)))
            audio = float(row['audio_prob'])
            w_start = float(row['window_start'])
            w_end = float(row['window_end'])
            
            matches[key].append({
                'start': w_start,
                'end': w_end,
                'raw_foul': raw_foul,
                'raw_sp': raw_sp,
                'audio': audio
            })
            
    # Load ground truths
    unique_games = sorted(list(set(m[0] for m in matches.keys())))
    gts_by_match = {}
    for g in unique_games:
        label_path = os.path.join(soccernet_dir, g, "Labels-v2.json")
        gt_f, gt_s = parse_labels_v2(label_path)
        gts_by_match[g] = (gt_f, gt_s)

    print(f"\nLoaded {len(matches)} halves across {len(unique_games)} matches.")
    print("=" * 80)
    print("GRID SEARCH: SWEEPING AUDIO THRESHOLD & BOOST PARAMETERS")
    print("=" * 80)
    
    # 1. FOUL SWEEP
    print("\n--- TOP CONFIGURATIONS FOR FOUL MODEL ---")
    foul_configs = []
    
    audio_thresholds = [0.50, 0.60, 0.70, 0.80]
    audio_bonuses = [0.00, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25]
    video_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
    
    for a_th in audio_thresholds:
        for bonus in audio_bonuses:
            for v_th in video_thresholds:
                total_tp, total_fp, total_fn = 0, 0, 0
                for (game, half), windows in matches.items():
                    gt_f, _ = gts_by_match[game]
                    half_gt = [sec for h, sec in gt_f if h == half]
                    
                    # Detect
                    dets = []
                    for w in windows:
                        prob = w['raw_foul']
                        if w['audio'] > a_th:
                            prob = min(1.0, prob + bonus)
                        if prob > v_th:
                            dets.append((w['start'], w['end'], prob, 'Foul'))
                    
                    kept = temporal_nms(dets, NMS_WINDOW)
                    matched_gt = set()
                    for p_start, p_end, p_prob, _ in kept:
                        hit = [g for g in half_gt if p_start - TOLERANCE_SECONDS <= g <= p_end + TOLERANCE_SECONDS]
                        if hit:
                            total_tp += 1
                            for g in hit: matched_gt.add(g)
                        else:
                            total_fp += 1
                    total_fn += len(half_gt) - len(matched_gt)
                    
                prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                
                foul_configs.append({
                    'video_th': v_th,
                    'audio_th': a_th,
                    'bonus': bonus,
                    'tp': total_tp,
                    'fp': total_fp,
                    'fn': total_fn,
                    'prec': prec,
                    'rec': rec,
                    'f1': f1
                })
                
    foul_configs.sort(key=lambda x: x['f1'], reverse=True)
    print(f"{'Rank':<5} | {'Vid Thresh':<10} | {'Aud Thresh':<10} | {'Bonus':<6} | {'TP':<4} {'FP':<4} {'FN':<4} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9}")
    print("-" * 85)
    for i, c in enumerate(foul_configs[:10]):
        print(f"#{i+1:<4} | {c['video_th']:<10.2f} | {c['audio_th']:<10.2f} | {c['bonus']:<6.2f} | {c['tp']:<4} {c['fp']:<4} {c['fn']:<4} | {c['prec']*100:>7.2f}% | {c['rec']*100:>7.2f}% | {c['f1']*100:>7.2f}%")

    # 2. SET-PIECE SWEEP
    print("\n--- TOP CONFIGURATIONS FOR SET-PIECE MODEL ---")
    sp_configs = []
    sp_video_thresholds = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
    
    sp_audio_bonuses = [0.00, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25]
    for a_th in audio_thresholds:
        for bonus in sp_audio_bonuses:
            for v_th in sp_video_thresholds:
                total_tp, total_fp, total_fn = 0, 0, 0
                for (game, half), windows in matches.items():
                    _, gt_sp = gts_by_match[game]
                    half_gt = [sec for h, sec in gt_sp if h == half]
                    
                    dets = []
                    for w in windows:
                        prob = w['raw_sp']
                        if w['audio'] > a_th:
                            prob = min(1.0, prob + bonus)
                        if prob > v_th:
                            dets.append((w['start'], w['end'], prob, 'SetPiece'))
                            
                    kept = temporal_nms(dets, NMS_WINDOW)
                    matched_gt = set()
                    for p_start, p_end, p_prob, _ in kept:
                        hit = [g for g in half_gt if p_start - TOLERANCE_SECONDS <= g <= p_end + TOLERANCE_SECONDS]
                        if hit:
                            total_tp += 1
                            for g in hit: matched_gt.add(g)
                        else:
                            total_fp += 1
                    total_fn += len(half_gt) - len(matched_gt)
                    
                prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                
                sp_configs.append({
                    'video_th': v_th,
                    'audio_th': a_th,
                    'bonus': bonus,
                    'tp': total_tp,
                    'fp': total_fp,
                    'fn': total_fn,
                    'prec': prec,
                    'rec': rec,
                    'f1': f1
                })
                
    sp_configs.sort(key=lambda x: x['f1'], reverse=True)
    print(f"{'Rank':<5} | {'Vid Thresh':<10} | {'Aud Thresh':<10} | {'Bonus':<6} | {'TP':<4} {'FP':<4} {'FN':<4} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9}")
    print("-" * 85)
    for i, c in enumerate(sp_configs[:10]):
        print(f"#{i+1:<4} | {c['video_th']:<10.2f} | {c['audio_th']:<10.2f} | {c['bonus']:<6.2f} | {c['tp']:<4} {c['fp']:<4} {c['fn']:<4} | {c['prec']*100:>7.2f}% | {c['rec']*100:>7.2f}% | {c['f1']*100:>7.2f}%")

if __name__ == "__main__":
    import sys
    log_path = sys.argv[1] if len(sys.argv) > 1 else "soccernet_eval_v3_log.csv"
    evaluate_thresholds(csv_log_path=log_path)
