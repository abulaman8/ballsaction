import os
import cv2
import json
import torch
import argparse
import subprocess
import csv
import numpy as np
import torchvision.transforms as transforms
import torchaudio
from SoccerNet.utils import getListGames
from model import X3DFreeKickModel
from model_audio_v2 import WhistleNetV2
from model_av_gate import AVGateNet, extract_gate_features

FPS = 2
WINDOW_SECONDS = 15
WINDOW_FRAMES = int(WINDOW_SECONDS * FPS) # 30 frames
STRIDE_SECONDS = 5
STRIDE_FRAMES = int(STRIDE_SECONDS * FPS)  # 10 frames
TOLERANCE_SECONDS = 10.0 # 20s center-point matching window (|t_pred - t_gt| <= 10.0s)
NMS_WINDOW = 15.0

# X3D-M square resolution (224x224)
vid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

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

def process_half(video_path, v_foul, v_sp, a_whistle, device, args):
    full_audio_wav = f"temp_hybrid_audio_{os.getpid()}_{np.random.randint(10000)}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-q:a", "0", "-map", "a", full_audio_wav], capture_output=True)
    
    try:
        full_waveform, sr = torchaudio.load(full_audio_wav)
        if full_waveform.shape[0] > 1:
            full_waveform = torch.mean(full_waveform, dim=0, keepdim=True)
        if sr != 16000:
            full_waveform = torchaudio.transforms.Resample(sr, 16000)(full_waveform)
    except Exception:
        full_waveform = torch.zeros((1, 150 * 60 * 16000))
    finally:
        if os.path.exists(full_audio_wav):
            try:
                os.remove(full_audio_wav)
            except Exception:
                pass
                
    win_sec = getattr(args, 'window_seconds', WINDOW_SECONDS)
    stride_sec = getattr(args, 'stride_seconds', STRIDE_SECONDS)
    win_frames = int(win_sec * FPS)
    stride_frames = int(stride_sec * FPS)

    def get_audio_window(t_start, t_end):
        target_len = int(win_sec * 16000)
        s_end = int(t_end * 16000)
        s_start = max(0, s_end - target_len)
        chunk = full_waveform[:, s_start:s_end]
        if chunk.shape[1] < target_len:
            pad = target_len - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (pad, 0))
        elif chunk.shape[1] > target_len:
            chunk = chunk[:, :target_len]
            
        spec = mel_transform(chunk)
        spec = amp_to_db(spec)
        return spec.unsqueeze(0) # (1, 1, 128, 469)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
        
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25.0
        
    frame_interval = int(round(original_fps / FPS))
    frame_idx = 0
    rolling_buffer = []
    windows_log = []
    pending_windows = []
    batch_size = 8

    def run_batch(batch):
        vid_batch = torch.cat([b['vid_tensor'] for b in batch], dim=0).to(device)
        audio_batch = torch.cat([b['audio_tensor'] for b in batch], dim=0).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                raw_vf = torch.softmax(v_foul(vid_batch), dim=1)[:, 1].cpu().numpy()
                raw_vsp = torch.softmax(v_sp(vid_batch), dim=1)[:, 1].cpu().numpy()
                raw_a = torch.softmax(a_whistle(audio_batch), dim=1)[:, 1].cpu().numpy()
                
        for i, item in enumerate(batch):
            windows_log.append({
                "window_start": item['t_start'],
                "window_end": item['t_end'],
                "raw_foul_prob": float(raw_vf[i]),
                "raw_sp_prob": float(raw_vsp[i]),
                "audio_prob": float(raw_a[i])
            })

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_sec = frame_idx / original_fps
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rolling_buffer.append((vid_transform(frame_rgb), current_sec))
            
            if len(rolling_buffer) >= win_frames:
                input_frames = [x[0] for x in rolling_buffer[:win_frames]]
                t_end = rolling_buffer[win_frames - 1][1]
                t_start = max(0.0, t_end - win_sec)
                
                vid_tensor = torch.stack(input_frames, dim=1).unsqueeze(0) # (1, 3, 30, 224, 224)
                audio_tensor = get_audio_window(t_start, t_end)           # (1, 1, 128, 469)
                
                pending_windows.append({
                    'vid_tensor': vid_tensor,
                    'audio_tensor': audio_tensor,
                    't_start': t_start,
                    't_end': t_end
                })
                
                if len(pending_windows) >= batch_size:
                    run_batch(pending_windows)
                    pending_windows = []
                    if len(windows_log) % 100 == 0:
                        print(f"    ... {len(windows_log)} windows ({t_end/60:.1f} min)", flush=True)
                    
                rolling_buffer = rolling_buffer[stride_frames:]
        frame_idx += 1

    if pending_windows:
        run_batch(pending_windows)
        
    cap.release()
    return windows_log

def evaluate_windows(halves_data, foul_thresh=0.80, sp_thresh=0.85, tol_sec=10.0, nms_window=15.0, av_gate=None, device='cpu', audio_bonus=False):
    tp_f, fp_f, fn_f = 0, 0, 0
    tp_sp, fp_sp, fn_sp = 0, 0, 0
    
    for d in halves_data:
        windows = d['windows']
        if not windows:
            continue
            
        if av_gate is not None:
            feats = extract_gate_features(windows).to(device)
            raw_vf = torch.tensor([w['raw_foul_prob'] for w in windows], dtype=torch.float32).to(device)
            raw_vsp = torch.tensor([w['raw_sp_prob'] for w in windows], dtype=torch.float32).to(device)
            with torch.no_grad():
                p_f, p_sp = av_gate(feats, raw_vf, raw_vsp)
                p_f = p_f.cpu().numpy()
                p_sp = p_sp.cpu().numpy()
        else:
            p_f = np.array([w['raw_foul_prob'] for w in windows])
            p_sp = np.array([w['raw_sp_prob'] for w in windows])
            if audio_bonus:
                for i, w in enumerate(windows):
                    if w['audio_prob'] > 0.70:
                        p_f[i] = min(1.0, p_f[i] + 0.05)
                        p_sp[i] = min(1.0, p_sp[i] + 0.05)
                        
        dets_f = []
        dets_sp = []
        for i, w in enumerate(windows):
            if p_f[i] > foul_thresh:
                dets_f.append((w['window_start'], w['window_end'], float(p_f[i]), 'Foul'))
            if p_sp[i] > sp_thresh:
                dets_sp.append((w['window_start'], w['window_end'], float(p_sp[i]), 'SetPiece'))
                
        kept_f = temporal_nms(dets_f, nms_window)
        matched_gt_f = set()
        for p_start, p_end, prob, _ in kept_f:
            p_center = (p_start + p_end) / 2.0
            hit = [g for g in d['gt_f'] if abs(p_center - g) <= tol_sec]
            if hit:
                tp_f += 1
                for g in hit:
                    matched_gt_f.add(g)
            else:
                fp_f += 1
        fn_f += len(d['gt_f']) - len(matched_gt_f)
        
        kept_sp = temporal_nms(dets_sp, nms_window)
        matched_gt_sp = set()
        for p_start, p_end, prob, _ in kept_sp:
            p_center = (p_start + p_end) / 2.0
            hit = [g for g in d['gt_sp'] if abs(p_center - g) <= tol_sec]
            if hit:
                tp_sp += 1
                for g in hit:
                    matched_gt_sp.add(g)
            else:
                fp_sp += 1
        fn_sp += len(d['gt_sp']) - len(matched_gt_sp)
        
    prec_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0
    rec_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
    f1_f = 2 * prec_f * rec_f / (prec_f + rec_f) if (prec_f + rec_f) > 0 else 0.0
    
    prec_sp = tp_sp / (tp_sp + fp_sp) if (tp_sp + fp_sp) > 0 else 0.0
    rec_sp = tp_sp / (tp_sp + fn_sp) if (tp_sp + fn_sp) > 0 else 0.0
    f1_sp = 2 * prec_sp * rec_sp / (prec_sp + rec_sp) if (prec_sp + rec_sp) > 0 else 0.0
    
    return {
        'foul': {'tp': tp_f, 'fp': fp_f, 'fn': fn_f, 'prec': prec_f, 'rec': rec_f, 'f1': f1_f},
        'sp': {'tp': tp_sp, 'fp': fp_sp, 'fn': fn_sp, 'prec': prec_sp, 'rec': rec_sp, 'f1': f1_sp},
        'comb_f1': (f1_f + f1_sp) / 2.0
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid X3D-M + WhistleNetV2 on SoccerNet")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-games", type=int, default=10)
    parser.add_argument("--foul-ckpt", type=str, default="checkpoints/x3d_foul_best.pth")
    parser.add_argument("--sp-ckpt", type=str, default="checkpoints/x3d_setpiece_best.pth")
    parser.add_argument("--audio-ckpt", type=str, default="checkpoints/audio_whistle_v2_best.pth")
    parser.add_argument("--gate-ckpt", type=str, default=None)
    parser.add_argument("--log-file", type=str, default="soccernet_eval_hybrid_test_log.csv")
    parser.add_argument("--report-file", type=str, default="soccernet_eval_hybrid_test_report.txt")
    parser.add_argument("--tolerance-seconds", type=float, default=10.0)
    parser.add_argument("--reuse-log", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(" SoccerNet Hybrid Evaluation Pipeline: X3D-M (Full Negatives) + WhistleNetV2 (ResNet-34)")
    print(f" Device: {device} | Split: {args.split} | Tolerance: +/- {args.tolerance_seconds}s (20s center-point window)")
    print("=" * 80)
    
    soccernet_dir = "/home/pilot/Desktop/ballsaction/soccernet_data"
    all_games = getListGames(args.split)
    valid_games = [
        g for g in all_games
        if os.path.exists(os.path.join(soccernet_dir, g, "1_224p.mkv"))
        and os.path.exists(os.path.join(soccernet_dir, g, "Labels-v2.json"))
    ]
    if args.num_games > 0:
        valid_games = valid_games[:args.num_games]
        
    print(f"Target Games ({len(valid_games)}):")
    for g in valid_games:
        print(f"  {g}")
        
    halves_data = []
    
    if args.reuse_log and os.path.exists(args.log_file):
        print(f"\n[INFO] Reusing existing window log: {args.log_file}")
        matches = {}
        with open(args.log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['match'], int(row['half']))
                if key not in matches:
                    matches[key] = []
                matches[key].append({
                    'window_start': float(row['window_start']),
                    'window_end': float(row['window_end']),
                    'raw_foul_prob': float(row['raw_foul_prob']),
                    'raw_sp_prob': float(row['raw_sp_prob']),
                    'audio_prob': float(row['audio_prob'])
                })
        for (game, half), windows in matches.items():
            label_path = os.path.join(soccernet_dir, game, "Labels-v2.json")
            gt_fouls, gt_sps = parse_labels_v2(label_path)
            halves_data.append({
                'game': game,
                'half': half,
                'windows': windows,
                'gt_f': [sec for h, sec in gt_fouls if h == half],
                'gt_sp': [sec for h, sec in gt_sps if h == half]
            })
    else:
        print("\nLoading models for inference...")
        v_foul = X3DFreeKickModel(num_classes=2, pretrained=False).to(device)
        foul_dict = torch.load(args.foul_ckpt, map_location=device)
        v_foul.load_state_dict(foul_dict.get('model_state_dict', foul_dict))
        v_foul.eval()
        print(f"Loaded X3D-M Foul model: {args.foul_ckpt}")
        
        v_sp = X3DFreeKickModel(num_classes=2, pretrained=False).to(device)
        sp_dict = torch.load(args.sp_ckpt, map_location=device)
        v_sp.load_state_dict(sp_dict.get('model_state_dict', sp_dict))
        v_sp.eval()
        print(f"Loaded X3D-M Set-Piece model: {args.sp_ckpt}")
        
        a_whistle = WhistleNetV2(pretrained=False).to(device)
        audio_dict = torch.load(args.audio_ckpt, map_location=device)
        a_whistle.load_state_dict(audio_dict.get('model_state_dict', audio_dict))
        a_whistle.eval()
        print(f"Loaded WhistleNetV2: {args.audio_ckpt}")
        
        with open(args.log_file, 'w', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['match', 'half', 'window_start', 'window_end', 'raw_foul_prob', 'raw_sp_prob', 'audio_prob'])
            writer.writeheader()
            
            for g_idx, game in enumerate(valid_games):
                print(f"\n[{g_idx+1}/{len(valid_games)}] Processing: {game}")
                game_dir = os.path.join(soccernet_dir, game)
                label_path = os.path.join(game_dir, "Labels-v2.json")
                gt_fouls, gt_sps = parse_labels_v2(label_path)
                
                for half in [1, 2]:
                    vid_path = os.path.join(game_dir, f"{half}_224p.mkv")
                    if not os.path.exists(vid_path):
                        continue
                    print(f"  Half {half}...")
                    win_log = process_half(vid_path, v_foul, v_sp, a_whistle, device, args)
                    for r in win_log:
                        writer.writerow({
                            'match': game,
                            'half': half,
                            'window_start': r['window_start'],
                            'window_end': r['window_end'],
                            'raw_foul_prob': r['raw_foul_prob'],
                            'raw_sp_prob': r['raw_sp_prob'],
                            'audio_prob': r['audio_prob']
                        })
                    f_csv.flush()
                    
                    halves_data.append({
                        'game': game,
                        'half': half,
                        'windows': win_log,
                        'gt_f': [sec for h, sec in gt_fouls if h == half],
                        'gt_sp': [sec for h, sec in gt_sps if h == half]
                    })
        print(f"\nWindow extraction completed! Log saved to {args.log_file}")
        
    av_gate = None
    if args.gate_ckpt and os.path.exists(args.gate_ckpt):
        print(f"\nLoading trained AVGateNet from {args.gate_ckpt}...")
        av_gate = AVGateNet(in_features=15, hidden_dim=32, dropout=0.0).to(device)
        gate_dict = torch.load(args.gate_ckpt, map_location=device)
        av_gate.load_state_dict(gate_dict.get('model_state_dict', gate_dict))
        av_gate.eval()
        
    print("\n" + "=" * 80)
    print(" EVALUATION RESULTS (20-Second Center-Point Window, 5s Slider)")
    print("=" * 80)
    
    thresholds = [0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90] if args.sweep else [0.75, 0.80, 0.82, 0.85]
    
    with open(args.report_file, 'w') as f_rep:
        f_rep.write(f"SoccerNet Hybrid Evaluation Report ({args.split} split, {len(valid_games)} matches)\n")
        f_rep.write(f"Matching window: 20-second center-point (+/- 10.0s), Slider: 5.0s\n")
        f_rep.write("=" * 80 + "\n\n")
        
        # 1. Raw Video Only
        f_rep.write("### 1. RAW X3D-M VIDEO ONLY (Full Negatives)\n")
        print("\n--- 1. RAW X3D-M VIDEO ONLY ---")
        for th in thresholds:
            res = evaluate_windows(halves_data, foul_thresh=th, sp_thresh=th, tol_sec=args.tolerance_seconds, av_gate=None, device=device, audio_bonus=False)
            f_line = (f"Thresh {th:.2f} | Foul: P={res['foul']['prec']*100:.1f}%, R={res['foul']['rec']*100:.1f}%, F1={res['foul']['f1']*100:.1f}% (TP={res['foul']['tp']}, FP={res['foul']['fp']}, FN={res['foul']['fn']}) | "
                      f"SP: P={res['sp']['prec']*100:.1f}%, R={res['sp']['rec']*100:.1f}%, F1={res['sp']['f1']*100:.1f}% (TP={res['sp']['tp']}, FP={res['sp']['fp']}, FN={res['sp']['fn']}) | "
                      f"Comb F1: {res['comb_f1']*100:.1f}%")
            print(f_line)
            f_rep.write(f_line + "\n")
            
        # 2. Video + WhistleNetV2 Soft Audio Bonus
        f_rep.write("\n### 2. X3D-M VIDEO + WHISTLENET V2 SOFT AUDIO BONUS (+0.05 if whistle > 0.70)\n")
        print("\n--- 2. X3D-M VIDEO + WHISTLENET V2 SOFT AUDIO BONUS ---")
        for th in thresholds:
            res = evaluate_windows(halves_data, foul_thresh=th, sp_thresh=th, tol_sec=args.tolerance_seconds, av_gate=None, device=device, audio_bonus=True)
            f_line = (f"Thresh {th:.2f} | Foul: P={res['foul']['prec']*100:.1f}%, R={res['foul']['rec']*100:.1f}%, F1={res['foul']['f1']*100:.1f}% (TP={res['foul']['tp']}, FP={res['foul']['fp']}, FN={res['foul']['fn']}) | "
                      f"SP: P={res['sp']['prec']*100:.1f}%, R={res['sp']['rec']*100:.1f}%, F1={res['sp']['f1']*100:.1f}% (TP={res['sp']['tp']}, FP={res['sp']['fp']}, FN={res['sp']['fn']}) | "
                      f"Comb F1: {res['comb_f1']*100:.1f}%")
            print(f_line)
            f_rep.write(f_line + "\n")
            
        # 3. Full Decoupled AVGateNet
        if av_gate is not None:
            f_rep.write("\n### 3. DECOUPLED AV-GATENET (Hybrid: X3D-M + WhistleNetV2)\n")
            print("\n--- 3. DECOUPLED AV-GATENET ---")
            for th in thresholds:
                res = evaluate_windows(halves_data, foul_thresh=th, sp_thresh=th, tol_sec=args.tolerance_seconds, av_gate=av_gate, device=device, audio_bonus=False)
                f_line = (f"Thresh {th:.2f} | Foul: P={res['foul']['prec']*100:.1f}%, R={res['foul']['rec']*100:.1f}%, F1={res['foul']['f1']*100:.1f}% (TP={res['foul']['tp']}, FP={res['foul']['fp']}, FN={res['foul']['fn']}) | "
                          f"SP: P={res['sp']['prec']*100:.1f}%, R={res['sp']['rec']*100:.1f}%, F1={res['sp']['f1']*100:.1f}% (TP={res['sp']['tp']}, FP={res['sp']['fp']}, FN={res['sp']['fn']}) | "
                          f"Comb F1: {res['comb_f1']*100:.1f}%")
                print(f_line)
                f_rep.write(f_line + "\n")
                
    print(f"\nReport written to {args.report_file}")

if __name__ == "__main__":
    main()
