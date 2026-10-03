import os
import cv2
import json
import torch
import argparse
import subprocess
import csv
import torchvision.transforms as transforms
import torchaudio
from SoccerNet.utils import getListGames
from model_fusion import X3DMultiModalModel

FPS = 2
WINDOW_SECONDS = 15
WINDOW_FRAMES = WINDOW_SECONDS * FPS
STRIDE_SECONDS = 5
STRIDE_FRAMES = STRIDE_SECONDS * FPS
TOLERANCE_SECONDS = 10.0

# Native 16:9 widescreen: (224, 398)
vid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 398)),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

def temporal_nms(detections, nms_window=15.0):
    """Keep highest confidence detection and suppress all within nms_window seconds."""
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

def process_half(video_path, v_foul, v_sp, device, args):
    full_audio_wav = f"eval_mm_temp_{os.getpid()}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", full_audio_wav
    ], capture_output=True)
    
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
            try: os.remove(full_audio_wav)
            except: pass

    win_sec = getattr(args, 'window_seconds', WINDOW_SECONDS)
    stride_sec = getattr(args, 'stride_seconds', STRIDE_SECONDS)
    win_frames = int(win_sec * FPS)
    stride_frames = int(stride_sec * FPS)

    def get_audio_window(t_start, t_end):
        s_start = int(t_start * 16000)
        s_end = int(t_end * 16000)
        target_len = int((t_end - t_start) * 16000)
        
        if s_start < 0:
            chunk = full_waveform[:, 0:max(0, s_end)]
            pad = target_len - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (pad, 0))
        else:
            chunk = full_waveform[:, s_start:s_end]
            if chunk.shape[1] < target_len:
                pad = target_len - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad))
            else:
                chunk = chunk[:, :target_len]
                
        spec = amp_to_db(mel_transform(chunk)) # (1, 128, ~469)
        if spec.shape[2] < 469:
            spec = torch.nn.functional.pad(spec, (0, 469 - spec.shape[2]))
        else:
            spec = spec[:, :, :469]
        return spec.unsqueeze(0) # (1, 1, 128, 469)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], []

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25.0

    frame_interval = int(round(original_fps / FPS))
    frame_idx = 0
    rolling_buffer = []

    detected_fouls = []
    detected_sp = []
    windows_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = frame_idx / original_fps
        if frame_idx % frame_interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rolling_buffer.append((vid_transform(img), current_sec))

            if len(rolling_buffer) >= win_frames:
                input_frames = [x[0] for x in rolling_buffer[:win_frames]]
                t_end = rolling_buffer[win_frames - 1][1]
                t_start = max(0.0, t_end - win_sec)

                vid_tensor = torch.stack(input_frames, dim=1).unsqueeze(0).to(device) # (1, 3, 30, 224, 398)
                audio_tensor = get_audio_window(t_start, t_end).to(device)           # (1, 1, 128, 469)

                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        foul_out, foul_weights = v_foul(vid_tensor, audio_tensor, return_head_weights=True)
                        foul_prob = torch.softmax(foul_out, dim=1)[0, 1].item()
                        
                        sp_out, sp_weights = v_sp(vid_tensor, audio_tensor, return_head_weights=True)
                        sp_prob = torch.softmax(sp_out, dim=1)[0, 1].item()

                windows_log.append({
                    "window_start": t_start,
                    "window_end": t_end,
                    "foul_prob": foul_prob,
                    "sp_prob": sp_prob,
                    "foul_head_weights": foul_weights[0].cpu().numpy().tolist(),
                    "sp_head_weights": sp_weights[0].cpu().numpy().tolist()
                })

                if foul_prob > args.foul_threshold:
                    detected_fouls.append((t_start, t_end, foul_prob, "Foul"))
                if sp_prob > args.sp_threshold:
                    detected_sp.append((t_start, t_end, sp_prob, "SetPiece"))

                rolling_buffer = rolling_buffer[stride_frames:]
        frame_idx += 1

    cap.release()

    final_fouls = temporal_nms(detected_fouls, nms_window=win_sec)
    final_sp = temporal_nms(detected_sp, nms_window=win_sec)

    return final_fouls, final_sp, windows_log

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

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    v_foul = X3DMultiModalModel(num_classes=2, num_heads=4, pretrained=False).to(device)
    foul_ckpt = torch.load(args.foul_checkpoint, map_location=device)
    v_foul.load_state_dict(foul_ckpt['model_state_dict'] if 'model_state_dict' in foul_ckpt else foul_ckpt)
    v_foul.eval()
    print(f"Loaded Foul Multimodal Model: {args.foul_checkpoint}")

    v_sp = X3DMultiModalModel(num_classes=2, num_heads=4, pretrained=False).to(device)
    sp_ckpt = torch.load(args.sp_checkpoint, map_location=device)
    v_sp.load_state_dict(sp_ckpt['model_state_dict'] if 'model_state_dict' in sp_ckpt else sp_ckpt)
    v_sp.eval()
    print(f"Loaded Set-Piece Multimodal Model: {args.sp_checkpoint}")

    if args.save_clips:
        os.makedirs(args.output_dir, exist_ok=True)

    all_games = getListGames(args.split)
    games = all_games[:args.num_games] if args.num_games > 0 else all_games

    soccernet_dir = "soccernet_data"
    valid_games = [
        g for g in games
        if os.path.exists(os.path.join(soccernet_dir, g, "1_224p.mkv"))
        and os.path.exists(os.path.join(soccernet_dir, g, "Labels-v2.json"))
    ]
    print(f"Evaluating on {len(valid_games)} {args.split} matches (Foul Thresh: {args.foul_threshold}, SetPiece Thresh: {args.sp_threshold})")

    metrics = {
        "Foul": {"TP": 0, "FP": 0, "FN": 0, "total_gt": 0},
        "SetPiece": {"TP": 0, "FP": 0, "FN": 0, "total_gt": 0}
    }

    all_window_logs = []
    
    with open(args.report_file, 'w') as report:
        report.write("=" * 80 + "\n")
        report.write(" SOCCERNET MULTIMODAL EVALUATION REPORT (16:9 + SYNCHRONIZED AUDIO)\n")
        report.write("=" * 80 + "\n\n")

        for g_idx, game in enumerate(valid_games):
            print(f"[{g_idx+1}/{len(valid_games)}] Evaluating: {game}")
            report.write(f"\n--- Match: {game} ---\n")

            game_dir = os.path.join(soccernet_dir, game)
            label_path = os.path.join(game_dir, "Labels-v2.json")
            gt_fouls, gt_sps = parse_labels_v2(label_path)

            for half in [1, 2]:
                video_path = os.path.join(game_dir, f"{half}_224p.mkv")
                if not os.path.exists(video_path):
                    continue

                half_gt_fouls = [g[1] for g in gt_fouls if g[0] == half]
                half_gt_sps = [g[1] for g in gt_sps if g[0] == half]
                metrics["Foul"]["total_gt"] += len(half_gt_fouls)
                metrics["SetPiece"]["total_gt"] += len(half_gt_sps)

                pred_fouls, pred_sps, windows_log = process_half(video_path, v_foul, v_sp, device, args)

                for w in windows_log:
                    w['match'] = game
                    w['half'] = half
                    all_window_logs.append(w)

                def match_preds(preds, gts, label_type):
                    matched_gts = set()
                    tol = args.tolerance_seconds
                    for p in preds:
                        p_start, p_end, p_prob, p_label = p
                        p_center = (p_start + p_end) / 2.0
                        candidates = [g for g in gts if abs(p_center - g) <= tol and g not in matched_gts]
                        if candidates:
                            best_g = min(candidates, key=lambda g: abs(p_center - g))
                            metrics[label_type]["TP"] += 1
                            matched_gts.add(best_g)
                            report.write(f"  TP {label_type} | Prob: {p_prob:.3f} | Window: [{p_start:.1f}s, {p_end:.1f}s] | Matched GT @ {best_g}s\n")
                        else:
                            metrics[label_type]["FP"] += 1
                            report.write(f"  FP {label_type} | Prob: {p_prob:.3f} | Window: [{p_start:.1f}s, {p_end:.1f}s]\n")

                    fn_count = len(gts) - len(matched_gts)
                    metrics[label_type]["FN"] += fn_count
                    if fn_count > 0:
                        report.write(f"  {fn_count} {label_type} Missed GTs (FN)\n")

                match_preds(pred_fouls, half_gt_fouls, "Foul")
                match_preds(pred_sps, half_gt_sps, "SetPiece")

        report.write("\n" + "=" * 80 + "\n")
        report.write(" SUMMARY PERFORMANCE\n")
        report.write("=" * 80 + "\n")
        print("\n" + "=" * 60)
        print(" SOCCERNET MULTIMODAL EVALUATION RESULTS")
        print("=" * 60)

        for label_type in ["Foul", "SetPiece"]:
            tp = metrics[label_type]["TP"]
            fp = metrics[label_type]["FP"]
            fn = metrics[label_type]["FN"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            res_str = f"{label_type:10s} | Precision: {prec:.4f} ({tp}/{tp+fp}) | Recall: {rec:.4f} ({tp}/{tp+fn}) | F1: {f1:.4f} | Total GT: {metrics[label_type]['total_gt']}"
            print(res_str)
            report.write(res_str + "\n")

    # Save window log CSV
    with open(args.log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['match', 'half', 'window_start', 'window_end', 'foul_prob', 'sp_prob'])
        for w in all_window_logs:
            writer.writerow([w['match'], w['half'], w['window_start'], w['window_end'], f"{w['foul_prob']:.4f}", f"{w['sp_prob']:.4f}"])

    print(f"\nEvaluation Report written to {args.report_file}")
    print(f"Window logs written to {args.log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-games", type=int, default=10)
    parser.add_argument("--foul-checkpoint", type=str, default="checkpoints/x3d_foul_multimodal_best.pth")
    parser.add_argument("--sp-checkpoint", type=str, default="checkpoints/x3d_setpiece_multimodal_best.pth")
    parser.add_argument("--foul-threshold", type=float, default=0.70)
    parser.add_argument("--sp-threshold", type=float, default=0.75)
    parser.add_argument("--tolerance-seconds", type=float, default=10.0)
    parser.add_argument("--window-seconds", type=float, default=15.0)
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, default="soccernet_eval_multimodal_highlights")
    parser.add_argument("--report-file", type=str, default="soccernet_eval_multimodal_report.txt")
    parser.add_argument("--log-file", type=str, default="soccernet_eval_multimodal_log.csv")
    parser.add_argument("--save-clips", action="store_true")
    args = parser.parse_args()
    
    evaluate(args)
