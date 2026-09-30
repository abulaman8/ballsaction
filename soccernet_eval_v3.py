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
from model import X3DFreeKickModel
from model_audio import WhistleNet

FPS = 2
WINDOW_SECONDS = 10
WINDOW_FRAMES = WINDOW_SECONDS * FPS
STRIDE_SECONDS = 5
STRIDE_FRAMES = STRIDE_SECONDS * FPS
TOLERANCE_SECONDS = 15.0

vid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

def temporal_nms(detections, nms_window=15.0):
    """Keep highest confidence detection and suppress all within nms_window seconds."""
    if not detections: return []
    # Sort by probability descending
    detections.sort(key=lambda x: x[2], reverse=True)
    kept = []
    suppressed = set()
    for i, det in enumerate(detections):
        if i in suppressed:
            continue
        kept.append(det)
        t_center = (det[0] + det[1]) / 2
        for j in range(i+1, len(detections)):
            if j in suppressed:
                continue
            other_center = (detections[j][0] + detections[j][1]) / 2
            if abs(t_center - other_center) < nms_window:
                suppressed.add(j)
    kept.sort(key=lambda x: x[0])  # Re-sort by time
    return kept

def process_half(video_path, v_foul, v_sp, a_whistle, device, args):
    full_audio_wav = "eval_v3_full_audio.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-q:a", "0", "-map", "a", full_audio_wav], capture_output=True)
    
    try:
        full_waveform, sr = torchaudio.load(full_audio_wav)
        if full_waveform.shape[0] > 1: full_waveform = torch.mean(full_waveform, dim=0, keepdim=True)
        if sr != 16000: full_waveform = torchaudio.transforms.Resample(sr, 16000)(full_waveform)
    except Exception:
        full_waveform = torch.zeros((1, 150 * 60 * 16000))
        
    def get_audio_window(t_start, t_end):
        s_start = int(t_start * 16000)
        s_end = int(t_end * 16000)
        chunk = full_waveform[:, s_start:s_end]
        target_len = int((t_end - t_start) * 16000)
        
        if chunk.shape[1] < target_len:
            pad = target_len - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
            
        spec = mel_transform(chunk)
        spec = amp_to_db(spec)
        if spec.shape[2] < 313: spec = torch.nn.functional.pad(spec, (0, 313 - spec.shape[2]))
        else: spec = spec[:, :, :313]
        return spec.unsqueeze(0)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return [], [], []
        
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0: original_fps = 25.0
        
    frame_interval = int(round(original_fps / FPS))
    frame_idx = 0
    rolling_buffer = []
    
    detected_fouls = []
    detected_sp = []
    windows_log = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        current_sec = frame_idx / original_fps
        if frame_idx % frame_interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rolling_buffer.append((vid_transform(img), current_sec))
            
            if len(rolling_buffer) >= WINDOW_FRAMES:
                input_frames = [x[0] for x in rolling_buffer[:WINDOW_FRAMES]]
                t_end = rolling_buffer[WINDOW_FRAMES-1][1]
                t_start = max(0, t_end - WINDOW_SECONDS)
                
                vid_tensor = torch.stack(input_frames, dim=1).unsqueeze(0).to(device)
                audio_tensor = get_audio_window(t_start, t_end).to(device)
                
                with torch.no_grad():
                    foul_prob = torch.softmax(v_foul(vid_tensor), dim=1)[0, 1].item()
                    sp_prob = torch.softmax(v_sp(vid_tensor), dim=1)[0, 1].item()
                    audio_prob = torch.softmax(a_whistle(audio_tensor), dim=1)[0, 1].item()
                    
                    if audio_prob > 0.7:
                        foul_prob = min(1.0, foul_prob + 0.05)
                        sp_prob = min(1.0, sp_prob + 0.05)
                
                windows_log.append({
                    "window_start": t_start,
                    "window_end": t_end,
                    "foul_prob": foul_prob,
                    "sp_prob": sp_prob,
                    "audio_prob": audio_prob
                })
                
                if foul_prob > args.foul_threshold:
                    detected_fouls.append((t_start, t_end, foul_prob, "Foul"))
                if sp_prob > args.sp_threshold:
                    detected_sp.append((t_start, t_end, sp_prob, "SetPiece"))
                    
                rolling_buffer = rolling_buffer[STRIDE_FRAMES:]
        frame_idx += 1
    
    cap.release()
    
    final_fouls = temporal_nms(detected_fouls)
    final_sp = temporal_nms(detected_sp)
    
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
        time_str = ann.get('gameTime', '') # "1 - 25:30"
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

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    v_foul = X3DFreeKickModel(num_classes=2, pretrained=False).to(device)
    v_foul.load_state_dict(torch.load("checkpoints/x3d_foul_best.pth", map_location=device))
    v_foul.eval()
    
    v_sp = X3DFreeKickModel(num_classes=2, pretrained=False).to(device)
    v_sp.load_state_dict(torch.load("checkpoints/x3d_setpiece_best.pth", map_location=device))
    v_sp.eval()
    
    a_whistle = WhistleNet().to(device)
    a_whistle.load_state_dict(torch.load("checkpoints/audio_whistle_best.pth", map_location=device))
    a_whistle.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    games = getListGames(args.split)[:args.num_games]
    soccernet_dir = "/home/pilot/Desktop/ballsaction/soccernet_data"
    
    metrics = {
        "Foul": {"TP": 0, "FP": 0, "FN": 0},
        "SetPiece": {"TP": 0, "FP": 0, "FN": 0}
    }
    
    review_data = []
    
    with open('soccernet_eval_v3_log.csv', 'w', newline='') as csvfile:
        fieldnames = ['match', 'half', 'window_start', 'window_end', 'foul_prob', 'sp_prob', 'audio_prob']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        with open('soccernet_eval_v3_report.txt', 'w') as report:
            for game in games:
                print(f"Evaluating game: {game}")
                report.write(f"--- Game: {game} ---\n")
                
                game_dir = os.path.join(soccernet_dir, game)
                label_path = os.path.join(game_dir, "Labels-v2.json")
                gt_fouls, gt_sps = parse_labels_v2(label_path)
                
                for half in [1, 2]:
                    video_path = os.path.join(game_dir, f"{half}_224p.mkv")
                    if not os.path.exists(video_path):
                        continue
                        
                    half_gt_fouls = [g[1] for g in gt_fouls if g[0] == half]
                    half_gt_sps = [g[1] for g in gt_sps if g[0] == half]
                    
                    pred_fouls, pred_sps, windows_log = process_half(video_path, v_foul, v_sp, a_whistle, device, args)
                    
                    for row in windows_log:
                        row['match'] = game
                        row['half'] = half
                        writer.writerow(row)
                        
                    def match_predictions(preds, gts, label_type):
                        matched_gts = set()
                        for p in preds:
                            p_start, p_end, p_prob, p_label = p
                            hit = [g for g in gts if p_start - TOLERANCE_SECONDS <= g <= p_end + TOLERANCE_SECONDS]
                            
                            clip_name = f"{game.replace('/','_')}_H{half}_{label_type}_{p_start:.0f}s.mp4"
                            clip_path = os.path.join(args.output_dir, clip_name)
                            
                            is_tp = False
                            if hit:
                                is_tp = True
                                metrics[label_type]["TP"] += 1
                                for g in hit: matched_gts.add(g)
                                report.write(f"  TP {label_type} | Conf: {p_prob:.2f} | Range: [{p_start:.1f}s, {p_end:.1f}s]\n")
                            else:
                                metrics[label_type]["FP"] += 1
                                report.write(f"  FP {label_type} | Conf: {p_prob:.2f} | Range: [{p_start:.1f}s, {p_end:.1f}s]\n")
                                
                            duration = p_end - p_start
                            clip_cmd = [
                                "ffmpeg", "-y", "-loglevel", "error",
                                "-ss", str(p_start), "-i", video_path, "-t", str(duration),
                                "-map", "0:v:0?", "-map", "0:a:0?",
                                "-vf", "scale=-2:480", "-c:v", "libx264", "-crf", "28", "-preset", "fast",
                                "-c:a", "aac", "-b:a", "128k",
                                clip_path
                            ]
                            subprocess.run(clip_cmd)
                            
                            review_data.append({
                                "id": clip_name,
                                "path": clip_name,
                                "type": "TP" if is_tp else "FP",
                                "label": label_type,
                                "pred_prob": p_prob,
                                "clip_start": p_start,
                                "clip_end": p_end
                            })
                            
                        for g in gts:
                            if g not in matched_gts:
                                metrics[label_type]["FN"] += 1
                                report.write(f"  FN {label_type} | Time: {g}s\n")
                                
                    match_predictions(pred_fouls, half_gt_fouls, "Foul")
                    match_predictions(pred_sps, half_gt_sps, "SetPiece")

            report.write("\n=== OVERALL SUMMARY ===\n")
            for cls in ["Foul", "SetPiece"]:
                tp = metrics[cls]["TP"]
                fp = metrics[cls]["FP"]
                fn = metrics[cls]["FN"]
                prec = tp / (tp + fp) if tp + fp > 0 else 0
                rec = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
                
                report.write(f"{cls}:\n")
                report.write(f"  TP: {tp}, FP: {fp}, FN: {fn}\n")
                report.write(f"  Precision: {prec:.4f}\n")
                report.write(f"  Recall:    {rec:.4f}\n")
                report.write(f"  F1 Score:  {f1:.4f}\n")
                
    with open(os.path.join(args.output_dir, "review_metadata.json"), "w") as f:
        json.dump(review_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-games", type=int, default=4)
    parser.add_argument("--foul-threshold", type=float, default=0.75)
    parser.add_argument("--sp-threshold", type=float, default=0.80)
    parser.add_argument("--output-dir", type=str, default="soccernet_eval_v3_highlights")
    args = parser.parse_args()
    
    evaluate(args)
