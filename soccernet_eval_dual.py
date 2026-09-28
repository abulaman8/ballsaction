import os
import cv2
import json
import torch
import torch.nn as nn
import subprocess
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchaudio
from model import X3DFreeKickModel
from train_dual_fusion import DualFusionModel
from SoccerNet.utils import getListGames
import shutil

FPS = 4
WINDOW_SECONDS = 10
WINDOW_FRAMES = WINDOW_SECONDS * FPS
STRIDE_SECONDS = 5
STRIDE_FRAMES = STRIDE_SECONDS * FPS

FOUL_THRESHOLD = 0.5
SETPIECE_THRESHOLD = 0.6
TOLERANCE_SECONDS = 10.0
BOOST_DURATION_SEC = 90.0

vid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

def merge_intervals(intervals, max_gap=10.0):
    if not intervals: return []
    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    for curr in intervals[1:]:
        prev = merged[-1]
        # curr[0] is start, prev[1] is end
        if curr[0] <= prev[1] + max_gap:
            prev[1] = max(prev[1], curr[1]) # End time
            prev[2] = max(prev[2], curr[2]) # Prob
            # Or keeps boosted status if applicable
        else:
            merged.append(list(curr))
    return merged

def process_video(video_path, foul_model, sp_model, device, out_dir, vid_name):
    print(f"Processing {os.path.basename(video_path)}...")
    
    # Extract full audio once for extreme speed
    full_audio_wav = "eval_full_audio.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-q:a", "0", "-map", "a", full_audio_wav], capture_output=True)
    
    try:
        full_waveform, sr = torchaudio.load(full_audio_wav)
        if full_waveform.shape[0] > 1: full_waveform = torch.mean(full_waveform, dim=0, keepdim=True)
        if sr != 16000: full_waveform = torchaudio.transforms.Resample(sr, 16000)(full_waveform)
    except Exception:
        full_waveform = torch.zeros((1, 90 * 60 * 16000)) # dummy silence
        
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
        if spec.shape[2] < 256: spec = torch.nn.functional.pad(spec, (0, 256 - spec.shape[2]))
        else: spec = spec[:, :, :256]
        return spec.unsqueeze(0)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
        
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0: original_fps = 25.0
        
    frame_interval = int(round(original_fps / FPS))
    frame_idx = 0
    rolling_buffer = []
    
    detected_fouls = []
    detected_sps = []
    last_foul_time = -999.0
    
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
                t_center = t_start + 5.0
                
                vid_tensor = torch.stack(input_frames, dim=1).unsqueeze(0).to(device)
                
                # Foul Audio [t_center, t_end]
                foul_audio = get_audio_window(t_center, t_end).to(device)
                # SP Audio [t_start, t_center]
                sp_audio = get_audio_window(t_start, t_center).to(device)
                
                with torch.no_grad():
                    foul_logits = foul_model(vid_tensor, foul_audio)
                    sp_logits = sp_model(vid_tensor, sp_audio)
                    foul_prob = torch.softmax(foul_logits, dim=1)[0, 1].item()
                    sp_prob = torch.softmax(sp_logits, dim=1)[0, 1].item()
                
                is_boosted = False
                if (t_center - last_foul_time) <= BOOST_DURATION_SEC:
                    sp_prob += 0.20 # Soft boost
                    is_boosted = True
                    
                if foul_prob > FOUL_THRESHOLD:
                    last_foul_time = t_center
                    # 20 second default by adding +- 5 seconds to the 10s window
                    clip_start = max(0, t_start - 5.0)
                    clip_end = t_end + 5.0
                    detected_fouls.append((clip_start, clip_end, foul_prob, "Foul", False))
                if sp_prob > SETPIECE_THRESHOLD:
                    clip_start = max(0, t_start - 5.0)
                    clip_end = t_end + 5.0
                    detected_sps.append((clip_start, clip_end, sp_prob, "SetPiece", is_boosted))
                    
                rolling_buffer = rolling_buffer[STRIDE_FRAMES:]
        frame_idx += 1
    
    cap.release()
    
    # Merge overlapping intervals (Cluster)
    merged_fouls = merge_intervals(detected_fouls, max_gap=5.0)
    merged_sps = merge_intervals(detected_sps, max_gap=5.0)
    
    # Extract merged clips via FFmpeg
    def extract_clips(clips, prefix):
        for idx, clip in enumerate(clips):
            c_start, c_end, prob, label, boosted = clip
            duration = c_end - c_start
            out_clip = os.path.join(out_dir, f"{vid_name}_{prefix}_clip{idx}_{c_start:.0f}s.mp4")
            clip_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(c_start),
                "-i", video_path,
                "-t", str(duration),
                "-vf", "scale=-2:480",
                "-c:v", "libx264", "-crf", "28", "-preset", "fast",
                "-c:a", "aac", "-b:a", "128k",
                out_clip
            ]
            subprocess.run(clip_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
    extract_clips(merged_fouls, "Foul")
    extract_clips(merged_sps, "SetPiece")
    
    return merged_fouls, merged_sps

def parse_gt(json_path):
    with open(json_path, 'r') as f: data = json.load(f)
    gt_fouls = {1: [], 2: []}
    gt_sps = {1: [], 2: []}
    for ann in data.get('annotations', []):
        half = int(ann['gameTime'].split(' - ')[0])
        mm, ss = map(int, ann['gameTime'].split(' - ')[1].split(':'))
        sec = mm * 60 + ss
        if ann['label'] == 'Foul': gt_fouls[half].append(sec)
        elif ann['label'] in ['Direct free-kick', 'Penalty']: gt_sps[half].append(sec)
    return gt_fouls, gt_sps

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    foul_model = DualFusionModel("checkpoints/x3d_foul_best.pth").to(device)
    foul_model.load_state_dict(torch.load("checkpoints/fusion_foul_best.pth", map_location=device))
    foul_model.eval()
    
    sp_model = DualFusionModel("checkpoints/x3d_setpiece_best.pth").to(device)
    sp_model.load_state_dict(torch.load("checkpoints/fusion_setpiece_best.pth", map_location=device))
    sp_model.eval()
    
    data_dir = "/home/pilot/Desktop/ballsaction/soccernet_data"
    out_dir = "/home/pilot/Desktop/ballsaction/soccernet_eval_dual_highlights"
    report_file = "/home/pilot/Desktop/ballsaction/soccernet_eval_dual_report.txt"
    
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(report_file): os.remove(report_file)
    
    test_games = getListGames("test")
    eval_games = [g for g in test_games[10:] if os.path.exists(os.path.join(data_dir, g, "1_224p.mkv"))][:5]
    
    print(f"Evaluating on 5 games: {eval_games}")
    
    stats = {"Foul": {"tp":0, "fp":0, "fn":0, "gt":0}, "SetPiece": {"tp":0, "fp":0, "fn":0, "gt":0}}
    
    with open(report_file, "a") as f: f.write("SOCCERNET DUAL EVALUATION REPORT\n================================\n\n")
        
    for game in eval_games:
        gt_fouls, gt_sps = parse_gt(os.path.join(data_dir, game, "Labels-v2.json"))
        
        for half in [1, 2]:
            vid_path = os.path.join(data_dir, game, f"{half}_224p.mkv")
            if not os.path.exists(vid_path): continue
            
            d_fouls, d_sps = process_video(vid_path, foul_model, sp_model, device, out_dir, f"{game.replace('/', '_')}_h{half}")
            
            for cls_name, preds, gts in [("Foul", d_fouls, gt_fouls[half]), ("SetPiece", d_sps, gt_sps[half])]:
                stats[cls_name]["gt"] += len(gts)
                matched = set()
                
                # Merge overlapping predictions roughly
                preds = sorted(preds, key=lambda x: x[0])
                
                for p in preds:
                    p_start, p_end, prob, _, boosted = p
                    hit = [g for g in gts if p_start - TOLERANCE_SECONDS <= g <= p_end + TOLERANCE_SECONDS]
                    if hit:
                        stats[cls_name]["tp"] += 1
                        for g in hit: matched.add(g)
                    else:
                        stats[cls_name]["fp"] += 1
                        
                stats[cls_name]["fn"] += len(gts) - len(matched)
                
    # Calculate metrics
    with open(report_file, "a") as f:
        for cls_name in ["Foul", "SetPiece"]:
            tp, fp, fn = stats[cls_name]["tp"], stats[cls_name]["fp"], stats[cls_name]["fn"]
            prec = tp / (tp + fp) if tp+fp > 0 else 0
            rec = tp / (tp + fn) if tp+fn > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if prec+rec > 0 else 0
            
            f.write(f"--- {cls_name} ---\n")
            f.write(f"TP: {tp} | FP: {fp} | FN: {fn} | GT: {stats[cls_name]['gt']}\n")
            f.write(f"Precision: {prec*100:.2f}%\n")
            f.write(f"Recall:    {rec*100:.2f}%\n")
            f.write(f"F1 Score:  {f1*100:.2f}%\n\n")

if __name__ == "__main__":
    evaluate()
