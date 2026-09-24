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
from torchvision import models
from model import X3DFreeKickModel
from SoccerNet.utils import getListGames
import shutil

class FusionNetwork(nn.Module):
    def __init__(self):
        super(FusionNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze(1)


FPS = 2
WINDOW_SECONDS = 20
WINDOW_FRAMES = WINDOW_SECONDS * FPS
STRIDE_SECONDS = 10
STRIDE_FRAMES = STRIDE_SECONDS * FPS
THRESHOLD = 0.85
TOLERANCE_SECONDS = 15.0

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

def extract_audio(video_path, start_sec, duration=5.0, temp_wav="temp_audio.wav"):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-q:a", "0", "-map", "a",
        temp_wav
    ]
    subprocess.run(cmd, capture_output=True)
    return temp_wav

def merge_intervals(intervals, max_gap=10.0):
    if not intervals: return []
    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    for curr in intervals[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1] + max_gap:
            prev[1] = max(prev[1], curr[1])
            prev[2] = max(prev[2], curr[2])
            prev[3] = max(prev[3], curr[3])
            prev[4] = max(prev[4], curr[4])
            prev[5] = prev[5] or curr[5]
        else:
            merged.append(list(curr))
    return merged

def process_video(video_path, video_model, audio_model, fusion_model, device, out_dir, vid_name):
    print(f"Processing {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return []
        
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25.0
        
    frame_interval = int(round(original_fps / FPS))
    frame_idx = 0
    rolling_buffer = []
    detected_windows = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_sec = frame_idx / original_fps
        if frame_idx % frame_interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            tensor_img = transform(img)
            rolling_buffer.append((tensor_img, current_sec))
            
            if len(rolling_buffer) >= WINDOW_FRAMES:
                input_frames = [x[0] for x in rolling_buffer[:WINDOW_FRAMES]]
                timestamps = [x[1] for x in rolling_buffer[:WINDOW_FRAMES]]
                t_end = timestamps[-1]
                t_start_video = max(0, t_end - WINDOW_SECONDS)
                
                max_audio_prob = 0.0
                for chunk_idx in range(4):
                    chunk_start = t_start_video + (chunk_idx * 5.0)
                    temp_wav = extract_audio(video_path, chunk_start, duration=5.0, temp_wav=f"eval_buf_{chunk_idx}.wav")
                    temp_jpg = f"eval_spec_{chunk_idx}.jpg"
                    if os.path.exists(temp_wav):
                        try:
                            waveform, sample_rate = torchaudio.load(temp_wav)
                            if waveform.shape[0] > 1:
                                waveform = waveform.mean(dim=0, keepdim=True)
                            mel_spec = torchaudio.transforms.MelSpectrogram(
                                sample_rate=sample_rate, n_fft=2048, hop_length=512, n_mels=128
                            )(waveform)
                            mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
                            plt.figure(figsize=(2.24, 2.24), dpi=100)
                            plt.imshow(mel_spec[0].numpy(), aspect='auto', origin='lower', cmap='magma')
                            plt.axis('off')
                            plt.tight_layout(pad=0)
                            plt.savefig(temp_jpg, bbox_inches='tight', pad_inches=0)
                            plt.close()
                            
                            audio_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
                            audio_img = Image.open(temp_jpg).convert('RGB')
                            audio_tensor = audio_transform(audio_img).unsqueeze(0).to(device)
                            with torch.no_grad():
                                audio_out = audio_model(audio_tensor)
                                audio_probs = torch.softmax(audio_out, dim=1)
                                max_audio_prob = max(max_audio_prob, audio_probs[0, 1].item())
                        except Exception as e:
                            pass
                
                audio_prob = max_audio_prob
                input_tensor = torch.stack(input_frames, dim=1).unsqueeze(0).to(device)
                with torch.no_grad():
                    if device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            video_out = video_model(input_tensor)
                            video_probs = torch.softmax(video_out, dim=1)
                    else:
                        video_out = video_model(input_tensor)
                        video_probs = torch.softmax(video_out, dim=1)
                    video_prob = video_probs[0, 1].item()
                
                fusion_input = torch.tensor([[video_prob, audio_prob]], dtype=torch.float32).to(device)
                with torch.no_grad():
                    final_prob = fusion_model(fusion_input).item()
                boosted = (final_prob > video_prob + 0.05)
                    
                if final_prob > THRESHOLD:
                    clip_start = max(0, t_start_video - 5.0)
                    clip_end = clip_start + 20.0
                    detected_windows.append((clip_start, clip_end, audio_prob, video_prob, final_prob, boosted))
                    
                rolling_buffer = rolling_buffer[STRIDE_FRAMES:]
        frame_idx += 1
    
    cap.release()
    merged_clips = merge_intervals(detected_windows, max_gap=5.0)
    
    # Extract clips
    for idx, clip in enumerate(merged_clips):
        c_start, c_end, a_prob, v_prob, f_prob, is_boosted = clip
        duration = c_end - c_start
        out_clip = os.path.join(out_dir, f"{vid_name}_clip{idx}_{c_start:.0f}s.mp4")
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
        
    return merged_clips

def parse_gt(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gt = {1: [], 2: []}
    target_labels = {'Foul', 'Direct free-kick', 'Indirect free-kick', 'Penalty'}
    for ann in data.get('annotations', []):
        if ann['label'] in target_labels:
            half_str, time_str = ann['gameTime'].split(' - ')
            half = int(half_str)
            mm, ss = map(int, time_str.split(':'))
            sec = mm * 60 + ss
            gt[half].append((sec, ann['label']))
    return gt

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    video_model = X3DFreeKickModel(num_classes=2, pretrained=False)
    video_model.load_state_dict(torch.load("checkpoints/x3d_foul_best.pth", map_location=device))
    video_model = video_model.to(device)
    video_model.eval()
    
    audio_model = models.resnet34(pretrained=False)
    audio_model.fc = nn.Linear(audio_model.fc.in_features, 2)
    audio_model.load_state_dict(torch.load("checkpoints/audio_resnet34_best.pth", map_location=device))
    audio_model = audio_model.to(device)
    audio_model.eval()
    
    fusion_model = FusionNetwork()
    fusion_model.load_state_dict(torch.load("checkpoints/fusion_mlp_best.pth", map_location=device))
    fusion_model = fusion_model.to(device)
    fusion_model.eval()
    
    data_dir = "/home/pilot/Desktop/ballsaction/soccernet_data"
    out_dir = "/home/pilot/Desktop/ballsaction/soccernet_eval_highlights"
    report_file = "/home/pilot/Desktop/ballsaction/soccernet_eval_report.txt"
    
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(report_file): os.remove(report_file)
    
    test_games = getListGames("test")
    # Take 5 unseen games
    eval_games = []
    for g in test_games[10:]:
        if os.path.exists(os.path.join(data_dir, g, "1_224p.mkv")):
            eval_games.append(g)
        if len(eval_games) == 5:
            break
            
    print(f"Evaluating on 5 games: {eval_games}")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    total_gt_per_class = {'Foul': 0, 'Direct free-kick': 0, 'Indirect free-kick': 0, 'Penalty': 0}
    total_tp_per_class = {'Foul': 0, 'Direct free-kick': 0, 'Indirect free-kick': 0, 'Penalty': 0}
    
    with open(report_file, "a") as f:
        f.write("SOCCERNET EVALUATION REPORT\n===========================\n\n")
        
    for game in eval_games:
        json_path = os.path.join(data_dir, game, "Labels-v2.json")
        gt_events = parse_gt(json_path)
        
        with open(report_file, "a") as f:
            f.write(f"\nEvaluating Match: {game}\n")
            f.write("-" * 50 + "\n")
            
        for half in [1, 2]:
            vid_path = os.path.join(data_dir, game, f"{half}_224p.mkv")
            if not os.path.exists(vid_path): continue
            
            vid_name = f"{game.replace('/', '_')}_half{half}"
            merged_clips = process_video(vid_path, video_model, audio_model, fusion_model, device, out_dir, vid_name)
            gt_half = gt_events[half]
            for gt_time, gt_label in gt_half:
                total_gt_per_class[gt_label] += 1
                
            # Match predictions to GT
            matched_gt = set()
            half_tp, half_fp = 0, 0
            
            for idx, clip in enumerate(merged_clips):
                c_start, c_end, a_prob, v_prob, f_prob, is_boosted = clip
                duration = c_end - c_start
                
                # Check if any GT event is within [c_start - TOLERANCE, c_end + TOLERANCE]
                hit_gts = [(gt_t, gt_l) for gt_t, gt_l in gt_half if c_start - TOLERANCE_SECONDS <= gt_t <= c_end + TOLERANCE_SECONDS]
                
                if hit_gts:
                    half_tp += 1
                    status = "TRUE POSITIVE"
                    for gt_t, gt_l in hit_gts: 
                        if (gt_t, gt_l) not in matched_gt:
                            matched_gt.add((gt_t, gt_l))
                            total_tp_per_class[gt_l] += 1
                else:
                    half_fp += 1
                    status = "FALSE POSITIVE"
                    
                with open(report_file, "a") as f:
                    f.write(f"Half {half} | Clip {idx} [{c_start:.0f}s - {c_end:.0f}s] -> {status}\n")
                    f.write(f"  Max Audio: {a_prob*100:.1f}%, Max Video: {v_prob*100:.1f}%, Max Fused: {f_prob*100:.1f}% {'(Boosted)' if is_boosted else ''}\n")
                    if hit_gts:
                        f.write(f"  Matched GTs: {[f'{gt_l} @ {gt_t}s' for gt_t, gt_l in hit_gts]}\n")
                    f.write("\n")
                    
            half_fn = len(gt_half) - len(matched_gt)
            total_tp += half_tp
            total_fp += half_fp
            total_fn += half_fn
            
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    with open(report_file, "a") as f:
        f.write("\n===========================\nOVERALL METRICS\n===========================\n")
        f.write(f"True Positives: {total_tp}\n")
        f.write(f"False Positives: {total_fp}\n")
        f.write(f"False Negatives: {total_fn}\n\n")
        f.write(f"Overall Precision: {precision*100:.2f}%\n")
        f.write(f"Overall Recall:    {recall*100:.2f}%\n")
        f.write(f"Overall F1 Score:  {f1*100:.2f}%\n\n")
        f.write("PER-CLASS RECALL\n")
        f.write("---------------------------\n")
        for cls_name in total_gt_per_class:
            c_gt = total_gt_per_class[cls_name]
            c_tp = total_tp_per_class[cls_name]
            c_rec = (c_tp / c_gt * 100) if c_gt > 0 else 0
            f.write(f"{cls_name}: {c_rec:.2f}% ({c_tp}/{c_gt})\n")
        
    print(f"\nEvaluation complete. Report saved to {report_file}")

if __name__ == "__main__":
    evaluate()
