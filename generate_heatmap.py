import os
import cv2
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchaudio
from torchvision import models
from model import X3DFreeKickModel
import re
import subprocess

# Config
FPS = 2
WINDOW_SECONDS = 20
WINDOW_FRAMES = WINDOW_SECONDS * FPS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def evaluate_window(video_path, t_start, video_model, audio_model):
    # Audio
    max_audio_prob = 0.0
    for chunk_idx in range(4):
        chunk_start = t_start + (chunk_idx * 5.0)
        temp_wav = f"hm_eval_{chunk_idx}_{os.getpid()}.wav"
        temp_jpg = f"hm_eval_{chunk_idx}_{os.getpid()}.jpg"
        extract_audio(video_path, chunk_start, duration=5.0, temp_wav=temp_wav)
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
                os.remove(temp_wav)
                os.remove(temp_jpg)
            except Exception as e:
                pass
                
    # Video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t_start * 1000))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0: original_fps = 25.0
    frame_interval = int(round(original_fps / FPS))
    
    frames = []
    frame_idx = 0
    while len(frames) < WINDOW_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % frame_interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            frames.append(transform(img))
        frame_idx += 1
    cap.release()
    
    if len(frames) < WINDOW_FRAMES:
        while len(frames) < WINDOW_FRAMES:
            frames.append(torch.zeros(3, 224, 224))
            
    input_tensor = torch.stack(frames, dim=1).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            video_out = video_model(input_tensor)
            video_probs = torch.softmax(video_out, dim=1)
        video_prob = video_probs[0, 1].item()
        
    if max_audio_prob > 0.85:
        final_prob = min(1.0, video_prob + (max_audio_prob * 0.3))
    else:
        final_prob = video_prob
        
    return final_prob

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

def main():
    print("Loading models...")
    video_model = X3DFreeKickModel(num_classes=2, pretrained=False)
    video_model.load_state_dict(torch.load("checkpoints/x3d_foul_best.pth", map_location=device))
    video_model = video_model.to(device)
    video_model.eval()
    
    audio_model = models.resnet34(pretrained=False)
    audio_model.fc = nn.Linear(audio_model.fc.in_features, 2)
    audio_model.load_state_dict(torch.load("checkpoints/audio_resnet34_best.pth", map_location=device))
    audio_model = audio_model.to(device)
    audio_model.eval()
    
    print("Parsing report...")
    report_file = "/home/pilot/Desktop/ballsaction/soccernet_eval_report.txt"
    data_dir = "/home/pilot/Desktop/ballsaction/soccernet_data"
    
    with open(report_file, 'r') as f:
        lines = f.readlines()
        
    current_game = None
    matched_gts = set()
    fps = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Evaluating Match:"):
            current_game = line.split("Evaluating Match: ")[1]
            
        if line.startswith("Half"):
            m = re.match(r"Half (\d+) \| Clip (\d+) \[(\d+)s - (\d+)s\] -> (.*)", line)
            if m:
                half = int(m.group(1))
                c_start = int(m.group(3))
                status = m.group(5)
                i += 1
                if status == "FALSE POSITIVE":
                    fps.append((current_game, half, c_start, "FP"))
                elif status == "TRUE POSITIVE":
                    i += 1
                    gt_line = lines[i].strip()
                    if gt_line.startswith("Matched GTs:"):
                        gt_matches = re.findall(r"([\w\s-]+?) @ (\d+)s", gt_line)
                        for lbl, sec_str in gt_matches:
                            matched_gts.add(f"{current_game}_half{half}_{sec_str}")
        i += 1
        
    eval_games = []
    for line in lines:
        if line.strip().startswith("Evaluating Match:"):
            eval_games.append(line.split("Evaluating Match: ")[1].strip())
            
    fns = []
    for game in eval_games:
        json_path = os.path.join(data_dir, game, "Labels-v2.json")
        gt_events = parse_gt(json_path)
        for half in [1, 2]:
            for sec, label in gt_events[half]:
                gt_id = f"{game}_half{half}_{sec}"
                if gt_id not in matched_gts:
                    fns.append((game, half, max(0, sec - 10), "FN"))
                    
    events = fps[:10] + fns[:10]
    print(f"Generating heatmap for {len(events)} events (10 FPs, 10 FNs)...")
    
    offsets = list(range(-5, 6))
    results = []
    labels = []
    
    for game, half, base_sec, typ in events:
        vid_path = os.path.join(data_dir, game, f"{half}_224p.mkv")
        row = []
        for offset in offsets:
            t_start = max(0, base_sec + offset)
            prob = evaluate_window(vid_path, t_start, video_model, audio_model)
            row.append(prob)
        results.append(row)
        lbl = f"[{typ}] {game.split('/')[-1]} H{half} @{base_sec}s"
        labels.append(lbl)
        print(f"Processed: {lbl}")
        
    data = np.array(results)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(data, annot=False, cmap="YlOrRd", xticklabels=offsets, yticklabels=labels)
    plt.title("1-Second Sliding Window Probabilities (-5s to +5s offset from clip start)")
    plt.xlabel("Offset (seconds)")
    plt.ylabel("Event")
    plt.tight_layout()
    
    out_path = "/home/pilot/.gemini/antigravity/brain/54051a72-f909-4f8f-afda-b38268595c9a/.user_uploaded/heatmap.png"
    plt.savefig(out_path)
    print(f"Heatmap saved to {out_path}")

if __name__ == "__main__":
    main()
