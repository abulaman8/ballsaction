import os
import cv2
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchaudio
from torchvision import models
from model import X3DFreeKickModel
import subprocess
import glob
import random

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

def evaluate_clip(clip_path, video_model, audio_model):
    max_audio_prob = 0.0
    for chunk_idx in range(4):
        chunk_start = chunk_idx * 5.0
        temp_wav = f"fus_eval_{chunk_idx}_{os.getpid()}.wav"
        temp_jpg = f"fus_eval_{chunk_idx}_{os.getpid()}.jpg"
        extract_audio(clip_path, chunk_start, duration=5.0, temp_wav=temp_wav)
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
                
    cap = cv2.VideoCapture(clip_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0: original_fps = 25.0
    frame_interval = int(round(original_fps / FPS))
    if frame_interval <= 0: frame_interval = 1
    
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
        
    return max_audio_prob, video_prob

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
    
    dataset_dir = "/home/pilot/Desktop/ballsaction/foul_dataset/train"
    foul_clips = glob.glob(os.path.join(dataset_dir, "set_piece", "*.mp4"))
    bg_clips = glob.glob(os.path.join(dataset_dir, "background", "*.mp4"))
    
    random.shuffle(foul_clips)
    random.shuffle(bg_clips)
    
    # Let's do 1,000 fouls and 1,000 backgrounds to keep it under 15 minutes
    sample_foul = foul_clips[:1000]
    sample_bg = bg_clips[:1000]
    
    print(f"Extracting probabilities for {len(sample_foul)} fouls and {len(sample_bg)} backgrounds...")
    
    results = []
    
    for i, clip in enumerate(sample_foul):
        if i % 100 == 0: print(f"Processing Fouls: {i}/{len(sample_foul)}")
        a_prob, v_prob = evaluate_clip(clip, video_model, audio_model)
        results.append({"video_prob": v_prob, "audio_prob": a_prob, "label": 1})
        
    for i, clip in enumerate(sample_bg):
        if i % 100 == 0: print(f"Processing Backgrounds: {i}/{len(sample_bg)}")
        a_prob, v_prob = evaluate_clip(clip, video_model, audio_model)
        results.append({"video_prob": v_prob, "audio_prob": a_prob, "label": 0})
        
    out_file = "/home/pilot/Desktop/ballsaction/fusion_dataset.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Done! Saved {len(results)} pairs to {out_file}")

if __name__ == "__main__":
    main()
