import os
import cv2
import glob
import torch
import torch.nn as nn
import queue
import threading
import subprocess
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchaudio
from torchvision import models
from model import X3DFreeKickModel
import shutil

def merge_intervals(intervals, max_gap=10.0):
    if not intervals: return []
    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    for curr in intervals[1:]:
        prev = merged[-1]
        # curr[0] is start_time, prev[1] is end_time
        if curr[0] <= prev[1] + max_gap:
            prev[1] = max(prev[1], curr[1])
            prev[2] = max(prev[2], curr[2]) # audio
            prev[3] = max(prev[3], curr[3]) # video
            prev[4] = max(prev[4], curr[4]) # fused
            prev[5] = prev[5] or curr[5]    # boosted
        else:
            merged.append(list(curr))
    return merged

FPS = 2
WINDOW_SECONDS = 20
WINDOW_FRAMES = WINDOW_SECONDS * FPS
STRIDE_SECONDS = 10
STRIDE_FRAMES = STRIDE_SECONDS * FPS
THRESHOLD = 0.85

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

def process_video(video_path, video_model, audio_model, device, out_dir, report_file):
    print(f"\nProcessing {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
        
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25.0
        
    frame_interval = int(round(original_fps / FPS))
    frame_idx = 0
    rolling_buffer = []
    
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    clip_count = 0
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
                
                # Audio Pipeline
                max_audio_prob = 0.0
                for chunk_idx in range(4):
                    chunk_start = t_start_video + (chunk_idx * 5.0)
                    temp_wav = extract_audio(video_path, chunk_start, duration=5.0, temp_wav=f"batch_buf_{chunk_idx}.wav")
                    temp_jpg = f"batch_spec_{chunk_idx}.jpg"
                    
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
                
                # Video Pipeline
                input_tensor = torch.stack(input_frames, dim=1).unsqueeze(0).to(device)
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        video_out = video_model(input_tensor)
                        video_probs = torch.softmax(video_out, dim=1)
                        video_prob = video_probs[0, 1].item()
                
                # Fusion
                if audio_prob > 0.85:
                    final_prob = min(1.0, video_prob + (audio_prob * 0.3))
                    boosted = True
                else:
                    final_prob = video_prob
                    boosted = False
                    
                if final_prob > THRESHOLD:
                    clip_start = max(0, t_start_video - 5.0)
                    clip_end = clip_start + 20.0
                    detected_windows.append((clip_start, clip_end, audio_prob, video_prob, final_prob, boosted))
                    
                # Slide window
                rolling_buffer = rolling_buffer[STRIDE_FRAMES:]
                
        frame_idx += 1
    
    cap.release()
    
    merged_clips = merge_intervals(detected_windows, max_gap=5.0)
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
        
        with open(report_file, "a") as f:
            f.write(f"[{vid_name}] Merged Clip {idx} (Clipped: {c_start:.1f}s -> {c_end:.1f}s, Dur: {duration:.1f}s)\n")
            f.write(f"  - Max Audio (Whistle): {a_prob*100:.1f}%\n")
            f.write(f"  - Max Video (Foul): {v_prob*100:.1f}%\n")
            f.write(f"  - Max Fused: {f_prob*100:.1f}% {'(Boosted)' if is_boosted else ''}\n\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading V2 X3D Video Model...")
    video_model = X3DFreeKickModel(num_classes=2, pretrained=False)
    video_model.load_state_dict(torch.load("checkpoints/x3d_foul_best.pth", map_location=device))
    video_model = video_model.to(device)
    video_model.eval()
    
    print("Loading Audio ResNet-18 Model...")
    audio_model = models.resnet18(pretrained=False)
    audio_model.fc = nn.Linear(audio_model.fc.in_features, 2)
    audio_model.load_state_dict(torch.load("checkpoints/audio_resnet18_best.pth", map_location=device))
    audio_model = audio_model.to(device)
    audio_model.eval()
    
    data_dir = "/home/pilot/Desktop/ballsaction/custom_data_2_compressed"
    out_dir = "/home/pilot/Desktop/ballsaction/custom_highlights"
    report_file = "/home/pilot/Desktop/ballsaction/batch_report.txt"
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.exists(report_file):
        os.remove(report_file)
        
    videos = glob.glob(os.path.join(data_dir, "*.mp4")) + glob.glob(os.path.join(data_dir, "*.mkv"))
    for vid in videos:
        process_video(vid, video_model, audio_model, device, out_dir, report_file)
        
    print("Batch processing complete.")

if __name__ == "__main__":
    main()
