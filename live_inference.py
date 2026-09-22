import os
import cv2
import time
import glob
import torch
import torch.nn as nn
import random
import queue
import threading
import subprocess
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchaudio
from torchvision import models
from model import X3DFreeKickModel

# Configurations
FPS = 2
WINDOW_SECONDS = 20
WINDOW_FRAMES = WINDOW_SECONDS * FPS # 40 frames
STRIDE_SECONDS = 10
STRIDE_FRAMES = STRIDE_SECONDS * FPS # 20 frames
THRESHOLD = 0.85
DATA_DIR = "/home/pilot/Desktop/ballsaction/custom_data_2_compressed"

# If custom_data_2_compressed doesn't exist or is empty, fallback to custom_data
if not os.path.exists(DATA_DIR) or len(glob.glob(os.path.join(DATA_DIR, "*.mp4"))) == 0:
    DATA_DIR = "/home/pilot/Desktop/ballsaction/custom_data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

def producer(video_path, frame_queue, stop_event):
    """Simulates a live broadcast camera feed by reading a video in real-time."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        stop_event.set()
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25.0
        
    print(f"[Producer] Started simulating live feed from {os.path.basename(video_path)} at {original_fps} FPS.")
    
    frame_delay = 1.0 / original_fps
    frame_interval = int(round(original_fps / FPS))
    frame_idx = 0
    
    while not stop_event.is_set():
        start_t = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("[Producer] Video ended.")
            break
            
        current_sec = frame_idx / original_fps
            
        # We only want to push 2 frames every second to the AI
        if frame_idx % frame_interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            tensor_img = transform(img)
            # Pass both the frame and its exact timestamp
            frame_queue.put((tensor_img, current_sec))
            
        frame_idx += 1
        
        # Sleep to perfectly simulate real-time playback
        elapsed = time.time() - start_t
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    stop_event.set()

def extract_live_audio(video_path, start_sec, duration=5.0, temp_wav="live_buffer.wav"):
    """Dynamically extracts a 5-second slice of audio using ffmpeg."""
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

def consumer(frame_queue, stop_event, video_model, audio_model, video_path, device):
    """Consumes the live frames, builds the 20-second rolling window, and runs Dual Inference."""
    print("[Consumer] Waiting for frames to build the initial 20-second buffer...")
    rolling_buffer = []
    
    # Create output directory for saved highlights
    highlights_dir = "live_highlights"
    os.makedirs(highlights_dir, exist_ok=True)
    
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            # Non-blocking get with timeout to check stop_event frequently
            item = frame_queue.get(timeout=0.1)
            rolling_buffer.append(item)
            
            # Once we hit exactly 40 frames (20 seconds), run a prediction
            if len(rolling_buffer) >= WINDOW_FRAMES:
                # Unzip frames and timestamps
                input_frames = [x[0] for x in rolling_buffer[:WINDOW_FRAMES]]
                timestamps = [x[1] for x in rolling_buffer[:WINDOW_FRAMES]]
                
                t_end = timestamps[-1]
                t_start_video = max(0, t_end - WINDOW_SECONDS)
                
                print(f"\n" + "="*60)
                print(f"[LIVE INFERENCE TRIGGERED] Window: {timestamps[0]:.1f}s -> {t_end:.1f}s")
                
                inf_start = time.time()
                
                # --- AUDIO PIPELINE ---
                max_audio_prob = 0.0
                
                # A foul happens at varying times within the 20s window, and the whistle follows it.
                # Scan four 5-second audio chunks across the entire window and take the max whistle confidence.
                for chunk_idx in range(4):
                    chunk_start = t_start_video + (chunk_idx * 5.0)
                    temp_wav = extract_live_audio(video_path, chunk_start, duration=5.0, temp_wav=f"live_buffer_{chunk_idx}.wav")
                    temp_jpg = f"live_spec_{chunk_idx}.jpg"
                    
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
                
                # --- VIDEO PIPELINE ---
                input_tensor = torch.stack(input_frames, dim=1).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        video_out = video_model(input_tensor)
                        video_probs = torch.softmax(video_out, dim=1)
                        video_prob = video_probs[0, 1].item()
                        
                inf_end = time.time()
                latency_ms = (inf_end - inf_start) * 1000
                
                # --- ASYMMETRIC FUSION ---
                # If audio hears a whistle (>85%), boost the video confidence.
                # If audio hears nothing, rely entirely on video (due to low recall).
                if audio_prob > 0.85:
                    final_prob = min(1.0, video_prob + (audio_prob * 0.3))
                    fusion_note = "(Boosted by Whistle!)"
                else:
                    final_prob = video_prob
                    fusion_note = ""
                    
                if final_prob > THRESHOLD:
                    status = "DETECTED HIGHLIGHT! (Clipping to disk...)"
                    # Asynchronously save the 20-second window so we don't block the live feed!
                    out_clip = os.path.join(highlights_dir, f"highlight_{t_start_video:.0f}s_to_{t_end:.0f}s.mp4")
                    
                    clip_cmd = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-ss", str(t_start_video),
                        "-i", video_path,
                        "-t", str(WINDOW_SECONDS),
                        "-c", "copy", # Super fast stream copy without re-encoding
                        out_clip
                    ]
                    subprocess.Popen(clip_cmd) # Fire and forget!
                else:
                    status = "Background"
                
                print(f"  --> Audio CNN (Whistle) Conf : {audio_prob*100:5.1f}%")
                print(f"  --> Video X3D (Visual) Conf  : {video_prob*100:5.1f}%")
                print(f"  --> FUSED CONFIDENCE         : {final_prob*100:5.1f}% {fusion_note}")
                print(f"  --> ACTION                   : {status}")
                print(f"  [Metrics] Dual-Inference Latency: {latency_ms:.1f}ms | Queue Backlog: {frame_queue.qsize()} frames")
                print("="*60)
                
                # Slide window forward by 10 seconds (drop the oldest 20 frames)
                rolling_buffer = rolling_buffer[STRIDE_FRAMES:]
                
        except queue.Empty:
            continue

    print("[Consumer] Finished.")

import sys

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System Check: Using {device} for inference")
    
    # 1. Load Video Model
    print("Loading V2 X3D Video Model...")
    video_model = X3DFreeKickModel(num_classes=2, pretrained=False)
    video_ckpt = "checkpoints/x3d_foul_best.pth"
    if os.path.exists(video_ckpt):
        video_model.load_state_dict(torch.load(video_ckpt, map_location=device))
        print("-> Video Weights loaded successfully!")
    else:
        print("WARNING: X3D Checkpoint not found! Using random weights.")
    video_model = video_model.to(device)
    video_model.eval()
    
    # 2. Load Audio Model
    print("Loading Audio ResNet-18 Model...")
    audio_model = models.resnet18(pretrained=False)
    audio_model.fc = nn.Linear(audio_model.fc.in_features, 2)
    audio_ckpt = "checkpoints/audio_resnet18_best.pth"
    if os.path.exists(audio_ckpt):
        audio_model.load_state_dict(torch.load(audio_ckpt, map_location=device))
        print("-> Audio Weights loaded successfully!")
    else:
        print("WARNING: Audio Checkpoint not found! Using random weights.")
    audio_model = audio_model.to(device)
    audio_model.eval()

    if len(sys.argv) > 1:
        test_video = sys.argv[1]
    else:
        test_video = os.path.join(DATA_DIR, "MD1 STP-BVB-012_720p.mp4")
        
    if not os.path.exists(test_video):
        print(f"Requested video not found: {test_video}")
        return
    
    frame_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Spawn Threads
    prod_thread = threading.Thread(target=producer, args=(test_video, frame_queue, stop_event))
    cons_thread = threading.Thread(target=consumer, args=(frame_queue, stop_event, video_model, audio_model, test_video, device))
    
    prod_thread.start()
    cons_thread.start()
    
    try:
        prod_thread.join()
        cons_thread.join()
    except KeyboardInterrupt:
        print("\nStopping threads gracefully...")
        stop_event.set()
        prod_thread.join()
        cons_thread.join()
        
    print("Live feed simulation terminated.")

if __name__ == "__main__":
    main()
