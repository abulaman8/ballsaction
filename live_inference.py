import os
import cv2
import time
import glob
import sys
import torch
import torch.nn as nn
import random
import queue
import threading
import subprocess
import torchvision.transforms as transforms
import torchaudio
from model import X3DFreeKickModel
from model_audio import WhistleNet

# Configurations
FPS = 2
WINDOW_SECONDS = 10
WINDOW_FRAMES = WINDOW_SECONDS * FPS # 20 frames
STRIDE_SECONDS = 5
STRIDE_FRAMES = STRIDE_SECONDS * FPS # 10 frames
THRESHOLD = 0.85
DATA_DIR = "/home/pilot/Desktop/ballsaction/custom_data_2_compressed"

# If custom_data_2_compressed doesn't exist or is empty, fallback to custom_data
if not os.path.exists(DATA_DIR) or len(glob.glob(os.path.join(DATA_DIR, "*.mp4"))) == 0:
    DATA_DIR = "/home/pilot/Desktop/ballsaction/custom_data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

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

def get_audio_window(full_waveform, t_start, t_end):
    s_start = int(t_start * 16000)
    s_end = int(t_end * 16000)
    chunk = full_waveform[:, s_start:s_end]
    target_len = int((t_end - t_start) * 16000)
    
    if chunk.shape[1] < target_len:
        pad = target_len - chunk.shape[1]
        chunk = torch.nn.functional.pad(chunk, (0, pad))
        
    spec = mel_transform(chunk)
    spec = amp_to_db(spec)
    if spec.shape[2] < 313: 
        spec = torch.nn.functional.pad(spec, (0, 313 - spec.shape[2]))
    else: 
        spec = spec[:, :, :313]
    return spec.unsqueeze(0)

def consumer(frame_queue, stop_event, video_model, audio_model, video_path, device):
    """Consumes the live frames, builds the 10-second rolling window, and runs Dual Inference."""
    print("[Consumer] Extracting full audio for simulation...")
    full_audio_wav = "live_full_audio.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-q:a", "0", "-map", "a", full_audio_wav], capture_output=True)
    
    try:
        full_waveform, sr = torchaudio.load(full_audio_wav)
        if full_waveform.shape[0] > 1: full_waveform = torch.mean(full_waveform, dim=0, keepdim=True)
        if sr != 16000: full_waveform = torchaudio.transforms.Resample(sr, 16000)(full_waveform)
    except Exception:
        print("Warning: Audio extraction failed or no audio track found.")
        full_waveform = torch.zeros((1, 150 * 60 * 16000))
    
    print("[Consumer] Waiting for frames to build the initial 10-second buffer...")
    rolling_buffer = []
    
    # Create output directory for saved highlights
    highlights_dir = "live_highlights"
    os.makedirs(highlights_dir, exist_ok=True)
    
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            # Non-blocking get with timeout to check stop_event frequently
            item = frame_queue.get(timeout=0.1)
            rolling_buffer.append(item)
            
            # Once we hit exactly WINDOW_FRAMES, run a prediction
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
                audio_tensor = get_audio_window(full_waveform, t_start_video, t_end).to(device)
                with torch.no_grad():
                    audio_out = audio_model(audio_tensor)
                    audio_probs = torch.softmax(audio_out, dim=1)
                    audio_prob = audio_probs[0, 1].item()
                
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
                if audio_prob > 0.7:
                    final_prob = min(1.0, video_prob + 0.05)
                    fusion_note = "(Boosted by Whistle!)"
                else:
                    final_prob = video_prob
                    fusion_note = ""
                    
                if final_prob > THRESHOLD:
                    status = "DETECTED HIGHLIGHT! (Clipping to disk...)"
                    # Asynchronously save the 10-second window so we don't block the live feed!
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
                
                # Slide window forward
                rolling_buffer = rolling_buffer[STRIDE_FRAMES:]
                
        except queue.Empty:
            continue

    print("[Consumer] Finished.")

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
    print("Loading Audio WhistleNet Model...")
    audio_model = WhistleNet()
    audio_ckpt = "checkpoints/audio_whistle_best.pth"
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
