import os
import cv2
import time
import glob
import torch
import random
import queue
import threading
import torchvision.transforms as transforms
from model import X3DFreeKickModel

# Configurations
FPS = 2
WINDOW_SECONDS = 20
WINDOW_FRAMES = WINDOW_SECONDS * FPS # 40 frames
STRIDE_SECONDS = 10
STRIDE_FRAMES = STRIDE_SECONDS * FPS # 20 frames
THRESHOLD = 0.85
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
        
    print(f"[Producer] Started simulating live feed at {original_fps} FPS.")
    
    frame_delay = 1.0 / original_fps
    frame_interval = int(round(original_fps / FPS))
    frame_idx = 0
    
    while not stop_event.is_set():
        start_t = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("[Producer] Video ended.")
            break
            
        # We only want to push 2 frames every second to the AI
        if frame_idx % frame_interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            tensor_img = transform(img)
            frame_queue.put(tensor_img)
            
        frame_idx += 1
        
        # Sleep to perfectly simulate real-time playback
        elapsed = time.time() - start_t
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    stop_event.set()

def consumer(frame_queue, stop_event, model, device):
    """Consumes the live frames, builds the 20-second rolling window, and predicts."""
    print("[Consumer] Waiting for frames to build the initial 20-second buffer...")
    rolling_buffer = []
    
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            # Non-blocking get with timeout to check stop_event frequently
            frame = frame_queue.get(timeout=0.1)
            rolling_buffer.append(frame)
            
            # Once we hit exactly 40 frames (20 seconds), run a prediction
            if len(rolling_buffer) >= WINDOW_FRAMES:
                input_frames = rolling_buffer[:WINDOW_FRAMES]
                
                # Stack to (C, T, H, W) and add batch dim
                input_tensor = torch.stack(input_frames, dim=1).unsqueeze(0).to(device)
                
                # Execute Inference & Measure Latency
                inf_start = time.time()
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        prob = probs[0, 1].item()
                inf_end = time.time()
                
                latency_ms = (inf_end - inf_start) * 1000
                q_size = frame_queue.qsize()
                
                status = "DETECTED SET PIECE!" if prob > THRESHOLD else "Background"
                
                print(f"\n[Consumer] Evaluated rolling window | Confidence: {prob*100:.1f}% -> {status}")
                print(f"           GPU Latency: {latency_ms:.1f} ms | Queue Backlog: {q_size} frames")
                
                # Slide window forward by 10 seconds (drop the oldest 20 frames)
                rolling_buffer = rolling_buffer[STRIDE_FRAMES:]
                
        except queue.Empty:
            continue

    print("[Consumer] Finished.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System Check: Using {device} for inference")
    
    print("Loading X3D model weights...")
    model = X3DFreeKickModel(num_classes=2, pretrained=False)
    ckpt_path = "checkpoints/x3d_best.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("-> Weights loaded successfully!")
    else:
        print("WARNING: Checkpoint not found! Using random weights.")
        
    model = model.to(device)
    model.eval()

    videos = glob.glob(os.path.join(DATA_DIR, "*.mp4"))
    if not videos:
        print("No videos found in custom_data.")
        return
        
    test_video = random.choice(videos)
    print(f"Selected random broadcast for simulation: {os.path.basename(test_video)}\n")
    
    frame_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Spawn Threads
    prod_thread = threading.Thread(target=producer, args=(test_video, frame_queue, stop_event))
    cons_thread = threading.Thread(target=consumer, args=(frame_queue, stop_event, model, device))
    
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
