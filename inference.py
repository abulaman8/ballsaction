import os
import cv2
import torch
import numpy as np
import subprocess
import shutil
from model import X3DFreeKickModel
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import time
import glob
from tqdm import tqdm

class FrameWindowDataset(Dataset):
    def __init__(self, frame_dir, num_frames, window_size=40, stride=4):
        # stride=4 frames at 2fps = 2 seconds
        self.frame_dir = frame_dir
        self.window_size = window_size
        self.stride = stride
        self.num_frames = num_frames
        
        # Valid starting indices
        self.start_indices = list(range(1, max(2, self.num_frames - self.window_size + 2), self.stride))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        frames = []
        
        for i in range(start_idx, start_idx + self.window_size):
            # Clamp to max frame just in case
            frame_idx = min(i, self.num_frames)
            frame_path = os.path.join(self.frame_dir, f"frame_{frame_idx:06d}.jpg")
            
            if os.path.exists(frame_path):
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(img))
            else:
                # Fallback if missing
                if len(frames) > 0:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros((3, 224, 224)))
                    
        video_tensor = torch.stack(frames, dim=1) # (C, T, H, W)
        time_sec = (start_idx - 1) / 2.0 # 2 FPS
        return video_tensor, time_sec

def process_video(video_path, model, device, output_dir, threshold=0.8, batch_size=16):
    print(f"\n[{time.strftime('%H:%M:%S')}] Processing: {os.path.basename(video_path)}")
    
    # 1. Extract frames at 2fps and 224x224
    temp_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    print("  -> Extracting frames via ffmpeg (2 fps, 224x224)...")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", video_path, 
        "-r", "2", "-s", "224x224", 
        "-q:v", "2", # High quality JPEG
        os.path.join(temp_dir, "frame_%06d.jpg")
    ]
    # Run silently
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    extracted_frames = len(glob.glob(os.path.join(temp_dir, "*.jpg")))
    if extracted_frames < 40:
        print("  -> Video too short. Skipping.")
        shutil.rmtree(temp_dir)
        return
        
    print(f"  -> Extracted {extracted_frames} frames.")
    
    # 2. Run Sliding Window Inference
    dataset = FrameWindowDataset(temp_dir, extracted_frames, window_size=40, stride=4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print("  -> Running model inference...")
    detected_highlights = []
    
    with torch.no_grad():
        for inputs, times in tqdm(loader, desc="Inference"):
            inputs = inputs.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
            # Class 1 is Set Piece
            set_piece_probs = probs[:, 1].cpu().numpy()
            times = times.numpy()
            
            for prob, t in zip(set_piece_probs, times):
                if prob > threshold:
                    detected_highlights.append((t, prob))
                    
    # 3. Post-process (Non-Maximum Suppression / Cooldown)
    # We want to avoid extracting the same free-kick multiple times.
    # Group overlapping segments and keep the max probability.
    final_clips = []
    detected_highlights.sort(key=lambda x: x[0]) # Sort by time
    
    current_group = []
    for t, p in detected_highlights:
        if not current_group:
            current_group.append((t, p))
        else:
            # If within 20 seconds of the first item in the group, add to group
            if t - current_group[0][0] <= 20.0:
                current_group.append((t, p))
            else:
                # Resolve group by taking max prob, using the time of the max prob
                best_t, best_p = max(current_group, key=lambda x: x[1])
                final_clips.append((best_t, best_p))
                current_group = [(t, p)]
                
    if current_group:
        best_t, best_p = max(current_group, key=lambda x: x[1])
        final_clips.append((best_t, best_p))
            
    print(f"  -> Detected {len(final_clips)} unique set pieces!")
    
    # 4. Extract clips
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    for idx, (start_time, prob) in enumerate(final_clips):
        if prob >= 0.95:
            target_dir = os.path.join(output_dir, "conf_95")
        elif prob >= 0.90:
            target_dir = os.path.join(output_dir, "conf_90")
        else:
            target_dir = os.path.join(output_dir, "conf_85")
            
        out_clip = os.path.join(target_dir, f"{vid_name}_prob{prob:.2f}_sec{int(start_time)}.mp4")
        
        # Use fast seek for cutting
        cut_cmd = [
            "ffmpeg", "-y", 
            "-ss", str(start_time), 
            "-i", video_path,
            "-t", "20",
            "-c:v", "copy", "-c:a", "copy",
            out_clip
        ]
        subprocess.run(cut_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    print("  -> Extraction complete.")
    
    # Cleanup temp frames
    shutil.rmtree(temp_dir)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    print("Loading X3D model...")
    model = X3DFreeKickModel(num_classes=2, pretrained=False)
    
    ckpt_path = "checkpoints/x3d_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    data_dir = "/home/pilot/Desktop/ballsaction/custom_data"
    out_dir = "/home/pilot/Desktop/ballsaction/custom_data_highlights"
    os.makedirs(os.path.join(out_dir, "conf_85"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "conf_90"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "conf_95"), exist_ok=True)
    
    videos = glob.glob(os.path.join(data_dir, "*.mp4"))
    if len(videos) == 0:
        print("No videos found in custom_data/")
        return
        
    print(f"Found {len(videos)} videos in custom_data/. Starting inference pipeline...")
    
    for vid in videos:
        process_video(vid, model, device, out_dir, threshold=0.85, batch_size=32)
        
    print(f"\nAll done! Highlights saved to {out_dir}")

if __name__ == "__main__":
    main()
