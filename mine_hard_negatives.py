import os
import cv2
cv2.setNumThreads(0)
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model import X3DFreeKickModel

class BackgroundClipsDataset(Dataset):
    def __init__(self, bg_dir):
        self.bg_dir = bg_dir
        self.clip_paths = []
        for fname in os.listdir(bg_dir):
            if fname.endswith(".mp4"):
                self.clip_paths.append(os.path.join(bg_dir, fname))
        print(f"Found {len(self.clip_paths)} background clips to inspect in {bg_dir}.")
        
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        video_path = self.clip_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < 20:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_frame = self.to_tensor(frame)
            t_frame = self.resize(t_frame)
            t_frame = self.normalize(t_frame)
            frames.append(t_frame)
            
        cap.release()
        
        while len(frames) < 20:
            if len(frames) > 0:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros((3, 224, 224)))
                
        video_tensor = torch.stack(frames, dim=1) # (C, T, H, W)
        return video_tensor, video_path

def mine_hard_negatives(
    checkpoint_path="checkpoints/x3d_foul_best.pth",
    bg_dir="foul_dataset_v2/train/background",
    out_json="hard_negatives_foul.json",
    batch_size=16,
    num_workers=4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load multi-head foul model
    model = X3DFreeKickModel(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded trained foul checkpoint: {checkpoint_path}")
    
    dataset = BackgroundClipsDataset(bg_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    hard_negatives = []
    total_clips = len(dataset)
    t_start = time.time()
    
    print("\nStarting False Alarm Mining across background clips...")
    print("=" * 70)
    
    with torch.no_grad():
        for batch_idx, (videos, paths) in enumerate(loader):
            videos = videos.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(videos)
                probs = torch.softmax(outputs, dim=1)[:, 1] # Foul prob
                
            probs_np = probs.cpu().numpy()
            
            for path, p in zip(paths, probs_np):
                # Filter for non-trivial false alarm probabilities
                if p >= 0.30:
                    hard_negatives.append({
                        "path": path,
                        "foul_prob": float(p)
                    })
                    
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                elapsed = time.time() - t_start
                done_clips = min((batch_idx + 1) * batch_size, total_clips)
                pct = 100.0 * done_clips / total_clips
                fps = done_clips / (elapsed + 1e-6)
                print(f"[{done_clips}/{total_clips} ({pct:.1f}%)] | Speed: {fps:.1f} clips/s | Hard Negatives (P>=0.30) found: {len(hard_negatives)}")
                
    # Sort descending by false alarm confidence
    hard_negatives.sort(key=lambda x: x["foul_prob"], reverse=True)
    
    # Summary stats
    c_50 = sum(1 for x in hard_negatives if x["foul_prob"] >= 0.50)
    c_60 = sum(1 for x in hard_negatives if x["foul_prob"] >= 0.60)
    c_70 = sum(1 for x in hard_negatives if x["foul_prob"] >= 0.70)
    c_80 = sum(1 for x in hard_negatives if x["foul_prob"] >= 0.80)
    
    print("\n" + "=" * 70)
    print("HARD NEGATIVE MINING COMPLETED")
    print("=" * 70)
    print(f"Total Background Clips Scanned: {total_clips}")
    print(f"Total False Alarms (P >= 0.30): {len(hard_negatives)} ({len(hard_negatives)/total_clips*100:.2f}%)")
    print(f"  P >= 0.50 (Confused):        {c_50} ({c_50/total_clips*100:.2f}%)")
    print(f"  P >= 0.60 (Strong Alarms):   {c_60} ({c_60/total_clips*100:.2f}%)")
    print(f"  P >= 0.70 (Severe Alarms):   {c_70} ({c_70/total_clips*100:.2f}%)")
    print(f"  P >= 0.80 (Critical Alarms): {c_80} ({c_80/total_clips*100:.2f}%)")
    
    with open(out_json, "w") as f:
        json.dump(hard_negatives, f, indent=2)
        
    print(f"Saved ranked hard negative list to: {out_json}")
    
    if hard_negatives:
        print("\nTop 5 Hardest False Alarm Clips:")
        for i, item in enumerate(hard_negatives[:5]):
            print(f"  #{i+1}: {os.path.basename(item['path'])} -> P(Foul) = {item['foul_prob']:.4f}")

if __name__ == "__main__":
    mine_hard_negatives()
