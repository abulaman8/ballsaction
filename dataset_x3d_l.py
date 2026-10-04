import os
import cv2
cv2.setNumThreads(0)
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class X3DLNativeDataset(Dataset):
    """
    Dataset loader for native 16:9 widescreen video clips (224x398).
    Supports temporal skip jitter augmentation and hard negative balancing.
    """
    def __init__(self, data_dir, is_training=True, target_frames=30, max_bg_ratio=2.5, max_samples_per_class=None):
        self.data_dir = data_dir
        self.is_training = is_training
        self.target_frames = target_frames
        
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        pos_dir = [d for d in subdirs if d != "background"][0]
        
        self.classes = {"background": 0, pos_dir: 1}
        self.clips = []
        
        for cls_name, cls_idx in self.classes.items():
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith(".mp4"):
                    self.clips.append({
                        "video_path": os.path.join(cls_dir, fname),
                        "label": cls_idx
                    })
                    
        # Apply balanced sampling
        if self.is_training or (max_samples_per_class is not None or max_bg_ratio is not None):
            random.seed(42)
            pos_clips = [c for c in self.clips if c["label"] == 1]
            bg_clips = [c for c in self.clips if c["label"] == 0]
            if max_samples_per_class is not None and len(pos_clips) > max_samples_per_class:
                pos_clips = random.sample(pos_clips, max_samples_per_class)
            if max_bg_ratio is not None:
                max_bgs = int(len(pos_clips) * max_bg_ratio)
                if len(bg_clips) > max_bgs:
                    bg_clips = random.sample(bg_clips, max_bgs)
            self.clips = pos_clips + bg_clips
            
        print(f"X3D-L Dataset ({data_dir}): {len(self.clips)} clips (Positives: {len([c for c in self.clips if c['label'] == 1])}, Backgrounds: {len([c for c in self.clips if c['label'] == 0])})")
        
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 398))
        self.normalize = transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        self.jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        cap = cv2.VideoCapture(clip_info["video_path"])
        
        frames = []
        total_in_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Temporal jitter during training: simulate random offset of 0-2 frames (0 to 1.0s)
        skip_start = 0
        if self.is_training and total_in_vid > self.target_frames:
            max_skip = min(2, total_in_vid - self.target_frames)
            skip_start = random.randint(0, max_skip)
            
        cur_idx = 0
        while cap.isOpened() and len(frames) < self.target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if cur_idx >= skip_start:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor_img = self.to_tensor(frame_rgb)
                tensor_img = self.resize(tensor_img)
                if self.is_training:
                    tensor_img = self.jitter(tensor_img)
                    if random.random() > 0.5:
                        tensor_img = torch.flip(tensor_img, dims=[2])
                tensor_img = self.normalize(tensor_img)
                frames.append(tensor_img)
            cur_idx += 1
            
        cap.release()
        
        # Pad with final frame if shorter than target
        while len(frames) < self.target_frames:
            if len(frames) > 0:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros((3, 224, 398)))
                
        # Stack to (C, T, H, W)
        video_tensor = torch.stack(frames, dim=1) # (3, 30, 224, 398)
        label = torch.tensor(clip_info["label"], dtype=torch.long)
        
        return video_tensor, label

if __name__ == "__main__":
    ds = X3DLNativeDataset("setpiece_dataset_v4/train", is_training=True, max_bg_ratio=2.5)
    v, y = ds[0]
    print(f"Sample tensor shape: {v.shape}, label: {y}")
