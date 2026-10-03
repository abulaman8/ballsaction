import os
import cv2
cv2.setNumThreads(0)
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class X3DBinaryDataset(Dataset):
    def __init__(self, data_dir, is_training=True, target_frames=30, max_bg_ratio=None):
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
                        "path": os.path.join(cls_dir, fname),
                        "label": cls_idx
                    })
                    
        if max_bg_ratio is not None and self.is_training:
            pos_clips = [c for c in self.clips if c["label"] == 1]
            bg_clips = [c for c in self.clips if c["label"] == 0]
            max_bgs = int(len(pos_clips) * max_bg_ratio)
            if len(bg_clips) > max_bgs:
                import random
                random.seed(42)
                bg_clips = random.sample(bg_clips, max_bgs)
                self.clips = pos_clips + bg_clips
                    
        print(f"Dataset initialized with {len(self.clips)} valid mp4 clips from {data_dir}.")
        
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        self.jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        video_path = clip_info["path"]
        label = clip_info["label"]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Determine consistent flip for the entire sequence if training
        do_flip = self.is_training and torch.rand(1).item() < 0.5
        
        # Temporal augmentation: randomly skip first 0-4 frames (0 to 2 seconds) during training
        if self.is_training:
            skip_frames = int(torch.randint(0, 5, (1,)).item())
            for _ in range(skip_frames):
                ret, _ = cap.read()
                if not ret: break
        
        while len(frames) < self.target_frames:
            ret, frame = cap.read()
            if not ret: break
            
            if do_flip:
                frame = cv2.flip(frame, 1)
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Convert to [0, 1] Tensor and Resize
            t_frame = self.to_tensor(frame)
            t_frame = self.resize(t_frame)
            
            # 2. Apply color jitter frame-by-frame if training (MUST happen before normalization)
            if self.is_training:
                t_frame = self.jitter(t_frame)
                
            # 3. Normalize (can result in negative values, which breaks ColorJitter)
            t_frame = self.normalize(t_frame)
                
            frames.append(t_frame)
            
        cap.release()
        
        while len(frames) < self.target_frames:
            if len(frames) > 0:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros((3, 224, 224)))
                
        video_tensor = torch.stack(frames, dim=1)
        return video_tensor, torch.tensor(label, dtype=torch.long)

