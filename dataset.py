import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class X3DBinaryDataset(Dataset):
    def __init__(self, data_dir, is_training=True):
        self.data_dir = data_dir
        self.is_training = is_training
        
        self.classes = {"background": 0, "set_piece": 1}
        self.clips = []
        
        # Load all valid mp4 files
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
                    
        print(f"Dataset initialized with {len(self.clips)} valid clips from {data_dir}.")
        
        # X3D preprocessing expects normalized tensors
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        video_path = clip_info["path"]
        label = clip_info["label"]
        
        # Read video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < 40:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame) # Shape: (C, H, W)
            frames.append(frame_tensor)
            
        cap.release()
        
        # Padding just in case a video is slightly short (though ffmpeg should make it 40)
        while len(frames) < 40:
            if len(frames) > 0:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros((3, 224, 224)))
                
        # Stack frames along Temporal dimension
        # Output expected by X3D is (C, T, H, W)
        video_tensor = torch.stack(frames, dim=1)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)
