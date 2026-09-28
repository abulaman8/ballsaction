import os
import cv2
cv2.setNumThreads(0)
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class X3DBinaryDataset(Dataset):
    def __init__(self, data_dir, is_training=True):
        self.data_dir = data_dir
        self.is_training = is_training
        
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
                    
        print(f"Dataset initialized with {len(self.clips)} valid mp4 clips from {data_dir}.")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        video_path = clip_info["path"]
        label = clip_info["label"]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < 40:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))
            
        cap.release()
        
        while len(frames) < 40:
            if len(frames) > 0:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros((3, 224, 224)))
                
        video_tensor = torch.stack(frames, dim=1)
        return video_tensor, torch.tensor(label, dtype=torch.long)
