import os
import cv2
cv2.setNumThreads(0)
import json
import time
import shutil
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model import X3DFreeKickModel
from focal_loss import FocalLoss

class HardMinedFoulDataset(Dataset):
    def __init__(self, foul_dir, bg_dir, hard_negatives_json, num_hard_negs=5000, num_random_bgs=5000, is_training=True):
        self.is_training = is_training
        self.clips = []
        
        # 1. Load All Positives
        pos_files = [os.path.join(foul_dir, f) for f in os.listdir(foul_dir) if f.endswith(".mp4")]
        for p in pos_files:
            self.clips.append({"path": p, "label": 1, "type": "positive"})
            
        print(f"Loaded {len(pos_files)} positive foul clips.")
        
        # 2. Load All Mined Hard Negatives with 2x weighting for severe false alarms
        with open(hard_negatives_json, "r") as f:
            hard_data = json.load(f)
            
        hard_paths = set()
        n_severe = 0
        for item in hard_data:
            self.clips.append({"path": item["path"], "label": 0, "type": "hard_negative"})
            hard_paths.add(item["path"])
            # Double-sample the severe false alarms (P >= 0.60) so the model sees them frequently
            if item.get("foul_prob", 0) >= 0.60:
                self.clips.append({"path": item["path"], "label": 0, "type": "critical_hard_negative"})
                n_severe += 1
                
        print(f"Loaded {len(hard_data)} unique hard negatives (+ {n_severe} severe false alarms oversampled 2x).")
        
        # 3. Load Regular Background Clips (excluding already selected hard negatives)
        all_bg_files = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith(".mp4")]
        remaining_bgs = [f for f in all_bg_files if f not in hard_paths]
        sampled_bgs = random.sample(remaining_bgs, min(num_random_bgs, len(remaining_bgs)))
        for p in sampled_bgs:
            self.clips.append({"path": p, "label": 0, "type": "random_background"})
            
        print(f"Loaded {len(sampled_bgs)} random regular background clips.")
        print(f"Total Hard-Mined Training Dataset Size: {len(self.clips)} clips.")
        
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
        
        do_flip = self.is_training and torch.rand(1).item() < 0.5
        if self.is_training:
            skip_frames = int(torch.randint(0, 3, (1,)).item())
            for _ in range(skip_frames):
                ret, _ = cap.read()
                if not ret: break
                
        while len(frames) < 20:
            ret, frame = cap.read()
            if not ret: break
            if do_flip: frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_frame = self.to_tensor(frame)
            t_frame = self.resize(t_frame)
            if self.is_training: t_frame = self.jitter(t_frame)
            t_frame = self.normalize(t_frame)
            frames.append(t_frame)
            
        cap.release()
        
        while len(frames) < 20:
            if len(frames) > 0: frames.append(frames[-1].clone())
            else: frames.append(torch.zeros((3, 224, 224)))
            
        video_tensor = torch.stack(frames, dim=1)
        return video_tensor, torch.tensor(label, dtype=torch.long)

def train_hard_mined_foul():
    print("========== Fine-Tuning FOUL Model with Hard Negative Mining ==========")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Dataset setup
    train_dataset = HardMinedFoulDataset(
        foul_dir='foul_dataset_v2/train/foul',
        bg_dir='foul_dataset_v2/train/background',
        hard_negatives_json='hard_negatives_foul.json',
        num_hard_negs=6000,
        num_random_bgs=6000,
        is_training=True
    )
    
    from dataset import X3DBinaryDataset
    val_dataset = X3DBinaryDataset(data_dir='foul_dataset_v2/val', is_training=False)
    
    batch_size = 8
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # 2. Model setup: Load the best multi-head foul model
    model = X3DFreeKickModel(num_classes=2, num_heads=4, pretrained=False).to(device)
    base_ckpt = "checkpoints/x3d_foul_best.pth"
    backup_ckpt = "checkpoints/x3d_foul_pre_mining.pth"
    
    if not os.path.exists(backup_ckpt) and os.path.exists(base_ckpt):
        shutil.copyfile(base_ckpt, backup_ckpt)
        print(f"Backed up pre-mining checkpoint to: {backup_ckpt}")
        
    model.load_state_dict(torch.load(base_ckpt, map_location=device))
    print(f"Loaded baseline checkpoint: {base_ckpt}")
    
    criterion = FocalLoss(alpha=0.65, gamma=2.0)
    scaler = torch.amp.GradScaler('cuda')
    
    # Low learning rate for gentle fine-tuning
    optimizer = torch.optim.AdamW([
        {'params': [p for i in range(5) for p in model.model.blocks[i].parameters()], 'lr': 5e-6},
        {'params': model.model.blocks[5].parameters(), 'lr': 3e-5}
    ], weight_decay=1e-4)
    
    epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_f1 = 0.7040 # Start from baseline best
    save_path = "checkpoints/x3d_foul_best.pth"
    
    for epoch in range(1, epochs + 1):
        t_start = time.time()
        model.train()
        train_loss = 0.0
        n_train = len(train_loader)
        
        for batch_idx, (videos, targets) in enumerate(train_loader):
            videos = videos.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(videos)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_train:
                pct = 100.0 * (batch_idx + 1) / n_train
                print(f"Fine-Tune Epoch {epoch}/{epochs} | [{batch_idx+1}/{n_train} ({pct:.1f}%)] | Loss: {loss.item():.4f}")
                
        scheduler.step()
        avg_train_loss = train_loss / n_train
        
        # Validation on full held-out validation set
        model.eval()
        val_loss = 0.0
        val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0
        
        with torch.no_grad():
            for videos, targets in val_loader:
                videos = videos.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(videos)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_tp += ((preds == 1) & (targets == 1)).sum().item()
                val_fp += ((preds == 1) & (targets == 0)).sum().item()
                val_fn += ((preds == 0) & (targets == 1)).sum().item()
                val_tn += ((preds == 0) & (targets == 0)).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        prec = val_tp / (val_tp + val_fp + 1e-8)
        rec = val_tp / (val_tp + val_fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        acc = (val_tp + val_tn) / (val_tp + val_fp + val_fn + val_tn + 1e-8)
        elapsed = time.time() - t_start
        
        print(f"--- Epoch {epoch} Val Summary ({elapsed:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f} ---")
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"==> Improved Foul Model! Saved new best checkpoint at epoch {epoch} with Val F1: {best_val_f1:.4f}")
        else:
            print(f"Val F1 ({f1:.4f}) did not beat best ({best_val_f1:.4f}).")

    print(f"\nFine-tuning complete. Best Val F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    train_hard_mined_foul()
