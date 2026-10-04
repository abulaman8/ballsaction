import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_audio_v2 import WhistleAudioDataset
from model_audio_v2 import WhistleNetV2

def train_whistle_v2():
    print("=" * 75)
    print(" Training Upgraded WhistleNetV2 (ResNet-34 + SE + SpecAugment + Freq-Preserving Stride)")
    print("=" * 75)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    batch_size = 64
    num_workers = 4
    
    train_dataset = WhistleAudioDataset("audio_dataset_v2/train", is_training=True)
    val_dataset = WhistleAudioDataset("audio_dataset_v2/val", is_training=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    model = WhistleNetV2(pretrained=True, num_heads=4).to(device)
    
    # Positive class weighting (~1:3 ratio in training)
    pos_weight = torch.tensor([1.0, 2.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = -1.0
    save_path = "checkpoints/audio_whistle_v2_best.pth"
    os.makedirs("checkpoints", exist_ok=True)
    
    epochs = 8
    patience = 3
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()
        train_loss = 0.0
        n_train_batches = len(train_loader)
        
        for batch_idx, (specs, targets) in enumerate(train_loader):
            specs = specs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(specs)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_train_batches:
                pct = 100.0 * (batch_idx + 1) / n_train_batches
                print(f"Epoch {epoch} | [{batch_idx + 1}/{n_train_batches} ({pct:.1f}%)] | Train Loss: {loss.item():.4f}")
                
        scheduler.step()
        avg_train_loss = train_loss / n_train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = len(val_loader)
        tp, fp, fn, tn = 0, 0, 0, 0
        
        with torch.no_grad():
            for specs, targets in val_loader:
                specs = specs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(specs)
                    loss = criterion(outputs, targets)
                    
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                
                for p, t in zip(preds, targets):
                    if p == 1 and t == 1: tp += 1
                    elif p == 1 and t == 0: fp += 1
                    elif p == 0 and t == 1: fn += 1
                    else: tn += 1
                    
        avg_val_loss = val_loss / n_val_batches
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        epoch_sec = time.time() - t_epoch_start
        
        print(f"\n>> Epoch {epoch} Evaluation ({epoch_sec:.1f}s):")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   Val Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
        print(f"   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': best_val_f1,
                'val_precision': precision,
                'val_recall': recall
            }, save_path)
            print(f"   *** NEW BEST VAL F1: {best_val_f1:.4f} -> Saved to {save_path} ***")
        else:
            epochs_no_improve += 1
            print(f"   No improvement in Val F1 for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs. Best Val F1: {best_val_f1:.4f}")
                break

    print(f"\nWhistleNetV2 Training Finished! Best Val F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    train_whistle_v2()
