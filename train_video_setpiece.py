import os
import time
import torch
from torch.utils.data import DataLoader
from dataset import X3DBinaryDataset
from model import X3DFreeKickModel
from focal_loss import FocalLoss

def train_setpiece_model():
    print("========== Training SET-PIECE Video Model ==========")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = X3DBinaryDataset(data_dir='setpiece_dataset_v2/train', is_training=True)
    val_dataset = X3DBinaryDataset(data_dir='setpiece_dataset_v2/val', is_training=False)
    
    batch_size = 8
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    model = X3DFreeKickModel().to(device)
    criterion = FocalLoss(alpha=0.65, gamma=2.0)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = -1.0
    save_path = "checkpoints/x3d_setpiece_best.pth"
    os.makedirs("checkpoints", exist_ok=True)
    
    epochs = 15
    patience = 3
    epochs_no_improve = 0
    stage2_started = False
    
    # Stage 1: Freeze backbone, train only head
    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = None
    
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        if epoch <= 3:
            print(f"\n--- Stage 1: Frozen Backbone (Epoch {epoch}/{epochs}) ---")
        elif epoch == 4:
            print(f"\n--- Stage 2: Fine-Tuning Backbone (Epoch {epoch}/{epochs}) ---")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {'params': [p for i in range(5) for p in model.model.blocks[i].parameters()], 'lr': 1e-5},
                {'params': model.model.blocks[5].parameters(), 'lr': 1e-4}
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
            stage2_started = True
        else:
            print(f"\n--- Stage 2: Fine-Tuning Backbone (Epoch {epoch}/{epochs}) ---")
        
        model.train()
        train_loss = 0.0
        n_train_batches = len(train_loader)
        
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
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_train_batches:
                pct = 100.0 * (batch_idx + 1) / n_train_batches
                print(f"Epoch {epoch} | [{batch_idx + 1}/{n_train_batches} ({pct:.1f}%)] | Train Loss: {loss.item():.4f}")
                
        if scheduler:
            scheduler.step()
            
        avg_train_loss = train_loss / n_train_batches
        
        # Validation
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
        precision = val_tp / (val_tp + val_fp + 1e-8)
        recall = val_tp / (val_tp + val_fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = (val_tp + val_tn) / (val_tp + val_fp + val_fn + val_tn + 1e-8)
        elapsed = time.time() - t_epoch_start
        
        print(f"--- Epoch {epoch} Val Summary ({elapsed:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {accuracy:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f} ---")
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"==> Saved best setpiece model at epoch {epoch} with Val F1: {best_val_f1:.4f}")
            if stage2_started:
                epochs_no_improve = 0
        else:
            if stage2_started:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs (best Val F1: {best_val_f1:.4f}).")
                
        if stage2_started and epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs!")
            break

    print(f"Training completed. Best Val F1: {best_val_f1:.4f}. Saved checkpoint: {save_path}")

if __name__ == "__main__":
    train_setpiece_model()
