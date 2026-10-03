import os
import time
import torch
from torch.utils.data import DataLoader
from dataset_multimodal import MultiModalBinaryDataset
from model_fusion import X3DMultiModalModel
from focal_loss import FocalLoss

def compute_diversity_loss(head_weights):
    # head_weights: (B, num_heads, T)
    B, H, T = head_weights.shape
    norm_w = torch.nn.functional.normalize(head_weights, p=2, dim=-1)
    sim_matrix = torch.bmm(norm_w, norm_w.transpose(1, 2))
    eye = torch.eye(H, device=head_weights.device).unsqueeze(0)
    off_diag = sim_matrix * (1.0 - eye)
    return (off_diag ** 2).sum() / (B * H * (H - 1))

def train_setpiece_multimodal():
    print("=" * 70)
    print(" Training Multimodal SET-PIECE Model (Native 16:9 + Synchronized Audio)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # RTX 4060 VRAM management: Batch size 4, gradient accumulation steps 2
    batch_size = 4
    accum_steps = 2
    num_workers = 4
    
    train_dataset = MultiModalBinaryDataset(
        data_dir='setpiece_dataset_v4/train', is_training=True, target_frames=30, max_bg_ratio=2.5
    )
    val_dataset = MultiModalBinaryDataset(
        data_dir='setpiece_dataset_v4/val', is_training=False, target_frames=30
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    model = X3DMultiModalModel(num_classes=2, num_heads=4, pretrained=True, dropout=0.2).to(device)
    criterion = FocalLoss(alpha=0.65, gamma=2.0)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = -1.0
    save_path = "checkpoints/x3d_setpiece_multimodal_best.pth"
    os.makedirs("checkpoints", exist_ok=True)
    
    epochs = 15
    patience = 4
    epochs_no_improve = 0
    stage2_started = False
    
    # Stage 1: Freeze backbone, train only multimodal head + audio encoder
    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = None
    
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        if epoch <= 3:
            print(f"\n--- Stage 1: Frozen Video Backbone (Epoch {epoch}/{epochs}) ---")
        elif epoch == 4:
            print(f"\n--- Stage 2: Fine-Tuning Full Multimodal Network (Epoch {epoch}/{epochs}) ---")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {'params': list(model.video_backbone.parameters()), 'lr': 1e-5},
                {'params': [p for name, p in model.named_parameters() if not name.startswith('video_backbone.')], 'lr': 1e-4}
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
            stage2_started = True
        else:
            print(f"\n--- Stage 2: Fine-Tuning Full Multimodal Network (Epoch {epoch}/{epochs}) ---")
            
        model.train()
        train_loss = 0.0
        n_train_batches = len(train_loader)
        optimizer.zero_grad()
        
        for batch_idx, (videos, audios, targets) in enumerate(train_loader):
            videos = videos.to(device, non_blocking=True)
            audios = audios.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs, head_weights = model(videos, audios, return_head_weights=True)
                cls_loss = criterion(outputs, targets)
                div_loss = compute_diversity_loss(head_weights)
                loss = (cls_loss + 0.05 * div_loss) / accum_steps
                
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == n_train_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += loss.item() * accum_steps
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_train_batches:
                pct = 100.0 * (batch_idx + 1) / n_train_batches
                print(f"Epoch {epoch} | [{batch_idx + 1}/{n_train_batches} ({pct:.1f}%)] | Train Loss: {loss.item() * accum_steps:.4f}")
                
        if scheduler:
            scheduler.step()
            
        avg_train_loss = train_loss / n_train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = len(val_loader)
        tp, fp, fn, tn = 0, 0, 0, 0
        
        with torch.no_grad():
            for videos, audios, targets in val_loader:
                videos = videos.to(device, non_blocking=True)
                audios = audios.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs, head_weights = model(videos, audios, return_head_weights=True)
                    cls_loss = criterion(outputs, targets)
                    div_loss = compute_diversity_loss(head_weights)
                    loss = cls_loss + 0.05 * div_loss
                    
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
            if stage2_started:
                epochs_no_improve += 1
                print(f"   No improvement in Val F1 for {epochs_no_improve}/{patience} epochs.")
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs. Best Val F1: {best_val_f1:.4f}")
                    break

    print(f"\nMultimodal Set-Piece Training Finished! Best Val F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    train_setpiece_multimodal()
