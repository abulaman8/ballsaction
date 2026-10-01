import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_audio import AudioDataset
from model_audio import WhistleNet
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_audio_model(model_name, data_dirs, num_epochs=15, batch_size=64):
    print(f"\n========== Training {model_name.upper()} AUDIO Model ==========")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dirs = [os.path.join(d, 'train') for d in data_dirs]
    train_dataset = AudioDataset(data_dirs=train_dirs, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dirs = [os.path.join(d, 'val') for d in data_dirs]
    val_dataset = AudioDataset(data_dirs=val_dirs, is_training=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    labels = [clip["label"] for clip in train_dataset.clips]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"{model_name} Class Weights: {class_weights}")
    
    model = WhistleNet().to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = -1.0
    save_path = f"checkpoints/audio_{model_name}_best.pth"
    os.makedirs("checkpoints", exist_ok=True)
    
    patience = 3 # How many epochs to wait for improvement
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (audios, targets) in enumerate(train_loader):
            audios = audios.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(audios)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                pct = 100.0 * (batch_idx + 1) / len(train_loader)
                print(f"Epoch {epoch+1}/{num_epochs} | [{batch_idx+1}/{len(train_loader)} ({pct:.1f}%)] | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
        
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = 100. * correct / total
        print(f"--- Epoch {epoch+1} Train Summary | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2f}% ---")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0
        
        with torch.no_grad():
            for audios, targets in val_loader:
                audios = audios.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(audios)
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
        
        print(f"--- Epoch {epoch+1} Val Summary | Loss: {avg_val_loss:.4f} | Acc: {accuracy:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f} ---")
        
        # Early stopping and saving based on val_f1
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"==> Saved best audio model at epoch {epoch+1} with Val F1: {best_val_f1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs (best Val F1: {best_val_f1:.4f}).")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs! Best Val F1: {best_val_f1:.4f}")
            break

if __name__ == "__main__":
    train_audio_model("whistle", ["audio_dataset_v2"], num_epochs=15, batch_size=64)

