import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import X3DBinaryDataset
from model import X3DFreeKickModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = "/home/pilot/Desktop/ballsaction/foul_dataset"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    train_dataset = X3DBinaryDataset(data_dir=train_dir, is_training=True)
    val_dataset = X3DBinaryDataset(data_dir=val_dir, is_training=False)
    
    if len(train_dataset) == 0:
        print("Dataset is empty. Run extract_foul_sequences.py first.")
        return
        
    batch_size = 2
    accumulation_steps = 4 # Simulate batch size 8
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = X3DFreeKickModel(num_classes=2, pretrained=True).to(device)
    
    # We extracted 12525 highlights and 34058 background. Ratio is 1:2.72
    class_weights = torch.tensor([1.0, 2.72], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    num_epochs = 15
    best_val_loss = float('inf')
    patience = 4
    epochs_no_improve = 0
    
    os.makedirs("checkpoints", exist_ok=True)
    
    print("Starting X3D Foul Detection training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            running_loss += loss.item() * accumulation_steps
            
            if i % 100 == 99:
                print(f"  Batch {i+1}/{len(train_loader)} | Loss: {running_loss/100:.4f}")
                running_loss = 0.0
                
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        print(f"  Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "checkpoints/x3d_foul_best.pth")
            print("  [*] Best Model Saved!")
        else:
            epochs_no_improve += 1
            print(f"  [!] No improvement for {epochs_no_improve} epoch(s).")
            
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/x3d_foul_epoch_{epoch+1}.pth")
            print(f"  [*] Periodic Checkpoint Saved: x3d_foul_epoch_{epoch+1}.pth")
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement in {patience} epochs.")
            break

if __name__ == "__main__":
    train()
