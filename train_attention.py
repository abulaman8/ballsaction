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
    
    model = X3DFreeKickModel(num_classes=2, pretrained=True)
    if os.path.exists("checkpoints/x3d_foul_best.pth"):
        print("Loading current champion weights for fine-tuning...")
        old_state = torch.load("checkpoints/x3d_foul_best.pth", map_location=device)
        missing, unexpected = model.load_state_dict(old_state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    
    # Freeze the backbone (blocks 0 to 4)
    for i in range(5):
        for param in model.model.blocks[i].parameters():
            param.requires_grad = False
            
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
            
    model = model.to(device)
    
    num_hl = len(os.listdir(os.path.join(train_dir, "set_piece")))
    num_bg = len(os.listdir(os.path.join(train_dir, "background")))
    ratio = num_bg / num_hl if num_hl > 0 else 1.0
    print(f"Highlights: {num_hl}, Backgrounds: {num_bg} -> Ratio: 1:{ratio:.2f}")
    
    class_weights = torch.tensor([1.0, ratio], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    num_epochs = 15
    best_val_loss = float('inf')
    patience = 4
    epochs_no_improve = 0
    
    os.makedirs("checkpoints", exist_ok=True)
    
    print("Starting X3D Attention training...")
    
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
            torch.save(model.state_dict(), "checkpoints/x3d_attention_best.pth")
            print("  [*] Best Model Saved!")
        else:
            epochs_no_improve += 1
            print(f"  [!] No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement in {patience} epochs.")
            break

if __name__ == "__main__":
    train()
