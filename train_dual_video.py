import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import X3DBinaryDataset
from model import X3DFreeKickModel
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_model(model_name, data_dir, num_epochs=15, batch_size=4):
    print(f"========== Training {model_name.upper()} Model ==========")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    train_dataset = X3DBinaryDataset(data_dir=os.path.join(data_dir, 'train'), is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Calculate Class Weights to handle imbalances
    labels = [clip["label"] for clip in train_dataset.clips]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"{model_name} Class Weights: {class_weights}")
    
    model = X3DFreeKickModel(num_classes=2, pretrained=True).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_loss = float('inf')
    save_path = f"checkpoints/x3d_{model_name}_best.pth"
    os.makedirs("checkpoints", exist_ok=True)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (videos, targets) in enumerate(train_loader):
            videos, targets = videos.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
        
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = 100. * correct / total
        print(f"--- Epoch {epoch+1} Summary | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2f}% ---")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

    print(f"Finished training {model_name}. Best Loss: {best_loss:.4f}\n")

if __name__ == "__main__":
    # Train the Foul Model
    # train_model(model_name="foul", data_dir="foul_dataset", num_epochs=5, batch_size=2)
    
    # Train the Set-Piece Model
    train_model(model_name="setpiece", data_dir="setpiece_dataset", num_epochs=5, batch_size=2)
