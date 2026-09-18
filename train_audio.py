import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    # Standard ResNet image transformations
    # ResNet expects 224x224 RGB images with this exact normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_dir = "/home/pilot/Desktop/ballsaction/audio_dataset"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Dataset not found! Run extract_audio_spectrograms.py first.")
        return
        
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    print(f"Loaded {len(train_dataset)} training spectrograms and {len(val_dataset)} validation spectrograms.")
    print(f"Class mapping: {train_dataset.class_to_idx}")
    
    # Initialize Pretrained ResNet-18
    # We use a pre-trained model because the early convolution layers are already perfect edge/line detectors
    print("Loading Pretrained ResNet-18...")
    model = models.resnet18(pretrained=True)
    
    # Modify the final Fully Connected (fc) layer for our binary task (0: Background, 1: Whistle)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Early Stopping tracking
    best_val_loss = float('inf')
    patience = 4
    epochs_no_improve = 0
    max_epochs = 15
    
    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = "checkpoints/audio_resnet18_best.pth"
    
    print("\nStarting Training Loop...")
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{max_epochs}] Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        epoch_time = time.time() - start_time
        
        print(f"\n--- Epoch {epoch+1} Summary ({epoch_time:.0f}s) ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved new champion model to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"--> No improvement. Patience: {epochs_no_improve}/{patience}")
            
        if epochs_no_improve >= patience:
            print("\n[Early Stopping Triggered] Validation loss has stopped improving.")
            break
            
    print("\nTraining Complete! Best model saved at:", best_model_path)

if __name__ == "__main__":
    main()
