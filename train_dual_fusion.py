import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_fusion import DualFusionDataset
from model import X3DFreeKickModel
import torchvision.models as models

class DualFusionModel(nn.Module):
    def __init__(self, x3d_ckpt_path):
        super().__init__()
        # Video branch (Frozen)
        self.video_model = X3DFreeKickModel(num_classes=2, pretrained=False)
        self.video_model.load_state_dict(torch.load(x3d_ckpt_path, map_location="cpu", weights_only=True))
        self.video_model.model.blocks[5].proj = nn.Identity()
        for param in self.video_model.parameters():
            param.requires_grad = False
            
        # Audio branch (Trainable)
        self.audio_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Modify first conv to accept 1 channel (spectrogram)
        self.audio_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.audio_model.fc = nn.Identity() # outputs 512
        
        # Fusion Head
        self.fusion_fc = nn.Sequential(
            nn.Linear(2048 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
    def forward(self, video, audio):
        with torch.no_grad():
            v_feat = self.video_model(video) # (B, 2048)
        a_feat = self.audio_model(audio) # (B, 1, 128, 256)
        fused = torch.cat((v_feat, a_feat), dim=1)
        out = self.fusion_fc(fused)
        return out

def train_fusion(model_name, data_dir, x3d_ckpt_path, num_epochs=5, batch_size=2):
    print(f"\n========== Training {model_name.upper()} FUSION Model ==========")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = DualFusionDataset(os.path.join(data_dir, "train"))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = DualFusionModel(x3d_ckpt_path).to(device)
    
    # Calculate class weights for imbalanced data
    labels = [info["label"] for info in train_dataset.clips]
    total = len(labels)
    pos = sum(labels)
    neg = total - pos
    weight_neg = total / (2 * neg)
    weight_pos = total / (2 * pos)
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(device)
    
    print(f"{model_name} Fusion Class Weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Only train audio model and fusion head
    optimizer = torch.optim.Adam([
        {'params': model.audio_model.parameters(), 'lr': 1e-4},
        {'params': model.fusion_fc.parameters(), 'lr': 1e-4}
    ])
    
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (videos, audios, targets) in enumerate(train_loader):
            videos, audios, targets = videos.to(device), audios.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos, audios)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == targets).float().mean() * 100
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")
                
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f"checkpoints/fusion_{model_name}_best.pth")
            print(f"Saved new best model for {model_name} with loss {best_loss:.4f}")

if __name__ == "__main__":
    train_fusion("foul", "foul_dataset", "checkpoints/x3d_foul_best.pth", num_epochs=5, batch_size=2)
    train_fusion("setpiece", "setpiece_dataset", "checkpoints/x3d_setpiece_best.pth", num_epochs=5, batch_size=2)
