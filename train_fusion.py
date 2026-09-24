import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

class FusionDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.tensor([item["video_prob"], item["audio_prob"]], dtype=torch.float32)
        y = torch.tensor(item["label"], dtype=torch.float32)
        return x, y

class FusionNetwork(nn.Module):
    def __init__(self):
        super(FusionNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze(1)

def main():
    print("Loading dataset...")
    with open("fusion_dataset.json", "r") as f:
        data = json.load(f)
        
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Artificially imbalance the dataset to heavily penalize false positives
    # 2 Backgrounds for every 1 Foul
    train_bg = [x for x in train_data if x["label"] == 0]
    train_foul = [x for x in train_data if x["label"] == 1]
    imbalanced_train = train_foul + (train_bg * 2)
    
    train_loader = DataLoader(FusionDataset(imbalanced_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(FusionDataset(val_data), batch_size=32)
    
    model = FusionNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 20
    best_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                val_loss += criterion(out, y).item()
                preds = (out > 0.5).float()
                all_preds.extend(preds.numpy())
                all_targets.extend(y.numpy())
                
        prec = precision_score(all_targets, all_preds, zero_division=0)
        rec = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val F1: {f1*100:.2f}% (P: {prec*100:.1f}%, R: {rec*100:.1f}%)")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "checkpoints/fusion_mlp_best.pth")
            print("  [!] Saved new champion fusion network!")

if __name__ == "__main__":
    main()
