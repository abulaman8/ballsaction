import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import X3DBinaryDataset
from model import X3DFreeKickModel

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    val_dir = "/home/pilot/Desktop/ballsaction/x3d_dataset/val"
    val_dataset = X3DBinaryDataset(data_dir=val_dir, is_training=False)
    
    if len(val_dataset) == 0:
        print("Dataset is empty.")
        return
        
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    model = X3DFreeKickModel(num_classes=2, pretrained=False)
    ckpt_path = "checkpoints/x3d_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    
    print("Evaluating X3D Model on Validation Set...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    # Calculate Metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    acc = np.mean(y_true == y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    print("\n" + "="*50)
    print("--- EVALUATION COMPLETE ---")
    print(f"Overall Accuracy (Set Piece vs Background): {acc * 100:.2f}%")
    print("="*50)
    
    print("\nClass Index Mapping: {0: 'background', 1: 'set_piece'}")
    
    print("\n--- Highlight Detection Matrix (Target: set_piece) ---")
    print(f"True Positives (Correct Highlights)  : {tp}")
    print(f"True Negatives (Correct Background)  : {tn}")
    print(f"False Positives (False Highlights)   : {fp}")
    print(f"False Negatives (Missed Highlights)  : {fn}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print("\n--- Advanced Metrics ---")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")

if __name__ == "__main__":
    evaluate()
