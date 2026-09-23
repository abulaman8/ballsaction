import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dir = "/home/pilot/Desktop/ballsaction/audio_dataset/val"
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("checkpoints/audio_resnet34_best.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    print("Evaluating Validation Set...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Assuming 'whistle' is class 1 and 'background' is class 0
            # Let's verify class mapping first:
            # val_dataset.class_to_idx is {'background': 0, 'whistle': 1}
            preds = predicted.cpu().numpy()
            targets = labels.numpy()
            
            for p, t in zip(preds, targets):
                if p == 1 and t == 1:
                    tp += 1
                elif p == 1 and t == 0:
                    fp += 1
                elif p == 0 and t == 0:
                    tn += 1
                elif p == 0 and t == 1:
                    fn += 1
                    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
    print("\n=== Audio CNN Evaluation ===")
    print(f"Classes: {val_dataset.class_to_idx}")
    print(f"\nAccuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"[{tn} (TN)] [{fp} (FP)]")
    print(f"[{fn} (FN)] [{tp} (TP)]")

if __name__ == "__main__":
    main()
