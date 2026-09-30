import torch
import torch.nn as nn
import torchvision.models as models

class WhistleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use ResNet18 as the backbone for spectrograms
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Modify the first conv layer to accept 1-channel grayscale spectrograms
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Replace the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5), # Add dropout to prevent instant overfitting
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.model(x)
