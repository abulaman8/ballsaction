import torch
import torch.nn as nn

class X3DFreeKickModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(X3DFreeKickModel, self).__init__()
        
        # Load X3D Medium from PyTorchVideo local cache
        import os
        cache_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_pytorchvideo_main')
        if os.path.exists(cache_dir):
            self.model = torch.hub.load(cache_dir, 'x3d_m', source='local', pretrained=pretrained)
        else:
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=pretrained)
        
        in_features = self.model.blocks[5].proj.in_features
        self.model.blocks[5].proj = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x expected shape: (B, C, T, H, W)
        return self.model(x)

if __name__ == "__main__":
    model = X3DFreeKickModel()
    dummy_input = torch.randn(2, 3, 40, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)
