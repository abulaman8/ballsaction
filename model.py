import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Linear(in_channels, 1)

    def forward(self, x):
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        weighted = x * weights
        out = weighted.sum(dim=1)
        return out, weights

class X3DAttentionHead(nn.Module):
    def __init__(self, original_head, num_classes):
        super().__init__()
        self.pool = original_head.pool
        self.dropout = original_head.dropout
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # We need to get the in_features from original_head.proj
        in_features = original_head.proj.in_features
        self.temporal_attention = TemporalAttention(in_channels=in_features)
        self.proj = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.dropout(x)
        x = self.spatial_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = x.permute(0, 2, 1)
        out, weights = self.temporal_attention(x)
        self.last_weights = weights.detach().cpu().numpy()
        out = self.proj(out)
        return out

class X3DFreeKickModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(X3DFreeKickModel, self).__init__()
        
        import os
        cache_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_pytorchvideo_main')
        if os.path.exists(cache_dir):
            self.model = torch.hub.load(cache_dir, 'x3d_m', source='local', pretrained=pretrained)
        else:
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=pretrained)
        
        # Replace the final block with our custom Attention Head
        original_head = self.model.blocks[5]
        self.model.blocks[5] = X3DAttentionHead(original_head, num_classes)
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = X3DFreeKickModel()
    dummy_input = torch.randn(2, 3, 40, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)
