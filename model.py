import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadTemporalAttention(nn.Module):
    """Multi-head temporal attention over temporal chunks.
    
    Each head independently attends over its slice of the feature dimension,
    allowing different heads to specialize on different temporal patterns
    (e.g., build-up, contact moment, aftermath, referee reaction).
    """
    def __init__(self, in_channels, num_heads=4, hidden_channels=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, \
            f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"
        
        head_hidden = hidden_channels // num_heads
        
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.head_dim, head_hidden, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv1d(head_hidden, 1, kernel_size=1)
            )
            for _ in range(num_heads)
        ])
        
        # Output projection to mix information across heads
        self.out_proj = nn.Linear(in_channels, in_channels)
    
    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        
        # Split channels across heads: (B, T, num_heads, head_dim)
        x_heads = x.view(B, T, self.num_heads, self.head_dim)
        
        head_outputs = []
        head_weights = []
        
        for i in range(self.num_heads):
            x_h = x_heads[:, :, i, :]          # (B, T, head_dim)
            x_t = x_h.permute(0, 2, 1)         # (B, head_dim, T) for Conv1d
            scores = self.attention_heads[i](x_t)  # (B, 1, T)
            scores = scores.permute(0, 2, 1)   # (B, T, 1)
            weights = F.softmax(scores, dim=1)
            head_weights.append(weights)
            
            weighted = x_h * weights            # (B, T, head_dim)
            out = weighted.sum(dim=1)           # (B, head_dim)
            head_outputs.append(out)
        
        # Concatenate all heads and project: (B, C) -> (B, C)
        concat = torch.cat(head_outputs, dim=1)
        out = self.out_proj(concat)
        
        # Average weights across heads for visualization: (B, T, 1)
        avg_weights = torch.stack(head_weights, dim=0).mean(dim=0)
        
        return out, avg_weights

class X3DAttentionHead(nn.Module):
    def __init__(self, original_head, num_classes, num_heads=4):
        super().__init__()
        self.pool = original_head.pool
        self.dropout = original_head.dropout
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # We need to get the in_features from original_head.proj
        in_features = original_head.proj.in_features
        self.temporal_attention = MultiHeadTemporalAttention(
            in_channels=in_features, num_heads=num_heads
        )
        
        # Single projection head
        self.proj_attn = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.dropout(x)
        x = self.spatial_pool(x)
        x = x.squeeze(-1).squeeze(-1) # (B, C, T)
        
        # Multi-Head Temporal Attention Pooling
        x_perm = x.permute(0, 2, 1) # (B, T, C)
        out_attn_feat, weights = self.temporal_attention(x_perm)
        self.last_weights = weights.detach().cpu().numpy()
        out_attn = self.proj_attn(out_attn_feat)
        
        return out_attn

class X3DFreeKickModel(nn.Module):
    def __init__(self, num_classes=2, num_heads=4, pretrained=True):
        super(X3DFreeKickModel, self).__init__()
        
        import os
        cache_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_pytorchvideo_main')
        if os.path.exists(cache_dir):
            self.model = torch.hub.load(cache_dir, 'x3d_m', source='local', pretrained=pretrained)
        else:
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=pretrained)
        
        # Replace the final block with our custom Multi-Head Attention Head
        original_head = self.model.blocks[5]
        self.model.blocks[5] = X3DAttentionHead(original_head, num_classes, num_heads=num_heads)

    def freeze_backbone(self):
        """Freeze all blocks except the custom head (blocks[5])."""
        for i in range(5):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all blocks."""
        for param in self.model.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = X3DFreeKickModel(num_heads=4)
    dummy_input = torch.randn(2, 3, 20, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)
    print("Attention weights shape:", model.model.blocks[5].last_weights.shape)
