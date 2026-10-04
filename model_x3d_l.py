import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head temporal attention over 30 discrete 0.5s temporal tokens.
    Operates over un-blurred spatial pooled tokens with diversity loss.
    """
    def __init__(self, in_channels=2048, num_heads=4, hidden_channels=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"
        
        head_hidden = hidden_channels // num_heads
        
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.head_dim, head_hidden, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv1d(head_hidden, 1, kernel_size=1)
            )
            for _ in range(num_heads)
        ])
        
        self.out_proj = nn.Linear(in_channels, in_channels)
    
    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        x_heads = x.view(B, T, self.num_heads, self.head_dim)
        
        head_outputs = []
        head_weights = []
        
        for i in range(self.num_heads):
            x_h = x_heads[:, :, i, :]          # (B, T, head_dim)
            x_t = x_h.permute(0, 2, 1)         # (B, head_dim, T)
            scores = self.attention_heads[i](x_t).permute(0, 2, 1) # (B, T, 1)
            weights = F.softmax(scores, dim=1)
            head_weights.append(weights)
            
            weighted = x_h * weights            # (B, T, head_dim)
            head_outputs.append(weighted.sum(dim=1)) # (B, head_dim)
        
        concat = torch.cat(head_outputs, dim=1) # (B, C)
        out = self.out_proj(concat)
        
        avg_weights = torch.stack(head_weights, dim=0).mean(dim=0) # (B, T, 1)
        all_head_weights = torch.stack(head_weights, dim=1).squeeze(-1) # (B, num_heads, T)
        
        return out, avg_weights, all_head_weights

class X3DLAttentionHead(nn.Module):
    """
    Replaces original x3d_l block 5 with:
    1. Pretrained pre_conv + pre_norm (192 -> 432 channels)
    2. Adaptive spatial pooling (preserves 30 discrete time tokens, collapses H and W)
    3. Pretrained post_conv (432 -> 2048 channels)
    4. 4-Head Temporal Attention pooling with diversity penalty
    5. Final classification projection
    """
    def __init__(self, original_head, num_classes=2, num_heads=4, dropout=0.5):
        super().__init__()
        self.pre_conv = nn.Conv3d(192, 432, kernel_size=1, bias=False)
        self.pre_norm = nn.BatchNorm3d(432)
        self.pre_act = nn.ReLU(inplace=True)
        
        # Copy pretrained weights from original Kinetics head
        if hasattr(original_head, 'pool') and hasattr(original_head.pool, 'pre_conv'):
            self.pre_conv.weight.data.copy_(original_head.pool.pre_conv.weight.data)
            self.pre_norm.load_state_dict(original_head.pool.pre_norm.state_dict())
            
        # Native widescreen adaptive spatial pool: preserves T, collapses H and W to 1
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        self.post_conv = nn.Conv3d(432, 2048, kernel_size=1, bias=False)
        self.post_act = nn.ReLU(inplace=True)
        if hasattr(original_head, 'pool') and hasattr(original_head.pool, 'post_conv'):
            self.post_conv.weight.data.copy_(original_head.pool.post_conv.weight.data)
            
        self.dropout = nn.Dropout(p=dropout)
        self.temporal_attention = MultiHeadTemporalAttention(in_channels=2048, num_heads=num_heads)
        self.proj_attn = nn.Linear(2048, num_classes)
        
        self.last_weights = None
        self.last_head_weights = None

    def forward(self, x, return_head_weights=False):
        # x: (B, 192, 30, H', W')
        x = self.pre_conv(x)
        x = self.pre_norm(x)
        x = self.pre_act(x)
        
        x = self.spatial_pool(x) # (B, 432, 30, 1, 1)
        x = self.post_conv(x)    # (B, 2048, 30, 1, 1)
        x = self.post_act(x)
        x = self.dropout(x)
        
        # (B, 30, 2048)
        tokens = x.squeeze(-1).squeeze(-1).transpose(1, 2)
        
        out_feat, weights, head_weights = self.temporal_attention(tokens)
        self.last_weights = weights.detach().cpu().numpy()
        self.last_head_weights = head_weights.detach().cpu().numpy()
        
        logits = self.proj_attn(out_feat)
        if return_head_weights:
            return logits, head_weights
        return logits

class X3DLVideoModel(nn.Module):
    """
    High-capacity X3D-L action spotter for native 16:9 widescreen video.
    Backbone: 6.15M parameters with deeper residual stages.
    Temporal Attention: 4 heads over 30 discrete 0.5s tokens.
    """
    def __init__(self, num_classes=2, num_heads=4, pretrained=True, dropout=0.5):
        super().__init__()
        cache_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_pytorchvideo_main')
        if os.path.exists(cache_dir):
            base_model = torch.hub.load(cache_dir, 'x3d_l', source='local', pretrained=pretrained)
        else:
            base_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_l', pretrained=pretrained)
            
        self.blocks = nn.ModuleList([base_model.blocks[i] for i in range(5)])
        orig_head = base_model.blocks[5]
        self.head = X3DLAttentionHead(orig_head, num_classes=num_classes, num_heads=num_heads, dropout=dropout)

    def freeze_backbone(self):
        """Freeze blocks 0-4 (video backbone), leaving head trainable."""
        for b in self.blocks:
            for param in b.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all blocks for end-to-end fine-tuning."""
        for b in self.blocks:
            for param in b.parameters():
                param.requires_grad = True

    def forward(self, x, return_head_weights=False):
        # x: (B, 3, 30, 224, 398)
        feat = x
        for b in self.blocks:
            feat = b(feat)
        return self.head(feat, return_head_weights=return_head_weights)

if __name__ == "__main__":
    model = X3DLVideoModel(num_classes=2, num_heads=4, pretrained=False)
    dummy = torch.randn(2, 3, 30, 224, 398)
    out, hw = model(dummy, return_head_weights=True)
    print("X3D-L Model test passed!")
    print("Output logits shape:", out.shape)
    print("Head weights shape:", hw.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params / 1e6:.2f}M")
