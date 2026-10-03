import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioTemporalEncoder(nn.Module):
    """
    Encodes 15s Mel-spectrograms (128 frequency bins x ~469 time steps)
    into 30 frame-aligned acoustic tokens (matching 30 video frames 1-to-1).
    """
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            # (B, 1, 128, T) -> (B, 32, 64, T)
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # (B, 32, 64, T) -> (B, 64, 32, T)
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # (B, 64, 32, T) -> (B, 128, 16, T)
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Collapse frequency dimension: (B, 128, 16, T) -> (B, out_dim, 1, T)
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(128, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.time_pool = nn.AdaptiveAvgPool1d(30)
        
    def forward(self, spec):
        # spec: (B, 1, 128, T_audio)
        x = self.conv(spec).squeeze(2) # (B, out_dim, T_audio)
        x = self.time_pool(x)          # (B, out_dim, 30)
        return x.permute(0, 2, 1)      # (B, 30, out_dim)

class MultiModalTemporalAttention(nn.Module):
    """
    Frame-synchronous multi-head temporal attention that cross-attends
    video frames with acoustic whistle energy at each of the 30 temporal steps.
    """
    def __init__(self, in_video=2048, in_audio=256, num_heads=4, dropout=0.2, modality_dropout=0.15):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_video // num_heads
        self.modality_dropout = modality_dropout
        
        self.audio_proj = nn.ModuleList([
            nn.Linear(in_audio, self.head_dim) for _ in range(num_heads)
        ])
        self.fusion_norm = nn.ModuleList([
            nn.LayerNorm(self.head_dim) for _ in range(num_heads)
        ])
        self.q_proj = nn.ModuleList([
            nn.Linear(self.head_dim, 1, bias=False) for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(in_video, in_video)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, v_tokens, a_tokens=None):
        # v_tokens: (B, 30, 2048), a_tokens: (B, 30, 256)
        B, T, C = v_tokens.shape
        head_outputs = []
        head_weights = []
        
        # Modality dropout during training: prevent over-reliance on audio
        use_audio = (a_tokens is not None)
        if self.training and use_audio and torch.rand(1).item() < self.modality_dropout:
            use_audio = False
            
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = (h + 1) * self.head_dim
            v_h = v_tokens[:, :, start:end] # (B, T, head_dim)
            
            if use_audio:
                a_h = self.audio_proj[h](a_tokens) # (B, T, head_dim)
                fused = self.fusion_norm[h](v_h + a_h)
            else:
                fused = v_h
                
            scores = self.q_proj[h](torch.tanh(fused)) # (B, T, 1)
            weights = F.softmax(scores, dim=1)         # (B, T, 1)
            head_weights.append(weights)
            
            weighted = fused * weights                 # (B, T, head_dim)
            head_outputs.append(weighted.sum(dim=1))   # (B, head_dim)
            
        concat = torch.cat(head_outputs, dim=1)        # (B, 2048)
        out = self.out_proj(self.dropout(concat))
        all_head_weights = torch.stack(head_weights, dim=1).squeeze(-1) # (B, num_heads, 30)
        return out, all_head_weights

class X3DMultiModalModel(nn.Module):
    def __init__(self, num_classes=2, num_heads=4, pretrained=True, dropout=0.2):
        super().__init__()
        
        cache_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_pytorchvideo_main')
        if os.path.exists(cache_dir):
            self.video_backbone = torch.hub.load(cache_dir, 'x3d_m', source='local', pretrained=pretrained)
        else:
            self.video_backbone = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=pretrained)
            
        # Video projection layers from block 5
        orig_head = self.video_backbone.blocks[5]
        self.pre_conv = orig_head.pool.pre_conv
        self.pre_norm = orig_head.pool.pre_norm
        self.pre_act = orig_head.pool.pre_act
        
        # Spatial adaptive pooling preserves all temporal frames (None) across any aspect ratio
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.post_conv = orig_head.pool.post_conv
        self.post_act = orig_head.pool.post_act
        
        # Free unused original block 5 to avoid duplicate parameter references
        self.video_backbone.blocks[5] = nn.Identity()
        
        # Audio temporal encoder
        self.audio_encoder = AudioTemporalEncoder(out_dim=256)
        
        # Multimodal temporal attention
        self.mm_attention = MultiModalTemporalAttention(
            in_video=2048, in_audio=256, num_heads=num_heads, dropout=dropout
        )
        self.classifier = nn.Linear(2048, num_classes)
        
    def freeze_backbone(self):
        """Freeze 3D-CNN video backbone blocks 0..4."""
        for param in self.video_backbone.parameters():
            param.requires_grad = False
                
    def unfreeze_backbone(self):
        """Unfreeze all blocks."""
        for param in self.video_backbone.parameters():
            param.requires_grad = True
                
    def forward(self, video, audio=None, return_head_weights=False):
        # video: (B, 3, 30, H, W), audio: (B, 1, 128, T_audio) or None
        # 1. Video feature extraction through X3D blocks 0..4
        x = video
        for i in range(5):
            x = self.video_backbone.blocks[i](x)
            
        # 2. Project and spatial pool
        x = self.pre_conv(x)
        x = self.pre_norm(x)
        x = self.pre_act(x)
        x = self.spatial_pool(x)       # (B, 432, 30, 1, 1)
        x = self.post_conv(x)          # (B, 2048, 30, 1, 1)
        x = self.post_act(x)
        x = x.squeeze(-1).squeeze(-1)  # (B, 2048, 30)
        v_tokens = x.permute(0, 2, 1)  # (B, 30, 2048)
        
        # 3. Audio feature extraction (if audio provided)
        a_tokens = None
        if audio is not None:
            a_tokens = self.audio_encoder(audio) # (B, 30, 256)
            
        # 4. Synchronized cross-attention fusion
        fused_feat, head_weights = self.mm_attention(v_tokens, a_tokens)
        logits = self.classifier(fused_feat)
        
        if return_head_weights:
            return logits, head_weights
        return logits

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = X3DMultiModalModel(num_classes=2).to(device)
    dummy_vid = torch.randn(2, 3, 30, 224, 398, device=device)
    dummy_aud = torch.randn(2, 1, 128, 469, device=device)
    
    out, hw = model(dummy_vid, dummy_aud, return_head_weights=True)
    print("MultiModal Model Test:")
    print("  Output logits shape:", out.shape)
    print("  Head weights shape:", hw.shape)
    
    out_video_only = model(dummy_vid)
    print("  Video-only fallback shape:", out_video_only.shape)
