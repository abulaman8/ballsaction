import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Models channel-wise interdependencies, amplifying whistle-sensitive frequencies
    and suppressing background crowd rumble.
    """
    def __init__(self, channels=512, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w

class MultiHeadAudioTemporalAttention(nn.Module):
    """
    Multi-Head Temporal Attention for Spectrograms.
    Identifies transient whistle spikes while suppressing surrounding dead air.
    """
    def __init__(self, in_channels=512, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.attn_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.head_dim, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 1, kernel_size=1)
            ) for _ in range(num_heads)
        ])
        
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x: (B, T, C)
        b, t, c = x.shape
        x_perm = x.permute(0, 2, 1) # (B, C, T)
        
        head_outputs = []
        head_weights = []
        
        for i in range(self.num_heads):
            head_x = x_perm[:, i*self.head_dim:(i+1)*self.head_dim, :] # (B, head_dim, T)
            score = self.attn_heads[i](head_x) # (B, 1, T)
            weight = torch.softmax(score, dim=-1) # (B, 1, T)
            head_weights.append(weight)
            out_i = torch.bmm(head_x, weight.permute(0, 2, 1)).squeeze(-1) # (B, head_dim)
            head_outputs.append(out_i)
            
        concat_out = torch.cat(head_outputs, dim=-1) # (B, C)
        proj_out = self.dropout(self.out_proj(concat_out))
        
        mean_feat = x.mean(dim=1)
        final_out = self.norm(proj_out + mean_feat)
        avg_weights = torch.stack(head_weights, dim=1).mean(dim=1).squeeze(1) # (B, T)
        return final_out, avg_weights

class WhistleNetV2(nn.Module):
    """
    Upgraded Audio Specialist for Whistle Detection.
    - Backbone: ResNet-34 (21M parameters) with frequency-preserving stride (1, 2) in conv1.
    - Squeeze-and-Excitation channel recalibration on Layer 4.
    - Multi-Head Temporal Attention pooling across arbitrary-duration audio streams.
    """
    def __init__(self, pretrained=True, num_heads=4):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)
        
        # Stride (1, 2): preserves full 128 Mel frequency bins while downsampling time
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 2), padding=(3, 3), bias=False)
        if pretrained:
            self.conv1.weight.data.copy_(resnet.conv1.weight.data.mean(dim=1, keepdim=True))
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.se = SEBlock(channels=512, reduction=16)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None)) # Collapses frequency to 1, preserves time
        self.temporal_attn = MultiHeadAudioTemporalAttention(in_channels=512, num_heads=num_heads)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        self.last_weights = None

    def forward(self, x, return_weights=False):
        # x: (B, 1, 128, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # (B, 512, H_f, T')
        x = self.se(x)
        
        # Pool frequency to 1, keep time T'
        x = self.freq_pool(x).squeeze(2).transpose(1, 2) # (B, T', 512)
        
        feat, weights = self.temporal_attn(x)
        self.last_weights = weights.detach().cpu().numpy()
        
        logits = self.fc(feat)
        if return_weights:
            return logits, weights
        return logits

if __name__ == "__main__":
    model = WhistleNetV2(pretrained=False)
    dummy_10s = torch.randn(2, 1, 128, 313)
    dummy_15s = torch.randn(2, 1, 128, 469)
    out10 = model(dummy_10s)
    out15 = model(dummy_15s)
    print("WhistleNetV2 test passed!")
    print("10s output shape:", out10.shape)
    print("15s output shape:", out15.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")
