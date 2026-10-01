import torch
import torch.nn as nn
import torchvision.models as models

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Recalibrates feature map channels by modeling channel-wise interdependencies,
    amplifying whistle-sensitive frequencies and suppressing background crowd noise.
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
    Evaluates 4 independent attention heads across the temporal frames (10 seconds),
    pinpointing the transient whistle spike while suppressing the surrounding seconds.
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
        
        # Residual skip connection to global average feature
        mean_feat = x.mean(dim=1)
        final_out = self.norm(proj_out + mean_feat)
        
        avg_weights = torch.stack(head_weights, dim=1).mean(dim=1).squeeze(1) # (B, T)
        return final_out, avg_weights

class WhistleNet(nn.Module):
    def __init__(self, use_attention=True, num_heads=4):
        super().__init__()
        self.use_attention = use_attention
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Modify the first conv layer to accept 1-channel grayscale spectrograms
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        if self.use_attention:
            self.se = SEBlock(channels=512, reduction=16)
            self.freq_pool = nn.AdaptiveAvgPool2d((1, None)) # Pool freq to 1, preserve time T
            self.temporal_attn = MultiHeadAudioTemporalAttention(in_channels=512, num_heads=num_heads)
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 2)
            )
            self.last_weights = None
        else:
            self.avgpool = resnet.avgpool
            num_ftrs = resnet.fc.in_features
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 2)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # (B, 512, 4, 10)
        
        if self.use_attention:
            x = self.se(x) # (B, 512, 4, 10)
            x_t = self.freq_pool(x).squeeze(2) # (B, 512, 10)
            x_t = x_t.permute(0, 2, 1) # (B, 10, 512)
            
            feat, weights = self.temporal_attn(x_t)
            self.last_weights = weights.detach().cpu().numpy()
            out = self.fc(feat)
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            out = self.fc(x)
            
        return out

if __name__ == "__main__":
    model = WhistleNet(use_attention=True)
    dummy_input = torch.randn(4, 1, 128, 313)
    output = model(dummy_input)
    print("WhistleNet (Attention) Output shape:", output.shape)
    print("Attention weights shape:", model.last_weights.shape)
    print("Attention weights sum for first sample:", model.last_weights[0].sum())
