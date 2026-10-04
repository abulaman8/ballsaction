import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_gate_features(windows):
    """
    Extracts 15-dimensional multimodal temporal features from a sequence of windows.
    windows: list of dicts with keys: 'raw_foul', 'raw_sp', 'audio'
    Returns: torch.FloatTensor of shape (N, 15)
    """
    N = len(windows)
    feats = []
    
    for i in range(N):
        w = windows[i]
        w_prev = windows[i-1] if i > 0 else w
        w_next = windows[i+1] if i < N - 1 else w
        
        vf_prev = float(w_prev.get('raw_foul', w_prev.get('raw_foul_prob', 0)))
        vf_curr = float(w.get('raw_foul', w.get('raw_foul_prob', 0)))
        vf_next = float(w_next.get('raw_foul', w_next.get('raw_foul_prob', 0)))
        
        vsp_prev = float(w_prev.get('raw_sp', w_prev.get('raw_sp_prob', 0)))
        vsp_curr = float(w.get('raw_sp', w.get('raw_sp_prob', 0)))
        vsp_next = float(w_next.get('raw_sp', w_next.get('raw_sp_prob', 0)))
        
        a_prev = float(w_prev.get('audio', w_prev.get('audio_prob', 0)))
        a_curr = float(w.get('audio', w.get('audio_prob', 0)))
        a_next = float(w_next.get('audio', w_next.get('audio_prob', 0)))
        
        a_max = max(a_prev, a_curr, a_next)
        a_min = min(a_prev, a_curr, a_next)
        a_fwd = a_next - a_curr
        a_back = a_curr - a_prev
        
        foul_audio_synergy = vf_curr * a_max
        foul_audio_dissonance = vf_curr * (1.0 - a_max)
        
        sp_audio_synergy = vsp_curr * a_max
        sp_audio_dissonance = vsp_curr * (1.0 - a_max)
        
        feat = [
            vf_prev, vf_curr, vf_next,
            vsp_prev, vsp_curr, vsp_next,
            a_prev, a_curr, a_next,
            a_max, a_min,
            foul_audio_synergy, foul_audio_dissonance,
            sp_audio_synergy, sp_audio_dissonance,
        ]
        feats.append(feat)
        
    return torch.tensor(feats, dtype=torch.float32)

class AVGateNet(nn.Module):
    """
    Learnable Audio-Visual Temporal Gating Network.
    Takes local temporal video and audio predictions, calculates cross-modal interaction,
    and learns residual corrections (Delta logits) to adjust foul and set-piece probabilities.
    """
    def __init__(self, in_features=15, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2) # [delta_logit_foul, delta_logit_sp]
        )
        
        # Initialize final layer weights and biases close to 0
        # so initial predictions default to unmodified video logits
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, feats, raw_vf, raw_vsp, return_logits=False, eps=1e-6):
        """
        feats: (B, 15)
        raw_vf: (B,) in [0, 1]
        raw_vsp: (B,) in [0, 1]
        Returns:
            final_foul_prob: (B,) in [0, 1]
            final_sp_prob: (B,) in [0, 1]
        """
        delta_logits = self.net(feats) # (B, 2)
        
        # Compute base video logits
        raw_vf_clamped = torch.clamp(raw_vf, eps, 1.0 - eps)
        base_foul_logit = torch.log(raw_vf_clamped / (1.0 - raw_vf_clamped))
        
        raw_vsp_clamped = torch.clamp(raw_vsp, eps, 1.0 - eps)
        base_sp_logit = torch.log(raw_vsp_clamped / (1.0 - raw_vsp_clamped))
        
        final_foul_logit = base_foul_logit + delta_logits[:, 0]
        final_sp_logit = base_sp_logit + delta_logits[:, 1]
        
        final_foul_prob = torch.clamp(torch.sigmoid(final_foul_logit), eps, 1.0 - eps)
        final_sp_prob = torch.clamp(torch.sigmoid(final_sp_logit), eps, 1.0 - eps)
        
        if return_logits:
            return final_foul_logit, final_sp_logit
        return final_foul_prob, final_sp_prob

if __name__ == "__main__":
    dummy_feats = torch.randn(4, 15)
    dummy_vf = torch.tensor([0.85, 0.40, 0.92, 0.10])
    dummy_vsp = torch.tensor([0.05, 0.80, 0.01, 0.02])
    
    gate = AVGateNet()
    f_prob, sp_prob = gate(dummy_feats, dummy_vf, dummy_vsp)
    print("Test passed! Shapes:", f_prob.shape, sp_prob.shape)
    print("Initial outputs (should be very close to raw inputs):")
    print("Foul:", dummy_vf, "->", f_prob)
    print("SetPiece:", dummy_vsp, "->", sp_prob)
