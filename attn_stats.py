import os
import glob
import torch
import numpy as np
from dataset import X3DBinaryDataset
from model import X3DFreeKickModel
from torch.utils.data import DataLoader

def analyze_model_attention(checkpoint_path, data_dir, model_name="Model", max_samples=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f"  Analyzing Attention: {model_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Dataset: {data_dir}")
    print(f"=======================================================")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
        
    model = X3DFreeKickModel(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    dataset = X3DBinaryDataset(data_dir=data_dir, is_training=False, target_frames=30)
    indices = np.random.RandomState(42).permutation(len(dataset))[:max_samples]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=2)
    
    head_entropies = [] # (N, num_heads)
    inter_head_sims = [] # (N, num_pairs)
    head_peaks = [] # (N, num_heads)
    last_frame_weights = [] # (N, num_heads)
    first_frame_weights = [] # (N, num_heads)
    
    pos_head_weights = []
    bg_head_weights = []
    
    with torch.no_grad():
        for videos, targets in loader:
            videos = videos.to(device)
            with torch.amp.autocast('cuda'):
                outputs, head_weights = model(videos, return_head_weights=True)
            
            # head_weights: (B, num_heads, T)
            B, H, T = head_weights.shape
            hw = head_weights.cpu().numpy()
            
            for b in range(B):
                t_label = targets[b].item()
                if t_label == 1:
                    pos_head_weights.append(hw[b])
                else:
                    bg_head_weights.append(hw[b])
                    
                sample_entropies = []
                sample_peaks = []
                sample_last = []
                sample_first = []
                
                for h in range(H):
                    w = hw[b, h] + 1e-12
                    ent = -np.sum(w * np.log(w))
                    sample_entropies.append(ent)
                    sample_peaks.append(np.argmax(w))
                    sample_last.append(w[-1])
                    sample_first.append(w[0])
                    
                head_entropies.append(sample_entropies)
                head_peaks.append(sample_peaks)
                last_frame_weights.append(sample_last)
                first_frame_weights.append(sample_first)
                
                # Pairwise cosine similarities between heads
                pairs = []
                for h1 in range(H):
                    for h2 in range(h1 + 1, H):
                        w1 = hw[b, h1]
                        w2 = hw[b, h2]
                        cos = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-8)
                        pairs.append(cos)
                inter_head_sims.append(pairs)
                
    head_entropies = np.array(head_entropies)
    inter_head_sims = np.array(inter_head_sims)
    head_peaks = np.array(head_peaks)
    last_frame_weights = np.array(last_frame_weights)
    first_frame_weights = np.array(first_frame_weights)
    
    uniform_ent = np.log(30)
    print(f"\n[1] HEAD ENTROPY (Uniform 30-frame Baseline: {uniform_ent:.3f}):")
    for h in range(4):
        mean_ent = np.mean(head_entropies[:, h])
        std_ent = np.std(head_entropies[:, h])
        print(f"  Head {h}: Mean Entropy = {mean_ent:.3f} +/- {std_ent:.3f} (Peaked vs Uniform: {(1 - mean_ent/uniform_ent)*100:.1f}% concentration)")
        
    print(f"\n[2] INTER-HEAD COSINE SIMILARITY (Target: < 0.60 for diverse specialization):")
    pair_labels = ["H0-H1", "H0-H2", "H0-H3", "H1-H2", "H1-H3", "H2-H3"]
    mean_sims = np.mean(inter_head_sims, axis=0)
    for lbl, sim in zip(pair_labels, mean_sims):
        print(f"  Pair {lbl}: Cosine Similarity = {sim:.3f}")
    overall_sim = np.mean(mean_sims)
    print(f"  => Average Inter-Head Similarity: {overall_sim:.3f}")
    
    print(f"\n[3] EDGE BIAS AUDIT:")
    for h in range(4):
        mean_last = np.mean(last_frame_weights[:, h]) * 100
        mean_first = np.mean(first_frame_weights[:, h]) * 100
        print(f"  Head {h}: First Frame Weight = {mean_first:.1f}%, Last Frame Weight = {mean_last:.1f}% (Uniform baseline = 3.3%)")
        
    print(f"\n[4] PEAK ATTENTION FRAME TIMING:")
    for h in range(4):
        peaks = head_peaks[:, h]
        p_mean = np.mean(peaks)
        p_median = np.median(peaks)
        t_sec = p_mean * 0.5
        print(f"  Head {h}: Average Peak Frame = {p_mean:.1f} ({t_sec:.1f}s into 15s window), Median Frame = {p_median:.0f}")

    return {
        "entropies": np.mean(head_entropies, axis=0).tolist(),
        "avg_similarity": float(overall_sim),
        "last_frame_pct": (np.mean(last_frame_weights) * 100),
    }

if __name__ == "__main__":
    analyze_model_attention("checkpoints/x3d_foul_best.pth", "foul_dataset_v3/val", "FOUL MODEL (v3 dataset)")
    analyze_model_attention("checkpoints/x3d_setpiece_best.pth", "setpiece_dataset_v4/val", "SET-PIECE MODEL (v4 dataset)")
