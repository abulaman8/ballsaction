import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import X3DFreeKickModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

def load_clip_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    tensor_frames = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    frame_interval = int(round(fps / 2.0)) # 2 FPS
    if frame_interval <= 0: frame_interval = 1
    
    frame_idx = 0
    while len(tensor_frames) < num_frames:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_frames.append(cv2.resize(rgb, (320, 180)))
            tensor_frames.append(transform(rgb))
        frame_idx += 1
    cap.release()
    
    while len(tensor_frames) < num_frames:
        if tensor_frames:
            tensor_frames.append(tensor_frames[-1].clone())
            raw_frames.append(raw_frames[-1].copy())
        else:
            tensor_frames.append(torch.zeros((3, 224, 224)))
            raw_frames.append(np.zeros((180, 320, 3), dtype=np.uint8))
            
    vid_tensor = torch.stack(tensor_frames, dim=1).unsqueeze(0).to(device)
    return vid_tensor, raw_frames

def visualize_attention(video_path, model_type='foul', output_dir='attention_heatmaps'):
    os.makedirs(output_dir, exist_ok=True)
    
    model = X3DFreeKickModel(num_classes=2, pretrained=False).to(device)
    if model_type == 'foul':
        ckpt = 'checkpoints/x3d_foul_best.pth'
        title_prefix = 'Foul Detection'
    else:
        ckpt = 'checkpoints/x3d_setpiece_best.pth'
        title_prefix = 'Set-Piece Detection'
        
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    
    vid_tensor, raw_frames = load_clip_frames(video_path)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(vid_tensor)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            
    weights = model.model.blocks[5].last_weights[0, :, 0] # (5,)
    
    # 5 temporal chunks across 10 seconds:
    # Chunk 0: 0-2s, Chunk 1: 2-4s, Chunk 2: 4-6s, Chunk 3: 6-8s, Chunk 4: 8-10s
    chunk_labels = ['0-2s', '2-4s', '4-6s', '6-8s', '8-10s']
    peak_idx = int(np.argmax(weights))
    
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.2, 1])
    
    # Row 1: Center frame of each 2s chunk
    chunk_frame_indices = [2, 6, 10, 14, 18]
    for i, f_idx in enumerate(chunk_frame_indices):
        ax = fig.add_subplot(gs[0, i])
        if f_idx < len(raw_frames):
            ax.imshow(raw_frames[f_idx])
        ax.axis('off')
        is_peak = (i == peak_idx)
        border_color = 'red' if is_peak else 'gray'
        lw = 3 if is_peak else 1
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(lw)
            spine.set_visible(True)
        ax.set_title(f"Chunk {i+1} ({chunk_labels[i]})\nWeight: {weights[i]*100:.1f}%" + (" ★ PEAK" if is_peak else ""), 
                     fontsize=10, fontweight='bold' if is_peak else 'normal', color='red' if is_peak else 'black')
    
    # Row 2: Attention Heatmap Bar Chart
    ax_bar = fig.add_subplot(gs[1, :])
    colors = plt.cm.plasma(weights / (max(weights) + 1e-6))
    bars = ax_bar.bar(chunk_labels, weights * 100, color=colors, edgecolor='black', width=0.5)
    
    for bar in bars:
        h = bar.get_height()
        ax_bar.annotate(f'{h:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    ax_bar.set_ylabel('Attention Weight (%)', fontsize=12)
    ax_bar.set_xlabel('Clip Time Interval (Seconds)', fontsize=12)
    ax_bar.set_ylim(0, max(weights * 100) * 1.25)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    
    clip_basename = os.path.basename(video_path)
    fig.suptitle(f"{title_prefix} — Clip: {clip_basename}\nModel Confidence: {prob*100:.1f}% | Peak Attention: {chunk_labels[peak_idx]} ({weights[peak_idx]*100:.1f}%)",
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    out_name = f"{model_type}_heatmap_{os.path.splitext(clip_basename)[0]}.png"
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved attention heatmap to: {out_path}")
    return out_path

if __name__ == '__main__':
    import glob
    # Test on a set-piece sample
    sp_samples = glob.glob('setpiece_dataset_v2/val/set_piece/*.mp4')
    if sp_samples:
        visualize_attention(sp_samples[0], model_type='setpiece')
        
    # Test on a foul sample
    foul_samples = glob.glob('foul_dataset_v2/val/foul/*.mp4')
    if foul_samples:
        visualize_attention(foul_samples[0], model_type='foul')
