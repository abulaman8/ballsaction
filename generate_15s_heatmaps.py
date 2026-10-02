import os
import glob
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

def load_15s_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    tensor_frames = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    frame_interval = int(round(fps / 2.0))
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

def generate_heatmap(video_path, model_type='foul', save_path='foul_heatmap_v3.png'):
    model = X3DFreeKickModel(num_classes=2, pretrained=False).to(device)
    if model_type == 'foul':
        ckpt = 'checkpoints/x3d_foul_best.pth'
        title_prefix = 'Foul Detection (15s Window: [-5s, +10s])'
    else:
        ckpt = 'checkpoints/x3d_setpiece_best.pth'
        title_prefix = 'Set-Piece Detection (15s Window: [-5s, +10s])'
        
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    
    vid_tensor, raw_frames = load_15s_frames(video_path)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(vid_tensor)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            
    weights = model.model.blocks[5].last_weights[0, :, 0] # shape: (15,)
    num_steps = len(weights)
    
    # Temporal labels relative to contact (contact at ~5.0s into clip)
    step_labels = [f"{-5+i}s..{-4+i}s" for i in range(num_steps)]
    peak_idx = int(np.argmax(weights))
    
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1])
    
    # Pick 6 representative frames across the 15-second window
    rep_indices = [0, 5, 10, 15, 20, 25] # frames at 0s, 2.5s, 5s (contact), 7.5s, 10s, 12.5s
    rep_labels = ["-5.0s (Build-up)", "-2.5s (Approach)", "0.0s (Contact Moment)", "+2.5s (Reaction)", "+5.0s (Referee Signal)", "+7.5s (Aftermath)"]
    
    for i, f_idx in enumerate(rep_indices):
        ax = fig.add_subplot(gs[0, i])
        if f_idx < len(raw_frames):
            ax.imshow(raw_frames[f_idx])
        ax.axis('off')
        is_contact = (i == 2)
        border_color = 'red' if is_contact else 'gray'
        lw = 3 if is_contact else 1
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(lw)
            spine.set_visible(True)
        ax.set_title(rep_labels[i], fontsize=10, fontweight='bold' if is_contact else 'normal', color='red' if is_contact else 'black')
    
    # Row 2: Attention Heatmap Bar Chart across 15 seconds
    ax_bar = fig.add_subplot(gs[1, :])
    norm_w = weights / (max(weights) + 1e-6)
    colors = plt.cm.viridis(norm_w)
    bars = ax_bar.bar(range(num_steps), weights * 100, color=colors, edgecolor='black', width=0.6)
    ax_bar.set_xticks(range(num_steps))
    ax_bar.set_xticklabels(step_labels, rotation=35, ha='right', fontsize=9)
    ax_bar.set_ylabel('Attention Weight (%)', fontsize=12, fontweight='bold')
    ax_bar.set_xlabel('Temporal Offset Relative to Ground Truth Contact Spot (0.0s)', fontsize=12, fontweight='bold')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Mark contact region
    ax_bar.axvspan(4.5, 5.5, color='red', alpha=0.15, label='Contact Moment (t=0s)')
    
    for bar, w in zip(bars, weights):
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{w*100:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    ax_bar.legend(loc='upper right')
    
    clip_basename = os.path.basename(video_path)
    fig.suptitle(f"{title_prefix}\nClip: {clip_basename} | Model Probability: {prob*100:.1f}% | Peak Attention: {step_labels[peak_idx]} ({weights[peak_idx]*100:.1f}%)",
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved 15s attention heatmap to: {save_path}")

if __name__ == "__main__":
    foul_clips = glob.glob('foul_dataset_v3/val/foul/*.mp4')
    if foul_clips:
        generate_heatmap(foul_clips[0], model_type='foul', save_path='foul_heatmap_v3.png')
        
    sp_clips = glob.glob('setpiece_dataset_v3/val/set_piece/*.mp4')
    if sp_clips:
        generate_heatmap(sp_clips[0], model_type='setpiece', save_path='setpiece_heatmap_v3.png')
