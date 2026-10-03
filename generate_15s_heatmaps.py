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

def generate_heatmap(video_path, model_type='foul', save_path='foul_heatmap_v4.png'):
    model = X3DFreeKickModel(num_classes=2, pretrained=False).to(device)
    if model_type == 'foul':
        ckpt = 'checkpoints/x3d_foul_best.pth'
        title_prefix = 'Foul Detection: Spatial Attention (15s Window: [-5s, +10s])'
        key_event_sec = 5.0 # contact at 5.0s
        key_event_label = "Contact Moment (t=0s)"
        step_offsets = [(-5.0 + i * 0.5) for i in range(30)]
        rep_indices = [0, 5, 10, 15, 20, 25] # 0s, 2.5s, 5s (contact), 7.5s, 10s, 12.5s
        rep_labels = ["-5.0s (Approach)", "-2.5s (Tackle Start)", "0.0s (Contact Moment)", "+2.5s (Tumble)", "+5.0s (Reaction)", "+7.5s (Whistle/Play Stop)"]
        key_frame_idx = 2
    else:
        ckpt = 'checkpoints/x3d_setpiece_best.pth'
        title_prefix = 'Set-Piece Detection: Pre-Kick Buildup (15s Window: [-10s, +5s])'
        key_event_sec = 10.0 # kick at 10.0s
        key_event_label = "Kick Moment (t=0s)"
        step_offsets = [(-10.0 + i * 0.5) for i in range(30)]
        rep_indices = [0, 6, 12, 18, 20, 26] # -10s, -7s, -4s, -1s (runup), 0s (kick), +3s (flight)
        rep_labels = ["-10.0s (Wall Setup)", "-7.0s (Ball Placed)", "-4.0s (Referee Stance)", "-1.0s (Run-up)", "0.0s (Kick Moment)", "+3.0s (Ball Flight)"]
        key_frame_idx = 4
        
    if not os.path.exists(ckpt):
        print(f"Checkpoint {ckpt} not found, skipping heatmap.")
        return
        
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    
    vid_tensor, raw_frames = load_15s_frames(video_path)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out, head_weights = model(vid_tensor, return_head_weights=True)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            
    # head_weights: (1, 4, 30)
    hw = head_weights[0].cpu().numpy() # (4, 30)
    avg_weights = hw.mean(axis=0)       # (30,)
    
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.3, 1.0, 0.9])
    
    # Row 1: Representative video frames
    for i, f_idx in enumerate(rep_indices):
        ax = fig.add_subplot(gs[0, i])
        if f_idx < len(raw_frames):
            ax.imshow(raw_frames[f_idx])
        ax.axis('off')
        is_key = (i == key_frame_idx)
        border_color = 'red' if is_key else '#444444'
        lw = 3.5 if is_key else 1
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(lw)
            spine.set_visible(True)
        ax.set_title(rep_labels[i], fontsize=10, fontweight='bold' if is_key else 'normal', color='red' if is_key else 'black')
    
    # Row 2: Aggregate Multi-Head Attention Bar Chart (30 steps = 15 seconds)
    ax_bar = fig.add_subplot(gs[1, :])
    norm_w = avg_weights / (max(avg_weights) + 1e-6)
    colors = plt.cm.plasma(norm_w)
    bars = ax_bar.bar(range(30), avg_weights * 100, color=colors, edgecolor='black', width=0.7)
    
    step_labels = [f"{offset:+.1f}s" for offset in step_offsets]
    ax_bar.set_xticks(range(30))
    ax_bar.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=9)
    ax_bar.set_ylabel('Avg Attention (%)', fontsize=11, fontweight='bold')
    ax_bar.set_xlabel('Time Relative to Event Timestamp (0.0s)', fontsize=11, fontweight='bold')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Mark the key event frame
    key_step = int(key_event_sec * 2)
    ax_bar.axvspan(key_step - 0.5, key_step + 0.5, color='red', alpha=0.2, label=key_event_label)
    
    for bar, w in zip(bars, avg_weights):
        h = bar.get_height()
        if h > 2.5:
            ax_bar.text(bar.get_x() + bar.get_width()/2., h + 0.2, f"{w*100:.1f}%", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax_bar.legend(loc='upper right')
    
    # Row 3: 4 Individual Head Subplots showing diversity and phase specialization
    head_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for h in range(4):
        ax_h = fig.add_subplot(gs[2, h if h < 3 else slice(3, 6) if False else h])
        # We place each head in its own sub-column
        ax_h.plot(range(30), hw[h] * 100, color=head_colors[h], lw=2.0, marker='o', markersize=3, label=f"Head {h}")
        ax_h.axvline(key_step, color='red', linestyle='--', alpha=0.5)
        ax_h.set_title(f"Head {h} Specialization", fontsize=10, fontweight='bold')
        ax_h.set_ylabel('Weight (%)', fontsize=9)
        ax_h.set_xticks([0, 10, 20, 29])
        ax_h.set_xticklabels([step_labels[0], step_labels[10], step_labels[20], step_labels[29]], fontsize=8)
        ax_h.grid(True, linestyle=':', alpha=0.5)
        ax_h.set_ylim(0, max(hw[h].max() * 120, 10))

    clip_basename = os.path.basename(video_path)
    fig.suptitle(f"{title_prefix}\nClip: {clip_basename} | Confidence: {prob*100:.1f}% | 30 Frames @ 2 FPS (Spatial Pool + Diversity Loss)",
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved 15s attention heatmap to: {save_path}")

if __name__ == "__main__":
    foul_clips = glob.glob('foul_dataset_v3/val/foul/*.mp4')
    if foul_clips:
        generate_heatmap(foul_clips[0], model_type='foul', save_path='foul_heatmap_v4.png')
        
    sp_clips = glob.glob('setpiece_dataset_v4/val/set_piece/*.mp4')
    if sp_clips:
        generate_heatmap(sp_clips[0], model_type='setpiece', save_path='setpiece_heatmap_v4.png')
