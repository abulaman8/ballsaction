import os
import glob
import cv2
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model_fusion import X3DMultiModalModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Native 16:9 widescreen
vid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 398)),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)
amp_db = torchaudio.transforms.AmplitudeToDB()

def load_multimodal_clip(video_path, audio_path, num_frames=30):
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
            raw_frames.append(cv2.resize(rgb, (356, 200))) # 16:9 aspect ratio
            tensor_frames.append(vid_transform(rgb))
        frame_idx += 1
    cap.release()
    
    while len(tensor_frames) < num_frames:
        if tensor_frames:
            tensor_frames.append(tensor_frames[-1].clone())
            raw_frames.append(raw_frames[-1].copy())
        else:
            tensor_frames.append(torch.zeros((3, 224, 398)))
            raw_frames.append(np.zeros((200, 356, 3), dtype=np.uint8))
            
    vid_tensor = torch.stack(tensor_frames, dim=1).unsqueeze(0).to(device)
    
    # Load audio
    spec_tensor = torch.zeros((1, 1, 128, 469), dtype=torch.float32)
    if os.path.exists(audio_path):
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000: wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        target_len = 240000
        if wav.shape[1] < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
        else:
            wav = wav[:, :target_len]
        s = amp_db(mel_tf(wav))
        if s.shape[2] < 469:
            s = torch.nn.functional.pad(s, (0, 469 - s.shape[2]))
        else:
            s = s[:, :, :469]
        spec_tensor = s.unsqueeze(0)
        
    return vid_tensor, spec_tensor.to(device), raw_frames

def generate_heatmap(video_path, model_type='foul', save_path='foul_heatmap_v4.png'):
    model = X3DMultiModalModel(num_classes=2, pretrained=False).to(device)
    if model_type == 'foul':
        ckpt = 'checkpoints/x3d_foul_multimodal_best.pth'
        title_prefix = 'Multimodal Foul Detection: Realigned Tackle & Synchronized Whistle (15s Window: [-5s, +10s])'
        step_offsets = [(-5.0 + i * 0.5) for i in range(30)]
        rep_indices = [0, 5, 10, 15, 20, 25] # 0s, 2.5s, 5s (contact), 7.5s, 10s, 12.5s
        rep_labels = ["-5.0s (Approach)", "-2.5s (Tackle Start)", "0.0s (Contact Moment)", "+2.5s (Tumble)", "+5.0s (Whistle/Reaction)", "+7.5s (Stoppage)"]
        key_frame_idx = 2
    else:
        ckpt = 'checkpoints/x3d_setpiece_multimodal_best.pth'
        title_prefix = 'Multimodal Set-Piece Detection: Pre-Kick Routine & Whistle (15s Window: [-10s, +5s])'
        step_offsets = [(-10.0 + i * 0.5) for i in range(30)]
        rep_indices = [0, 6, 12, 18, 20, 26] # -10s, -7s, -4s, -1s (runup), 0s (kick), +3s (flight)
        rep_labels = ["-10.0s (Wall Setup)", "-7.0s (Placement)", "-4.0s (Referee Whistle)", "-1.0s (Run-up)", "0.0s (Kick Moment)", "+3.0s (Ball Flight)"]
        key_frame_idx = 4
        
    if not os.path.exists(ckpt):
        print(f"Checkpoint {ckpt} not found, skipping heatmap.")
        return
        
    ckpt_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_dict['model_state_dict'] if 'model_state_dict' in ckpt_dict else ckpt_dict)
    model.eval()
    
    audio_path = video_path.replace(".mp4", ".wav")
    vid_tensor, spec_tensor, raw_frames = load_multimodal_clip(video_path, audio_path)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out, head_weights = model(vid_tensor, spec_tensor, return_head_weights=True)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            
    hw = head_weights[0].cpu().numpy() # (4, 30)
    avg_weights = hw.mean(axis=0)       # (30,)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 6, height_ratios=[1.2, 0.7, 1.0, 0.9])
    
    # Row 1: Representative 16:9 video frames
    for i, f_idx in enumerate(rep_indices):
        ax = fig.add_subplot(gs[0, i])
        if f_idx < len(raw_frames):
            ax.imshow(raw_frames[f_idx])
        ax.axis('off')
        is_key = (i == key_frame_idx)
        border_color = '#e74c3c' if is_key else '#333333'
        lw = 3.5 if is_key else 1
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(lw)
            spine.set_visible(True)
        ax.set_title(rep_labels[i], fontsize=11, fontweight='bold' if is_key else 'normal', color='#222222')
        
    # Row 2: Synchronized Audio Mel-Spectrogram
    ax_spec = fig.add_subplot(gs[1, :])
    spec_np = spec_tensor[0, 0].cpu().numpy()
    im_spec = ax_spec.imshow(spec_np, aspect='auto', origin='lower', cmap='inferno', extent=[step_offsets[0], step_offsets[-1] + 0.5, 0, 8000])
    ax_spec.set_ylabel("Freq (Hz)", fontsize=11, fontweight='bold')
    ax_spec.set_title("Synchronized Audio Mel-Spectrogram (16kHz Acoustic Stream with Whistle Energy)", fontsize=12, fontweight='bold')
    plt.colorbar(im_spec, ax=ax_spec, orientation='vertical', pad=0.01).set_label('dB', fontsize=10)
    
    # Row 3: 4-Head Attention Heatmap
    ax_heat = fig.add_subplot(gs[2, :])
    im_heat = ax_heat.imshow(hw, aspect='auto', cmap='plasma', interpolation='nearest',
                            extent=[step_offsets[0], step_offsets[-1] + 0.5, 3.5, -0.5])
    ax_heat.set_yticks(range(4))
    head_names = ["Head 0 (Pre-Event Buildup)", "Head 1 (Acoustic Whistle Align)", "Head 2 (Physical Impact / Strike)", "Head 3 (Aftermath / Reaction)"]
    ax_heat.set_yticklabels(head_names, fontsize=10, fontweight='bold')
    ax_heat.set_title("Frame-Synchronous Multimodal Cross-Attention Across 30 Half-Second Steps", fontsize=12, fontweight='bold')
    plt.colorbar(im_heat, ax=ax_heat, orientation='vertical', pad=0.01).set_label('Attn Weight', fontsize=10)
    
    # Row 4: Mean Attention Curve
    ax_curve = fig.add_subplot(gs[3, :])
    time_pts = [s + 0.25 for s in step_offsets]
    uniform_weight = 1.0 / 30.0
    
    ax_curve.plot(time_pts, avg_weights, color='#1f77b4', linewidth=2.5, marker='o', label='Fused Multimodal Attention')
    ax_curve.axhline(uniform_weight, color='gray', linestyle='--', linewidth=1.5, label='Uniform Baseline (1/30 = 0.033)')
    ax_curve.axvline(0.0, color='red', linestyle='-', linewidth=2.0, alpha=0.8, label='Reference Event (t=0s)')
    
    ax_curve.set_xlim(step_offsets[0], step_offsets[-1] + 0.5)
    ax_curve.set_xlabel("Relative Timeline to Event (seconds)", fontsize=12, fontweight='bold')
    ax_curve.set_ylabel("Attention Weight", fontsize=11, fontweight='bold')
    ax_curve.set_title(f"Ensemble Attention Profile (Detection Confidence: {prob*100:.1f}%)", fontsize=12, fontweight='bold')
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc='upper right', framealpha=0.9)
    
    fig.suptitle(f"{title_prefix}\nSample: {os.path.basename(video_path)} | P({model_type}) = {prob:.4f}", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved multimodal heatmap to {save_path}")

if __name__ == "__main__":
    sp_samples = glob.glob("setpiece_dataset_v4/val/set_piece/*.mp4")
    if sp_samples:
        generate_heatmap(sp_samples[0], model_type='setpiece', save_path='setpiece_heatmap_v4.png')
    foul_samples = glob.glob("foul_dataset_v4/val/foul/*.mp4")
    if foul_samples:
        generate_heatmap(foul_samples[0], model_type='foul', save_path='foul_heatmap_v4.png')
