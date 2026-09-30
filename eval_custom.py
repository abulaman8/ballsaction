import os
import cv2
import json
import torch
import subprocess
import torchvision.transforms as transforms
import torchaudio
from model import X3DFreeKickModel
from model_audio import WhistleNet

FPS = 2
WINDOW_SECONDS = 10
WINDOW_FRAMES = WINDOW_SECONDS * FPS
STRIDE_SECONDS = 5
STRIDE_FRAMES = STRIDE_SECONDS * FPS

FOUL_THRESHOLD = 0.75
TOLERANCE_SECONDS = 10.0

vid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

def merge_intervals(intervals, max_gap=10.0):
    if not intervals: return []
    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    for curr in intervals[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1] + max_gap:
            prev[1] = max(prev[1], curr[1]) 
            prev[2] = max(prev[2], curr[2]) 
        else:
            merged.append(list(curr))
    return merged

def process_video(video_path, v_foul, a_whistle, device, out_dir, vid_name):
    print(f"Processing {os.path.basename(video_path)}...")
    
    full_audio_wav = "eval_custom_full_audio.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-q:a", "0", "-map", "a", full_audio_wav], capture_output=True)
    
    try:
        full_waveform, sr = torchaudio.load(full_audio_wav)
        if full_waveform.shape[0] > 1: full_waveform = torch.mean(full_waveform, dim=0, keepdim=True)
        if sr != 16000: full_waveform = torchaudio.transforms.Resample(sr, 16000)(full_waveform)
    except Exception:
        full_waveform = torch.zeros((1, 150 * 60 * 16000))
        
    def get_audio_window(t_start, t_end):
        s_start = int(t_start * 16000)
        s_end = int(t_end * 16000)
        chunk = full_waveform[:, s_start:s_end]
        target_len = int((t_end - t_start) * 16000)
        
        if chunk.shape[1] < target_len:
            pad = target_len - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
            
        spec = mel_transform(chunk)
        spec = amp_to_db(spec)
        if spec.shape[2] < 313: spec = torch.nn.functional.pad(spec, (0, 313 - spec.shape[2]))
        else: spec = spec[:, :, :313]
        return spec.unsqueeze(0)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
        
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0: original_fps = 25.0
        
    frame_interval = int(round(original_fps / FPS))
    frame_idx = 0
    rolling_buffer = []
    
    detected_fouls = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        current_sec = frame_idx / original_fps
        if frame_idx % frame_interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rolling_buffer.append((vid_transform(img), current_sec))
            
            if len(rolling_buffer) >= WINDOW_FRAMES:
                input_frames = [x[0] for x in rolling_buffer[:WINDOW_FRAMES]]
                t_end = rolling_buffer[WINDOW_FRAMES-1][1]
                t_start = max(0, t_end - WINDOW_SECONDS)
                
                vid_tensor = torch.stack(input_frames, dim=1).unsqueeze(0).to(device)
                audio_tensor = get_audio_window(t_start, t_end).to(device)
                
                with torch.no_grad():
                    output = v_foul(vid_tensor)
                    v_prob_f = torch.softmax(output, dim=1)[0, 1].item()
                    
                    a_prob = torch.softmax(a_whistle(audio_tensor), dim=1)[0, 1].item()
                    
                    if a_prob > 0.7:
                        v_prob_f = min(1.0, v_prob_f + 0.05)
                
                if v_prob_f > FOUL_THRESHOLD:
                    clip_start = max(0, t_start - 5.0)
                    clip_end = t_end + 5.0
                    detected_fouls.append((clip_start, clip_end, v_prob_f, "Foul", False))
                    
                rolling_buffer = rolling_buffer[STRIDE_FRAMES:]
        frame_idx += 1
    
    cap.release()
    
    merged_fouls = merge_intervals(detected_fouls, max_gap=5.0)
    
    extracted_info = []
    for idx, clip in enumerate(merged_fouls):
        c_start, c_end, prob, label, boosted = clip
        duration = c_end - c_start
        out_clip_name = f"{vid_name}_Foul_clip{idx}_{c_start:.0f}s.mp4"
        out_clip = os.path.join(out_dir, out_clip_name)
        clip_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(c_start),
            "-i", video_path,
            "-t", str(duration),
            "-map", "0:v:0", "-map", "0:a:0",
            "-vf", "scale=-2:480",
            "-c:v", "libx264", "-crf", "28", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            out_clip
        ]
        subprocess.run(clip_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        extracted_info.append({
            "clip_name": out_clip_name,
            "c_start": c_start,
            "c_end": c_end,
            "prob": prob
        })
    return extracted_info

def parse_gt(json_path):
    with open(json_path, 'r') as f: data = json.load(f)
    gt_fouls = []
    for ann in data.get('events', []):
        action = ann.get('primary_action', '')
        if 'Foul' in action or 'Gelbe Karte' in action or 'Rote Karte' in action:
            frame_in = ann['frame_in']
            sec = frame_in / 25.0
            gt_fouls.append(sec)
    return gt_fouls

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    v_foul = X3DFreeKickModel(num_classes=2, pretrained=False).to(device)
    v_foul.load_state_dict(torch.load("checkpoints/x3d_foul_best.pth", map_location=device))
    v_foul.eval()
    
    a_whistle = WhistleNet().to(device)
    a_whistle.load_state_dict(torch.load("checkpoints/audio_whistle_best.pth", map_location=device))
    a_whistle.eval()
    
    out_dir = "/home/pilot/Desktop/ballsaction/custom_eval_highlights"
    os.makedirs(out_dir, exist_ok=True)
    
    vid_path = "custom_data/SOK - PGM, International HD - 1. BL 2025-2026 1. Spieltag FC Bayern München vs. RB Leipzig-010.mp4"
    json_path = "custom_data/SOK-PGM-International-HD-1-BL-2025-2026-1-Spieltag-FC-Bayern-Muenchen-vs-RB-Leipzig_4693964.json"
    
    vid_name = "Bayern_Leipzig_Match1"
    
    gt_fouls = parse_gt(json_path)
    
    foul_info = process_video(vid_path, v_foul, a_whistle, device, out_dir, vid_name)
    
    review_data = []
    matched = set()
    
    for p_info in foul_info:
        p_start = p_info["c_start"]
        p_end = p_info["c_end"]
        prob = p_info["prob"]
        clip_name = p_info["clip_name"]
        
        hit = [g for g in gt_fouls if p_start - TOLERANCE_SECONDS <= g <= p_end + TOLERANCE_SECONDS]
        if hit:
            is_tp = True
            for g in hit: matched.add(g)
        else:
            is_tp = False
            
        review_data.append({
            "id": clip_name,
            "path": clip_name,
            "type": "TP" if is_tp else "FP",
            "label": "Foul",
            "pred_prob": prob,
            "clip_start": p_start,
            "clip_end": p_end
        })
    
    for g in gt_fouls:
        if g not in matched:
            fn_start = max(0, g - 10.0)
            fn_end = g + 10.0
            duration = fn_end - fn_start
            out_clip_name = f"{vid_name}_Foul_FN_{g:.0f}s.mp4"
            out_clip = os.path.join(out_dir, out_clip_name)
            clip_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(fn_start),
                "-i", vid_path,
                "-t", str(duration),
                "-map", "0:v:0", "-map", "0:a:0",
                "-vf", "scale=-2:480",
                "-c:v", "libx264", "-crf", "28", "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                out_clip
            ]
            subprocess.run(clip_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            review_data.append({
                "id": out_clip_name,
                "path": out_clip_name,
                "type": "FN",
                "label": "Foul",
                "pred_prob": 0.0,
                "clip_start": fn_start,
                "clip_end": fn_end
            })
            
    with open(os.path.join(out_dir, "review_metadata.json"), "w") as f:
        json.dump(review_data, f, indent=4)
        
    print("Saved review_metadata.json in custom_eval_highlights!")

if __name__ == "__main__":
    evaluate()
