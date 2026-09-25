import os
import shutil
import subprocess
import json
import re
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import X3DFreeKickModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

model = X3DFreeKickModel(num_classes=2, pretrained=False)
model.load_state_dict(torch.load("checkpoints/x3d_attention_best.pth", map_location=device))
model = model.to(device)
model.eval()

def generate_attention_plot(video_path, out_img_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 40:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(transform(frame))
    cap.release()
    
    while len(frames) < 40:
        if len(frames) > 0: frames.append(frames[-1].clone())
        else: frames.append(torch.zeros((3, 224, 224)))
        
    input_tensor = torch.stack(frames, dim=1).unsqueeze(0).to(device)
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                out = model(input_tensor)
        else:
            out = model(input_tensor)
            
    weights = model.model.blocks[5].last_weights.squeeze()
    num_weights = len(weights)
    
    plt.figure(figsize=(10, 2))
    plt.bar(np.arange(num_weights), weights, color='red')
    plt.xlim(-0.5, num_weights - 0.5)
    plt.ylim(0, max(0.1, weights.max() * 1.1))
    plt.title(f"Temporal Attention Weights ({num_weights} steps = 20 seconds)")
    plt.xlabel(f"Temporal Step (0 to {num_weights-1})")
    plt.ylabel("Relevance Score")
    plt.tight_layout()
    plt.savefig(out_img_path)
    plt.close()

out_dir = "/home/pilot/.gemini/antigravity/brain/54051a72-f909-4f8f-afda-b38268595c9a/error_clips"
if os.path.exists(out_dir): shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

data_dir = "/home/pilot/Desktop/ballsaction/soccernet_data"
report_file = "/home/pilot/Desktop/ballsaction/soccernet_eval_report.txt"
highlights_dir = "/home/pilot/Desktop/ballsaction/soccernet_eval_highlights"

def parse_gt(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gt = {1: [], 2: []}
    target_labels = {'Foul', 'Direct free-kick', 'Indirect free-kick', 'Penalty'}
    for ann in data.get('annotations', []):
        if ann['label'] in target_labels:
            half_str, time_str = ann['gameTime'].split(' - ')
            half = int(half_str)
            mm, ss = map(int, time_str.split(':'))
            sec = mm * 60 + ss
            gt[half].append((sec, ann['label']))
    return gt

fps = [] # {path, plot, info}
fns = [] # {path, plot, info}
tps = [] # {path, plot, info}

with open(report_file, 'r') as f:
    lines = f.readlines()

current_game = None
matched_gts = set()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith("Evaluating Match:"):
        current_game = line.split("Evaluating Match: ")[1]
    
    if line.startswith("Half"):
        m = re.match(r"Half (\d+) \| Clip (\d+) \[(\d+)s - (\d+)s\] -> (.*)", line)
        if m:
            half = int(m.group(1))
            clip_idx = int(m.group(2))
            c_start = int(m.group(3))
            c_end = int(m.group(4))
            status = m.group(5)
            
            i += 1
            prob_line = lines[i].strip()
            
            vid_name = f"{current_game.replace('/', '_')}_half{half}"
            src_clip = os.path.join(highlights_dir, f"{vid_name}_clip{clip_idx}_{c_start}s.mp4")
            
            if status == "FALSE POSITIVE":
                dest_clip = os.path.join(out_dir, f"FP_{vid_name}_clip{clip_idx}.mp4")
                if os.path.exists(src_clip):
                    shutil.copy(src_clip, dest_clip)
                    plot_path = os.path.join(out_dir, f"FP_{vid_name}_clip{clip_idx}_attn.png")
                    generate_attention_plot(dest_clip, plot_path)
                    fps.append({'path': dest_clip, 'plot': plot_path, 'info': f"{vid_name} [{c_start}s-{c_end}s]\n{prob_line}"})
            elif status == "TRUE POSITIVE":
                i += 1
                gt_line = lines[i].strip()
                if gt_line.startswith("Matched GTs:"):
                    # Extract timestamps: e.g. 'Foul @ 70s'
                    gt_matches = re.findall(r"([\w\s-]+?) @ (\d+)s", gt_line)
                    for lbl, sec_str in gt_matches:
                        matched_gts.add(f"{current_game}_half{half}_{sec_str}")
                
                dest_clip = os.path.join(out_dir, f"TP_{vid_name}_clip{clip_idx}.mp4")
                if os.path.exists(src_clip):
                    shutil.copy(src_clip, dest_clip)
                    plot_path = os.path.join(out_dir, f"TP_{vid_name}_clip{clip_idx}_attn.png")
                    generate_attention_plot(dest_clip, plot_path)
                    tps.append({'path': dest_clip, 'plot': plot_path, 'info': f"{vid_name} [{c_start}s-{c_end}s]\n{prob_line}\n{gt_line}"})
    i += 1

eval_games = []
for line in lines:
    if line.strip().startswith("Evaluating Match:"):
        eval_games.append(line.split("Evaluating Match: ")[1].strip())

processes = []
for game in eval_games:
    json_path = os.path.join(data_dir, game, "Labels-v2.json")
    gt_events = parse_gt(json_path)
    for half in [1, 2]:
        vid_path = os.path.join(data_dir, game, f"{half}_224p.mkv")
        vid_name = f"{game.replace('/', '_')}_half{half}"
        for sec, label in gt_events[half]:
            gt_id = f"{game}_half{half}_{sec}"
            if gt_id not in matched_gts:
                c_start = max(0, sec - 10)
                duration = 20
                out_clip = os.path.join(out_dir, f"FN_{vid_name}_{sec}s.mp4")
                clip_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", str(c_start),
                    "-i", vid_path,
                    "-t", str(duration),
                    "-vf", "scale=-2:480",
                    "-c:v", "libx264", "-crf", "28", "-preset", "fast",
                    "-c:a", "aac", "-b:a", "128k",
                    out_clip
                ]
                p = subprocess.Popen(clip_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                processes.append(p)
                fns.append({'path': out_clip, 'info': f"{vid_name} [{sec}s]\nMissed Label: {label}"})
                
                # Limit concurrent ffmpeg to 8 to avoid OOM
                if len(processes) >= 8:
                    for p in processes:
                        p.wait()
                    processes = []

for p in processes:
    p.wait()

for fn in fns:
    plot_path = fn['path'].replace('.mp4', '_attn.png')
    generate_attention_plot(fn['path'], plot_path)
    fn['plot'] = plot_path

md_path = "/home/pilot/.gemini/antigravity/brain/54051a72-f909-4f8f-afda-b38268595c9a/error_analysis.md"

with open(md_path, 'w') as f:
    f.write("# False Positive & False Negative Analysis\n\n")
    
    f.write("## False Positives (FPs)\n")
    f.write("Clips that the model thought contained a foul, but didn't actually match any Ground Truth events.\n\n")
    
    if len(fps) > 0:
        f.write("````carousel\n")
        for idx, fp in enumerate(fps):
            f.write(f"![FP Clip {idx}]({fp['path']})\n")
            f.write(f"![Attention Weights]({fp['plot']})\n")
            lines_info = fp['info'].split('\n')
            f.write(f"**{lines_info[0]}**\n")
            if len(lines_info) > 1:
                f.write(f"{lines_info[1]}\n")
            if idx < len(fps) - 1:
                f.write("<!-- slide -->\n")
        f.write("````\n\n")
    else:
        f.write("No False Positives found.\n\n")
        
    f.write("## False Negatives (FNs)\n")
    f.write("Ground Truth events that the model completely missed. 20-second clips centered on the actual event.\n\n")
    
    if len(fns) > 0:
        f.write("````carousel\n")
        for idx, fn in enumerate(fns):
            f.write(f"![FN Clip {idx}]({fn['path']})\n")
            f.write(f"![Attention Weights]({fn['plot']})\n")
            lines_info = fn['info'].split('\n')
            f.write(f"**{lines_info[0]}**\n")
            if len(lines_info) > 1:
                f.write(f"*{lines_info[1]}*\n")
            if idx < len(fns) - 1:
                f.write("<!-- slide -->\n")
        f.write("````\n\n")
    else:
        f.write("No False Negatives found.\n\n")
        
    f.write("## True Positives (TPs)\n")
    f.write("Successful detections of Ground Truth events. 20-second clips showing what the model correctly identified.\n\n")
    
    if len(tps) > 0:
        f.write("````carousel\n")
        for idx, tp in enumerate(tps):
            f.write(f"![TP Clip {idx}]({tp['path']})\n")
            f.write(f"![Attention Weights]({tp['plot']})\n")
            lines_info = tp['info'].split('\n')
            f.write(f"**{lines_info[0]}**\n")
            if len(lines_info) > 1:
                f.write(f"{lines_info[1]}\n")
            if len(lines_info) > 2:
                f.write(f"*{lines_info[2]}*\n")
            if idx < len(tps) - 1:
                f.write("<!-- slide -->\n")
        f.write("````\n")
    else:
        f.write("No True Positives found.\n")

print(f"Extracted {len(tps)} TPs, {len(fps)} FPs, and {len(fns)} FNs.")
