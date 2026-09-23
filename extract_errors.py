import os
import shutil
import subprocess
import json
import re

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

fps = [] # {path, info}
fns = [] # {path, info}

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
                    fps.append({'path': dest_clip, 'info': f"{vid_name} [{c_start}s-{c_end}s]\n{prob_line}"})
            elif status == "TRUE POSITIVE":
                i += 1
                gt_line = lines[i].strip()
                if gt_line.startswith("Matched GTs:"):
                    # Extract timestamps: e.g. 'Foul @ 70s'
                    gt_matches = re.findall(r"([\w\s-]+?) @ (\d+)s", gt_line)
                    for lbl, sec_str in gt_matches:
                        matched_gts.add(f"{current_game}_half{half}_{sec_str}")
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

md_path = "/home/pilot/.gemini/antigravity/brain/54051a72-f909-4f8f-afda-b38268595c9a/error_analysis.md"

with open(md_path, 'w') as f:
    f.write("# False Positive & False Negative Analysis\n\n")
    
    f.write("## False Positives (FPs)\n")
    f.write("Clips that the model thought contained a foul, but didn't actually match any Ground Truth events.\n\n")
    
    if len(fps) > 0:
        f.write("````carousel\n")
        for idx, fp in enumerate(fps):
            f.write(f"![FP Clip {idx}]({fp['path']})\n")
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
            lines_info = fn['info'].split('\n')
            f.write(f"**{lines_info[0]}**\n")
            if len(lines_info) > 1:
                f.write(f"*{lines_info[1]}*\n")
            if idx < len(fns) - 1:
                f.write("<!-- slide -->\n")
        f.write("````\n")
    else:
        f.write("No False Negatives found.\n")

print(f"Extracted {len(fps)} FPs and {len(fns)} FNs.")
