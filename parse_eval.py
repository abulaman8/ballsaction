import re

with open("soccernet_eval_output.txt", "r") as f:
    lines = f.readlines()

clips = []
current_window = ""
audio_conf = 0
video_conf = 0
fused_conf = 0
boosted = False

for line in lines:
    if "Window:" in line:
        current_window = line.strip().split("Window: ")[1]
    elif "Audio CNN" in line:
        audio_conf = line.strip().split(": ")[1]
    elif "Video X3D" in line:
        video_conf = line.strip().split(": ")[1]
    elif "FUSED CONFIDENCE" in line:
        fused_conf = line.strip().split(": ")[1]
        if "Boosted" in line:
            boosted = True
        else:
            boosted = False
    elif "DETECTED HIGHLIGHT!" in line:
        clips.append({
            "window": current_window,
            "audio": audio_conf,
            "video": video_conf,
            "fused": fused_conf,
            "boosted": boosted
        })

print(f"Total Clips Detected: {len(clips)}")
for idx, c in enumerate(clips):
    print(f"Clip {idx+1} ({c['window']}):")
    print(f"  - Audio (Whistle): {c['audio']}")
    print(f"  - Video (Foul): {c['video']}")
    print(f"  - Fused: {c['fused']} {'(Boosted by Whistle!)' if c['boosted'] else ''}")

