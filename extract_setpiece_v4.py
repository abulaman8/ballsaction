import os
import json
import random
import multiprocessing
import subprocess
import shutil
from SoccerNet.utils import getListGames

DATA_DIR = "soccernet_data"
OUT_DIR_SETP = "setpiece_dataset_v4"

def process_game(args):
    game_path, split = args
    game_dir = os.path.join(DATA_DIR, game_path)
    labels_file = os.path.join(game_dir, "Labels-v2.json")
    if not os.path.exists(labels_file):
        return
    
    with open(labels_file, "r") as f:
        data = json.load(f)
        
    for half in ["1", "2"]:
        video_file = os.path.join(game_dir, f"{half}_224p.mkv")
        if not os.path.exists(video_file):
            continue
            
        events = [a for a in data["annotations"] if a["gameTime"].startswith(half + " - ")]
        events.sort(key=lambda x: int(x["position"]))
        
        setp_pos = []
        setp_hard_neg = []
        all_events = []
        
        for e in events:
            t = int(e["position"]) / 1000.0
            label = e["label"]
            all_events.append(t)
            
            # SET-PIECE POSITIVES
            if label in ["Direct free-kick", "Penalty"]:
                setp_pos.append(t)
            elif label in ["Corner", "Indirect free-kick", "Throw-in", "Goal kick", "Kick-off", "Offside"]:
                setp_hard_neg.append(t)

        def generate_bg(num_samples, existing_events):
            bg = []
            max_duration_seconds = 45 * 60
            attempts = 0
            while len(bg) < num_samples and attempts < num_samples * 10:
                attempts += 1
                random_t = random.uniform(10, max_duration_seconds - 10)
                overlap = any(abs(random_t - e_t) < 30 for e_t in existing_events)
                if not overlap:
                    bg.append(random_t)
            return bg
            
        # Target ratios: 1 Positive : 2 Hard Negative : 5 Backgrounds
        target_setp = len(setp_pos)
        sampled_setp_hn = random.sample(setp_hard_neg, min(target_setp * 2, len(setp_hard_neg)))
        sampled_setp_bg = generate_bg(target_setp * 5, all_events)

        prefix_base = game_dir.replace("/", "_").replace("soccernet_data_", "") + f"_h{half}"
        
        def extract(t, out_vid):
            # Asymmetric 15-second window focused on PRE-KICK BUILDUP: [-10s, +5s]
            t_start = max(0, t - 10.0)
            
            if out_vid and not os.path.exists(out_vid):
                os.makedirs(os.path.dirname(out_vid), exist_ok=True)
                # Video extraction: 2 FPS, 15 seconds = 30 frames.
                subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", 
                                "-ss", str(t_start), "-i", video_file, "-t", "15", 
                                "-r", "2", "-vf", "scale=-2:224", 
                                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", out_vid])

        for t in setp_pos:
            extract(t, os.path.join(OUT_DIR_SETP, split, "set_piece", f"{prefix_base}_{int(t)}s.mp4"))
        for t in sampled_setp_hn:
            extract(t, os.path.join(OUT_DIR_SETP, split, "background", f"{prefix_base}_hn_{int(t)}s.mp4"))
        for t in sampled_setp_bg:
            extract(t, os.path.join(OUT_DIR_SETP, split, "background", f"{prefix_base}_bg_{int(t)}s.mp4"))

def main():
    if os.path.exists(OUT_DIR_SETP): 
        shutil.rmtree(OUT_DIR_SETP)
        
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUT_DIR_SETP, split, "set_piece"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR_SETP, split, "background"), exist_ok=True)
    
    train_games = getListGames('train')
    val_games = getListGames('valid')
    tasks = [(g, 'train') for g in train_games] + [(g, 'val') for g in val_games]
    
    print(f"Found {len(tasks)} match tasks for Set-Piece extraction (train and val splits).")
    print(f"Target window: [-10s, +5s] (pre-kick buildup: 15s duration @ 2 FPS = 30 frames)")
    print(f"Extracting to '{OUT_DIR_SETP}'...")
    
    pool = multiprocessing.Pool(processes=20)
    pool.map(process_game, tasks)
    pool.close()
    pool.join()
    print("Done extracting Set-Piece V4 ([-10s, +5s] / 30-frame) dataset!")

if __name__ == "__main__":
    main()
