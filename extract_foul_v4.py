import os
import json
import random
import multiprocessing
import subprocess
import shutil
from SoccerNet.utils import getListGames

DATA_DIR = "soccernet_data"
OUT_DIR_FOUL = "foul_dataset_v4"

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
            
        events = [a for a in data.get("annotations", []) if a.get("gameTime", "").startswith(half + " - ")]
        events.sort(key=lambda x: int(x["position"]))
        
        fouls_list = []
        cards_list = []
        foul_hard_neg = []
        all_event_times = []
        
        for e in events:
            t = int(e["position"]) / 1000.0
            label = e["label"]
            all_event_times.append(t)
            
            if label == "Foul":
                fouls_list.append(t)
            elif label in ["Yellow card", "Red card", "Yellow->red card"]:
                cards_list.append(t)
            elif label in ["Clearance", "Shots on target", "Shots off target", "Ball out of play", "Offside", "Corner", "Throw-in", "Goal kick", "Substitution", "Goal"]:
                foul_hard_neg.append(t)
                
        # 1. Collect true physical fouls
        foul_pos = list(fouls_list)
        
        # 2. Realign Cards: If a card has an explicit foul in the preceding 60s, do NOT extract
        # a duplicate clip at the card time (the tackle was already captured at t_foul).
        # If it is a lone card with no foul stamp, anchor 7.0s earlier (median delay)
        # to capture the tackle that caused the card.
        for c_t in cards_list:
            has_preceding_foul = any(0 <= c_t - f_t <= 60.0 for f_t in fouls_list)
            if not has_preceding_foul:
                estimated_foul_t = max(5.0, c_t - 7.0)
                foul_pos.append(estimated_foul_t)
                
        # Deduplicate any timestamps within 3.0s of each other
        foul_pos = sorted(foul_pos)
        deduped_pos = []
        for t in foul_pos:
            if not deduped_pos or t - deduped_pos[-1] > 3.0:
                deduped_pos.append(t)
        foul_pos = deduped_pos

        # Generate balanced random background clips with 30s exclusion zone
        def generate_bg(num_samples, existing_events):
            bg = []
            max_duration_seconds = 45 * 60
            attempts = 0
            while len(bg) < num_samples and attempts < num_samples * 15:
                attempts += 1
                random_t = random.uniform(5, max_duration_seconds - 15)
                overlap = any(abs(random_t - e_t) < 30 for e_t in existing_events)
                if not overlap:
                    bg.append(random_t)
            return bg

        # 1 positive : 2 hard negatives : 2 random backgrounds
        n_pos = len(foul_pos)
        sampled_foul_hn = random.sample(foul_hard_neg, min(len(foul_hard_neg), n_pos * 2)) if foul_hard_neg else []
        sampled_foul_bg = generate_bg(n_pos * 2, all_event_times)
        
        prefix_base = f"{game_path.replace('/', '_')}_h{half}"
        
        def extract(t, out_base_path):
            # Window: [-5.0s, +10.0s] (15s duration @ 2 FPS = 30 frames)
            t_start = max(0.0, t - 5.0)
            out_vid = out_base_path
            out_wav = out_base_path.replace(".mp4", ".wav")
            
            if os.path.exists(out_vid) and os.path.exists(out_wav) and os.path.getsize(out_vid) > 1000 and os.path.getsize(out_wav) > 1000:
                return
                
            # Dual extraction: video and audio simultaneously
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(t_start), "-t", "15", "-i", video_file,
                "-r", "2", "-vf", "scale=-2:224", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", out_vid,
                "-vn", "-ar", "16000", "-ac", "1", out_wav
            ])

        for t in foul_pos:
            extract(t, os.path.join(OUT_DIR_FOUL, split, "foul", f"{prefix_base}_{int(t)}s.mp4"))
        for t in sampled_foul_hn:
            extract(t, os.path.join(OUT_DIR_FOUL, split, "background", f"{prefix_base}_hn_{int(t)}s.mp4"))
        for t in sampled_foul_bg:
            extract(t, os.path.join(OUT_DIR_FOUL, split, "background", f"{prefix_base}_bg_{int(t)}s.mp4"))

def main():
    if os.path.exists(OUT_DIR_FOUL):
        shutil.rmtree(OUT_DIR_FOUL)
        
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUT_DIR_FOUL, split, "foul"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR_FOUL, split, "background"), exist_ok=True)
        
    train_games = getListGames('train')
    val_games = getListGames('valid')
    tasks = [(g, 'train') for g in train_games] + [(g, 'val') for g in val_games]
    
    print(f"Found {len(tasks)} match tasks for Realigned Foul V4 extraction.")
    print("Window: [-5s, +10s] (15s duration @ 2 FPS = 30 frames, 16kHz mono audio).")
    print(f"Extracting to '{OUT_DIR_FOUL}'...")
    
    with multiprocessing.Pool(processes=20) as pool:
        pool.map(process_game, tasks)
        
    print("Done extracting Realigned Foul V4 dataset!")

if __name__ == "__main__":
    main()
