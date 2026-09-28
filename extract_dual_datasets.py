import os
import json
import random
import multiprocessing
import subprocess
from glob import glob

DATA_DIR = "soccernet_data"
OUT_DIR_FOUL = "foul_dataset"
OUT_DIR_SETP = "setpiece_dataset"

def process_game(game_dir):
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
        
        # Sort events by time
        events.sort(key=lambda x: int(x["position"]))
        
        foul_count = 0
        setp_count = 0
        hard_neg_count = 0
        bg_foul_count = 0
        
        for e in events:
            time_ms = int(e["position"])
            t = time_ms / 1000.0
            label = e["label"]
            
            # Prefix for unique ID
            prefix = game_dir.replace("/", "_").replace("soccernet_data_", "") + f"_h{half}_{int(t)}s"
            
            # Foul Dataset: Fouls (Positive)
            if label == "Foul":
                out_vid = os.path.join(OUT_DIR_FOUL, "train", "foul", f"{prefix}.mp4")
                out_aud = os.path.join(OUT_DIR_FOUL, "train", "foul", f"{prefix}.wav")
                if not os.path.exists(out_vid):
                    subprocess.run(f"ffmpeg -y -hide_banner -loglevel error -ss {t-5} -i '{video_file}' -t 10 -r 4 -c:v libx264 -preset ultrafast -crf 23 '{out_vid}'", shell=True)
                if not os.path.exists(out_aud):
                    subprocess.run(f"ffmpeg -y -hide_banner -loglevel error -ss {t} -i '{video_file}' -t 5 -vn -acodec pcm_s16le -ar 16000 -ac 1 '{out_aud}'", shell=True)
                foul_count += 1
                
            # Set-Piece Dataset: Direct FK, Penalty (Positive)
            elif label in ["Direct free-kick", "Penalty"]:
                out_vid = os.path.join(OUT_DIR_SETP, "train", "set_piece", f"{prefix}.mp4")
                out_aud = os.path.join(OUT_DIR_SETP, "train", "set_piece", f"{prefix}.wav")
                if not os.path.exists(out_vid):
                    subprocess.run(f"ffmpeg -y -hide_banner -loglevel error -ss {t-5} -i '{video_file}' -t 10 -r 4 -c:v libx264 -preset ultrafast -crf 23 '{out_vid}'", shell=True)
                if not os.path.exists(out_aud):
                    subprocess.run(f"ffmpeg -y -hide_banner -loglevel error -ss {max(0, t-5)} -i '{video_file}' -t 5 -vn -acodec pcm_s16le -ar 16000 -ac 1 '{out_aud}'", shell=True)
                setp_count += 1
                
            # Set-Piece Dataset: Throw-in, Corner, Goal kick (Hard Negative)
            elif label in ["Throw-in", "Corner", "Goal kick"]:
                out_vid = os.path.join(OUT_DIR_SETP, "train", "background", f"{prefix}.mp4")
                out_aud = os.path.join(OUT_DIR_SETP, "train", "background", f"{prefix}.wav")
                if not os.path.exists(out_vid):
                    subprocess.run(f"ffmpeg -y -hide_banner -loglevel error -ss {t-5} -i '{video_file}' -t 10 -r 4 -c:v libx264 -preset ultrafast -crf 23 '{out_vid}'", shell=True)
                if not os.path.exists(out_aud):
                    subprocess.run(f"ffmpeg -y -hide_banner -loglevel error -ss {max(0, t-5)} -i '{video_file}' -t 5 -vn -acodec pcm_s16le -ar 16000 -ac 1 '{out_aud}'", shell=True)
                hard_neg_count += 1

        # Generate Random Background for Fouls
        # We need roughly equal number of backgrounds as fouls
        num_fouls = foul_count
        bg_samples = 0
        max_duration_seconds = 45 * 60 # Assume roughly 45 min per half
        attempts = 0
        while bg_samples < num_fouls and attempts < num_fouls * 10:
            attempts += 1
            random_t = random.uniform(5, max_duration_seconds - 5)
            
            # Check if it overlaps with ANY event in the labels
            overlap = False
            for e in events:
                e_t = int(e["position"]) / 1000.0
                if abs(random_t - e_t) < 10: # If within 10 seconds of any event, skip
                    overlap = True
                    break
            
            if not overlap:
                prefix = game_dir.replace("/", "_").replace("soccernet_data_", "") + f"_h{half}_bg_{int(random_t)}s"
                out_vid = os.path.join(OUT_DIR_FOUL, "train", "background", f"{prefix}.mp4")
                out_aud = os.path.join(OUT_DIR_FOUL, "train", "background", f"{prefix}.wav")
                if not os.path.exists(out_vid):
                    subprocess.run(f"ffmpeg -y -hide_banner -loglevel error -ss {random_t-5} -i '{video_file}' -t 10 -r 4 -c:v libx264 -preset ultrafast -crf 23 '{out_vid}'", shell=True)
                if not os.path.exists(out_aud):
                    subprocess.run(f"ffmpeg -y -hide_banner -loglevel error -ss {random_t} -i '{video_file}' -t 5 -vn -acodec pcm_s16le -ar 16000 -ac 1 '{out_aud}'", shell=True)
                bg_samples += 1

def main():
    os.makedirs(os.path.join(OUT_DIR_FOUL, "train", "foul"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR_FOUL, "train", "background"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR_SETP, "train", "set_piece"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR_SETP, "train", "background"), exist_ok=True)
    
    games = glob("soccernet_data/*/*/*")
    print(f"Found {len(games)} games to process.")
    
    # Process using multiprocessing
    pool = multiprocessing.Pool(processes=8)
    pool.map(process_game, games)
    pool.close()
    pool.join()
    print("Done extracting datasets!")

if __name__ == "__main__":
    main()
