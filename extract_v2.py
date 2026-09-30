import os
import json
import random
import multiprocessing
import subprocess
import shutil
from SoccerNet.utils import getListGames

DATA_DIR = "soccernet_data"
OUT_DIR_FOUL = "foul_dataset_v2"
OUT_DIR_SETP = "setpiece_dataset_v2"
OUT_DIR_AUDIO = "audio_dataset_v2"

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
        
        foul_pos = []
        foul_hard_neg = []
        
        setp_pos = []
        setp_hard_neg = []
        
        audio_pos = []
        
        for e in events:
            t = int(e["position"]) / 1000.0
            label = e["label"]
            
            # AUDIO POSITIVES (ANY event with a whistle)
            if label in ["Foul", "Yellow card", "Red card", "Yellow->red card", 
                         "Direct free-kick", "Penalty", "Offside", "Kick-off", "Goal", "Substitution"]:
                audio_pos.append(t)
            
            # FOUL
            if label in ["Foul", "Yellow card", "Red card", "Yellow->red card"]:
                foul_pos.append(t)
            elif label in ["Clearance", "Shots on target", "Shots off target", "Ball out of play", "Offside", "Corner", "Throw-in", "Goal kick", "Substitution", "Goal"]:
                foul_hard_neg.append(t)
                
            # SET-PIECE
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
                random_t = random.uniform(5, max_duration_seconds - 5)
                overlap = any(abs(random_t - e_t) < 30 for e_t in existing_events)
                if not overlap:
                    bg.append(random_t)
            return bg
            
        # Target ratios:
        # Foul/Set-piece: 1 Positive : 2 Hard Negative : 5 Backgrounds
        
        # FOUL SAMPLING
        target_foul = len(foul_pos)
        sampled_foul_hn = random.sample(foul_hard_neg, min(target_foul * 2, len(foul_hard_neg)))
        sampled_foul_bg = generate_bg(target_foul * 5, [t for t in foul_pos + foul_hard_neg])
        
        # SET-PIECE SAMPLING
        target_setp = len(setp_pos)
        sampled_setp_hn = random.sample(setp_hard_neg, min(target_setp * 2, len(setp_hard_neg)))
        sampled_setp_bg = generate_bg(target_setp * 5, [t for t in setp_pos + setp_hard_neg])
        
        # AUDIO SAMPLING (1 Positive : 3 Backgrounds)
        target_audio = len(audio_pos)
        sampled_audio_bg = generate_bg(target_audio * 3, audio_pos)

        prefix_base = game_dir.replace("/", "_").replace("soccernet_data_", "") + f"_h{half}"
        
        def extract(t, out_vid, out_aud):
            t_start = max(0, t - 5) # 10 second window
            
            if out_vid and not os.path.exists(out_vid):
                # Video extraction: 2 FPS, 10 seconds = 20 frames. Web-friendly encoding just in case.
                subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", 
                                "-ss", str(t_start), "-i", video_file, "-t", "10", 
                                "-r", "2", "-vf", "scale=-2:224", 
                                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", out_vid])
                                
            if out_aud and not os.path.exists(out_aud):
                # Audio extraction: 10 seconds, 16kHz mono WAV
                subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", 
                                "-ss", str(t_start), "-i", video_file, "-t", "10", 
                                "-map", "0:a:0", # Ensure we grab the audio stream
                                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_aud])

        
        # EXECUTIONS
        for t in foul_pos:
            extract(t, os.path.join(OUT_DIR_FOUL, split, "foul", f"{prefix_base}_{int(t)}s.mp4"), None)
        for t in sampled_foul_hn:
            extract(t, os.path.join(OUT_DIR_FOUL, split, "background", f"{prefix_base}_hn_{int(t)}s.mp4"), None)
        for t in sampled_foul_bg:
            extract(t, os.path.join(OUT_DIR_FOUL, split, "background", f"{prefix_base}_bg_{int(t)}s.mp4"), None)
            
        for t in setp_pos:
            extract(t, os.path.join(OUT_DIR_SETP, split, "set_piece", f"{prefix_base}_{int(t)}s.mp4"), None)
        for t in sampled_setp_hn:
            extract(t, os.path.join(OUT_DIR_SETP, split, "background", f"{prefix_base}_hn_{int(t)}s.mp4"), None)
        for t in sampled_setp_bg:
            extract(t, os.path.join(OUT_DIR_SETP, split, "background", f"{prefix_base}_bg_{int(t)}s.mp4"), None)
            
        for t in audio_pos:
            extract(t, None, os.path.join(OUT_DIR_AUDIO, split, "whistle", f"{prefix_base}_{int(t)}s.wav"))
        for t in sampled_audio_bg:
            extract(t, None, os.path.join(OUT_DIR_AUDIO, split, "background", f"{prefix_base}_bg_{int(t)}s.wav"))


def main():
    for d in [OUT_DIR_FOUL, OUT_DIR_SETP, OUT_DIR_AUDIO]:
        if os.path.exists(d): shutil.rmtree(d)
        
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUT_DIR_FOUL, split, "foul"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR_FOUL, split, "background"), exist_ok=True)
        
        os.makedirs(os.path.join(OUT_DIR_SETP, split, "set_piece"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR_SETP, split, "background"), exist_ok=True)
        
        os.makedirs(os.path.join(OUT_DIR_AUDIO, split, "whistle"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR_AUDIO, split, "background"), exist_ok=True)
    
    train_games = getListGames('train')
    val_games = getListGames('valid')
    
    tasks = [(g, 'train') for g in train_games] + [(g, 'val') for g in val_games]
    
    print(f"Found {len(tasks)} tasks to process (train and val splits).")
    
    pool = multiprocessing.Pool(processes=8)
    pool.map(process_game, tasks)
    pool.close()
    pool.join()
    print("Done extracting V2 datasets!")

if __name__ == "__main__":
    main()
