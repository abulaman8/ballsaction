import os
import json
import subprocess
import random
from datetime import timedelta
from SoccerNet.utils import getListGames

# Configuration
DATA_DIR = "/home/pilot/Desktop/ballsaction/soccernet_data"
OUTPUT_DIR = "/home/pilot/Desktop/ballsaction/x3d_dataset"
# We target a 1:4.5 imbalance (1 set piece for every 4.5 backgrounds)
BG_RATIO = 4.5

HIGHLIGHT_CLASSES = ["Direct free-kick", "Indirect free-kick", "Penalty"]
HARD_NEGATIVE_CLASSES = ["Kick-off", "Corner", "Throw-in", "Substitution", "Yellow card", "Red card", "Offside", "Foul"]
CLIP_DURATION = 20  # 10s before, 10s after
FPS = 2
NUM_FRAMES = CLIP_DURATION * FPS # 40

os.makedirs(os.path.join(OUTPUT_DIR, "train", "set_piece"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train", "background"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val", "set_piece"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val", "background"), exist_ok=True)

def parse_time(time_str):
    """Convert mm:ss to seconds"""
    m, s = map(int, time_str.split(':'))
    return m * 60 + s

def extract_clip(video_path, start_sec, out_path):
    """Extract a highly compressed 2 FPS MP4 clip using ffmpeg"""
    if os.path.exists(out_path):
        return True
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(CLIP_DURATION),
        "-vf", f"fps={FPS},scale=224:224",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-an", # No audio
        out_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def check_overlap(target_start, target_end, intervals):
    for start, end in intervals:
        if max(target_start, start) < min(target_end, end):
            return True
    return False

def main():
    print("Scanning downloaded games...")
    all_games = getListGames(["train", "valid", "test"])
    
    total_highlights_extracted = 0
    total_background_extracted = 0
    
    # We will process games that actually exist on disk
    downloaded_games = []
    for game in all_games:
        if os.path.exists(os.path.join(DATA_DIR, game, "Labels-v2.json")):
            downloaded_games.append(game)
            
    print(f"Found {len(downloaded_games)} games on disk. Beginning extraction...")
    
    for game_idx, game in enumerate(downloaded_games):
        split = "val" if game_idx % 5 == 0 else "train"
        game_path = os.path.join(DATA_DIR, game)
        labels_path = os.path.join(game_path, "Labels-v2.json")
        
        with open(labels_path, 'r') as f:
            labels = json.load(f)
            
        highlight_intervals = []
        hard_negative_intervals = []
        
        # First Pass: Extract Set Pieces and collect hard negatives
        for ann in labels["annotations"]:
            label = ann["label"]
            half = ann["gameTime"].split(" - ")[0]
            if half not in ["1", "2"]:
                continue
                
            time_sec = parse_time(ann["gameTime"].split(" - ")[-1])
            start_sec = max(0, time_sec - (CLIP_DURATION // 2))
            end_sec = start_sec + CLIP_DURATION
            
            video_file = f"{half}_224p.mkv"
            video_path = os.path.join(game_path, video_file)
            
            if not os.path.exists(video_path):
                continue
                
            if label in HIGHLIGHT_CLASSES:
                # Save as Set Piece
                clip_name = f"{game.replace('/', '_')}_{half}_{time_sec}.mp4"
                out_path = os.path.join(OUTPUT_DIR, split, "set_piece", clip_name)
                
                if extract_clip(video_path, start_sec, out_path):
                    total_highlights_extracted += 1
                    highlight_intervals.append((half, start_sec, end_sec))
            elif label in HARD_NEGATIVE_CLASSES:
                # Save interval for hard negative mining
                hard_negative_intervals.append((half, start_sec, end_sec))
                
        # Second Pass: Extract Background clips
        # Number of background clips to extract for this game
        num_bg_needed = int(len([iv for iv in highlight_intervals if iv[0] in ["1", "2"]]) * BG_RATIO)
        if num_bg_needed == 0:
            num_bg_needed = 10
            
        bg_extracted = 0
        attempts = 0
        
        # We want 60% of backgrounds to be hard negatives
        target_hn = int(num_bg_needed * 0.6)
        hn_extracted = 0
        
        random.shuffle(hard_negative_intervals)
        
        # Extract Hard Negatives first
        for (half, start_sec, end_sec) in hard_negative_intervals:
            if hn_extracted >= target_hn:
                break
                
            # Make sure this hard negative doesn't overlap with a highlight
            overlap = False
            for (h_half, h_start, h_end) in highlight_intervals:
                if half == h_half and not (end_sec <= h_start or start_sec >= h_end):
                    overlap = True
                    break
                    
            if not overlap:
                video_file = f"{half}_224p.mkv"
                video_path = os.path.join(game_path, video_file)
                
                if not os.path.exists(video_path):
                    continue
                    
                clip_name = f"{game.replace('/', '_')}_{half}_bg_hn_{start_sec}.mp4"
                out_path = os.path.join(OUTPUT_DIR, split, "background", clip_name)
                
                if extract_clip(video_path, start_sec, out_path):
                    bg_extracted += 1
                    hn_extracted += 1
                    total_background_extracted += 1
                    # Treat this hard negative as an interval to avoid for open play
                    highlight_intervals.append((half, start_sec, end_sec))
                    
        # Fill the rest with Open Play
        while bg_extracted < num_bg_needed and attempts < num_bg_needed * 3:
            attempts += 1
            half = random.choice(["1", "2"])
            video_file = f"{half}_224p.mkv"
            video_path = os.path.join(game_path, video_file)
            
            if not os.path.exists(video_path):
                continue
                
            # Random start time between 0 and 45 minutes
            start_sec = random.randint(0, 45 * 60 - CLIP_DURATION)
            end_sec = start_sec + CLIP_DURATION
            
            overlap = False
            for (h_half, h_start, h_end) in highlight_intervals:
                if half == h_half and not (end_sec <= h_start or start_sec >= h_end):
                    overlap = True
                    break
                    
            if not overlap:
                clip_name = f"{game.replace('/', '_')}_{half}_bg_open_{start_sec}.mp4"
                out_path = os.path.join(OUTPUT_DIR, split, "background", clip_name)
                
                if extract_clip(video_path, start_sec, out_path):
                    bg_extracted += 1
                    total_background_extracted += 1
                    highlight_intervals.append((half, start_sec, end_sec))
                    
        print(f"[{game_idx+1}/{len(downloaded_games)}] {game} | Highlights: {total_highlights_extracted} | Backgrounds: {bg_extracted} ({hn_extracted} HN)")
        
    print(f"\nEXTRACTION COMPLETE.")
    print(f"Total Highlights Extracted: {total_highlights_extracted}")
    print(f"Total Background Extracted: {total_background_extracted}")

if __name__ == "__main__":
    main()
