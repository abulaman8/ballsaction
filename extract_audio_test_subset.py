import os
import json
import csv
import subprocess
from SoccerNet.utils import getListGames

DATA_DIR = "/home/pilot/Desktop/ballsaction/soccernet_data"
OUTPUT_DIR = "/home/pilot/Desktop/ballsaction/audio_test_subset"
CSV_PATH = os.path.join(OUTPUT_DIR, "whistle_timestamps_test.csv")
WHISTLE_CLASSES = ["Direct free-kick", "Indirect free-kick", "Penalty", "Kick-off", "Foul", "Offside"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_audio(video_path, start_sec, dur, out_wav):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", 
        "-ss", str(start_sec), "-i", video_path, 
        "-t", str(dur), "-q:a", "0", "-map", "a", out_wav
    ]
    subprocess.run(cmd)
    return os.path.exists(out_wav)

def main():
    games = getListGames(["train"])
    downloaded = [g for g in games if os.path.exists(os.path.join(DATA_DIR, g, "Labels-v2.json"))]
    
    # We just do this for 3 games for testing
    test_games = downloaded[:3]
    
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Game", "Half", "Label", "Original Time (MM:SS)", "Extracted Start (Sec)", "Extracted End (Sec)", "Audio File"])
        
        for game in test_games:
            print(f"Processing {game}...")
            labels_path = os.path.join(DATA_DIR, game, "Labels-v2.json")
            
            with open(labels_path, 'r') as f_json:
                labels = json.load(f_json)
                
            for ann in labels["annotations"]:
                label = ann["label"]
                if label not in WHISTLE_CLASSES:
                    continue
                    
                half = ann["gameTime"].split(" - ")[0]
                if half not in ["1", "2"]: continue
                
                original_time = ann["gameTime"].split(" - ")[1]
                m, s = map(int, original_time.split(':'))
                time_sec = m * 60 + s
                
                # 4 second window (2s before, 2s after)
                start_sec = max(0, time_sec - 2)
                end_sec = start_sec + 4
                
                video_path = os.path.join(DATA_DIR, game, f"{half}_224p.mkv")
                if not os.path.exists(video_path): continue
                
                wav_name = f"{game.replace('/', '_')}_{half}_{time_sec}.wav"
                out_wav = os.path.join(OUTPUT_DIR, wav_name)
                
                if extract_audio(video_path, start_sec, 4, out_wav):
                    writer.writerow([game, half, label, original_time, start_sec, end_sec, wav_name])
                    
    print(f"\nDone! Extracted sample whistles and wrote timestamps to {CSV_PATH}")

if __name__ == "__main__":
    main()
