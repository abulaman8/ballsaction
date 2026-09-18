import os
import json
import random
import subprocess
import torchaudio
import matplotlib.pyplot as plt
from SoccerNet.utils import getListGames

DATA_DIR = "/home/pilot/Desktop/ballsaction/soccernet_data"
OUTPUT_DIR = "/home/pilot/Desktop/ballsaction/audio_dataset"
WHISTLE_CLASSES = ["Direct free-kick", "Indirect free-kick", "Penalty", "Kick-off", "Foul", "Offside"]

os.makedirs(os.path.join(OUTPUT_DIR, "train", "whistle"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train", "background"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val", "whistle"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val", "background"), exist_ok=True)

def extract_and_plot(video_path, start_sec, dur, out_png):
    wav_path = "/tmp/temp_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", 
        "-ss", str(start_sec), "-i", video_path, 
        "-t", str(dur), "-q:a", "0", "-map", "a", wav_path
    ]
    subprocess.run(cmd)
    
    if not os.path.exists(wav_path):
        return False
        
    try:
        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=2048, hop_length=512, n_mels=128
        )(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        plt.figure(figsize=(2.24, 2.24), dpi=100) # 224x224 pixels
        plt.imshow(mel_spec[0].numpy(), aspect='auto', origin='lower', cmap='magma')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
        plt.close()
        os.remove(wav_path)
        return True
    except Exception as e:
        return False

def main():
    games = getListGames(["train", "valid", "test"])
    downloaded = [g for g in games if os.path.exists(os.path.join(DATA_DIR, g, "Labels-v2.json"))]
    
    print(f"Found {len(downloaded)} matches for Audio Extraction...")
    
    for game_idx, game in enumerate(downloaded):
        split = "val" if game_idx % 5 == 0 else "train"
        labels_path = os.path.join(DATA_DIR, game, "Labels-v2.json")
        
        with open(labels_path, 'r') as f:
            labels = json.load(f)
            
        intervals = []
        whistle_count = 0
        bg_count = 0
        
        for ann in labels["annotations"]:
            label = ann["label"]
            half = ann["gameTime"].split(" - ")[0]
            if half not in ["1", "2"]: continue
            
            m, s = map(int, ann["gameTime"].split(" - ")[1].split(':'))
            time_sec = m * 60 + s
            
            # 5 second window (2s before, 3s after)
            start_sec = max(0, time_sec - 2)
            end_sec = start_sec + 5
            
            video_path = os.path.join(DATA_DIR, game, f"{half}_224p.mkv")
            if not os.path.exists(video_path): continue
            
            if label in WHISTLE_CLASSES:
                png_name = f"{game.replace('/', '_')}_{half}_{time_sec}.jpg"
                out_path = os.path.join(OUTPUT_DIR, split, "whistle", png_name)
                if extract_and_plot(video_path, start_sec, 5, out_path):
                    whistle_count += 1
                    intervals.append((half, start_sec, end_sec))
                    
        # Extract Backgrounds (1:1 ratio)
        attempts = 0
        while bg_count < whistle_count and attempts < whistle_count * 3:
            attempts += 1
            half = random.choice(["1", "2"])
            video_path = os.path.join(DATA_DIR, game, f"{half}_224p.mkv")
            if not os.path.exists(video_path): continue
            
            start_sec = random.randint(0, 45*60 - 5)
            end_sec = start_sec + 5
            
            overlap = any(h == half and not (end_sec <= s or start_sec >= e) for h, s, e in intervals)
            if not overlap:
                png_name = f"{game.replace('/', '_')}_{half}_bg_{start_sec}.jpg"
                out_path = os.path.join(OUTPUT_DIR, split, "background", png_name)
                if extract_and_plot(video_path, start_sec, 5, out_path):
                    bg_count += 1
                    intervals.append((half, start_sec, end_sec))
                    
        print(f"[{game_idx+1}/{len(downloaded)}] {game} | Whistles: {whistle_count} | BGs: {bg_count}")

if __name__ == "__main__":
    main()
