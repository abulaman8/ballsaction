import os
import glob
import subprocess
import multiprocessing

def extract_audio(mp4_path):
    wav_path = mp4_path.replace(".mp4", ".wav")
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
        return
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-hide_banner",
        "-i", mp4_path, "-vn", "-ar", "16000", "-ac", "1", wav_path
    ])

def main():
    clips = glob.glob("setpiece_dataset_v4/**/*.mp4", recursive=True)
    print(f"Found {len(clips)} set-piece clips in setpiece_dataset_v4.")
    print("Extracting 16kHz mono WAV files in parallel...")
    with multiprocessing.Pool(processes=20) as pool:
        pool.map(extract_audio, clips)
    print("Done extracting all WAV files for setpiece_dataset_v4!")

if __name__ == "__main__":
    main()
