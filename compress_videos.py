import os
import glob
import subprocess
import time

INPUT_DIR = "/home/pilot/Desktop/ballsaction/custom_data_2"
OUTPUT_DIR = "/home/pilot/Desktop/ballsaction/custom_data_2_compressed"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    videos = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    
    if not videos:
        print(f"No videos found in {INPUT_DIR}")
        return
        
    print(f"Found {len(videos)} videos. Starting compression...")
    
    for idx, video_path in enumerate(videos):
        filename = os.path.basename(video_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Skip if already compressed
        if os.path.exists(output_path):
            print(f"[{idx+1}/{len(videos)}] Skipping {filename} - already exists.")
            continue
            
        print(f"\n[{idx+1}/{len(videos)}] Compressing {filename}...")
        start_t = time.time()
        
        # ffmpeg command for visually lossless compression using GPU hardware acceleration (NVENC):
        # -map 0:v:0 -> takes only the first video track
        # -map 0:a:0 -> takes only the first audio track (drops extra commentary/languages)
        # -c:v h264_nvenc -> uses NVIDIA GPU hardware encoder for lightning fast compression
        # -cq 19 -preset p6 -> Constant Quality 19 (visually lossless) with high-quality preset
        # -c:a aac -b:a 128k -> compress audio track efficiently
        cmd = [
            "ffmpeg", "-y", 
            "-i", video_path,
            "-map", "0:v:0", "-map", "0:a:0?", 
            "-c:v", "h264_nvenc", "-cq", "19", "-preset", "p6",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]
        
        try:
            # We don't suppress output entirely so you can see ffmpeg progress if run interactively,
            # but we run it via subprocess to keep python in control.
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start_t
            
            orig_size = os.path.getsize(video_path) / (1024**3)
            new_size = os.path.getsize(output_path) / (1024**3)
            
            print(f"Done in {elapsed/60:.1f} minutes!")
            print(f"Size Reduction: {orig_size:.2f} GB -> {new_size:.2f} GB")
            
            # Delete original file
            os.remove(video_path)
            print(f"Deleted original file: {filename}")
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to compress {filename}. Error: {e}")

if __name__ == "__main__":
    main()
