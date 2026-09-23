import os
import glob
import subprocess
import time

def compress_videos():
    target_dir = "/home/pilot/Desktop/ballsaction/custom_data_2_compressed"
    videos = glob.glob(os.path.join(target_dir, "*.mp4")) + glob.glob(os.path.join(target_dir, "*.mkv"))
    
    for vid in videos:
        if vid.endswith("_480p.mp4"):
            continue
            
        print(f"[{time.strftime('%H:%M:%S')}] Compressing {os.path.basename(vid)}...")
        out_name = os.path.splitext(vid)[0] + "_480p.mp4"
        
        # Use NVIDIA GPU Hardware Encoding (HEVC/H.265) for maximum compression and speed
        # scale=-2:480 keeps the aspect ratio while scaling to 480p height
        # -cq 28 provides excellent compression for 480p
        cmd = [
            "/usr/bin/ffmpeg", "-y", "-loglevel", "error",
            "-i", vid,
            "-vf", "scale=-2:480",
            "-c:v", "hevc_nvenc",
            "-preset", "p6",
            "-cq", "28",
            "-c:a", "aac",
            "-b:a", "128k",
            out_name
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f" -> Success! Output saved to {os.path.basename(out_name)}.")
            # Delete original as requested
            try:
                os.remove(vid)
                print(f" -> Deleted original file {os.path.basename(vid)}.")
            except Exception as e:
                print(f" -> Error deleting {vid}: {e}")
        else:
            print(f" -> Error compressing {vid}! FFmpeg returned {result.returncode}.")

if __name__ == "__main__":
    compress_videos()
    print("\nCompression batch finished!")
