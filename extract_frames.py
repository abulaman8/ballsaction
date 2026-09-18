import os
import csv
import cv2
import numpy as np
import argparse

def extract_dataset(manifest_csv, data_dir, output_dir, target_fps=2, padding_sec=30, target_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    
    # Read manifest
    with open(manifest_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        samples = list(reader)
        
    print(f"Loaded {len(samples)} samples from manifest.")
    
    # Group samples by video to optimize I/O (we only open each massive video once)
    videos = {}
    for i, row in enumerate(samples):
        v = row['Video_File']
        if v not in videos:
            videos[v] = []
        import re
        # Create a unique ID for the file, stripping out invalid filename characters like '/'
        safe_class = re.sub(r'[^a-zA-Z0-9_]', '_', row['Target_Class'].replace(" ", "_"))
        row['sample_id'] = f"clip_{i:04d}_{safe_class}"
        videos[v].append(row)
        
    total_processed = 0
    for video_file, clips in videos.items():
        video_path = os.path.join(data_dir, video_file)
        if not os.path.exists(video_path):
            print(f"Skipping {video_file}, not found in {data_dir}.")
            continue
            
        print(f"\nProcessing {video_file} ({len(clips)} clips)...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue
            
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 25.0 # sensible fallback for European broadcast
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(source_fps / target_fps))
        padding_frames = int(padding_sec * source_fps)
        
        for clip in clips:
            frame_in = int(clip['Frame_In'])
            frame_out = int(clip['Frame_Out'])
            
            # Capture from 30s before the action starts, to 30s after the action ends
            start_frame = max(0, frame_in - padding_frames)
            end_frame = min(total_frames - 1, frame_out + padding_frames)
            
            # Seek to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            clip_frames = []
            current_frame = start_frame
            
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Downsample temporally
                if (current_frame - start_frame) % frame_step == 0:
                    # Resize spatially
                    resized = cv2.resize(frame, target_size)
                    # Convert BGR (OpenCV default) to RGB (PyTorch standard)
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    clip_frames.append(resized)
                    
                current_frame += 1
                
            # Save the clip as a highly optimized compressed NumPy array
            if len(clip_frames) > 0:
                clip_array = np.array(clip_frames, dtype=np.uint8)
                # Shape will be (Time, Height, Width, Channels)
                out_path = os.path.join(output_dir, f"{clip['sample_id']}.npy")
                np.save(out_path, clip_array)
                total_processed += 1
                print(f"  Saved {out_path} with shape {clip_array.shape}")
            else:
                print(f"  Warning: No frames extracted for {clip['sample_id']}")
                
        cap.release()
        
    print(f"\nExtraction complete! Processed and saved {total_processed} clips as .npy arrays.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract, resize, and save video clips.")
    parser.add_argument('--manifest', type=str, default="/home/pilot/Desktop/ballsaction/custom_extraction_manifest.csv")
    parser.add_argument('--data_dir', type=str, default="/home/pilot/Desktop/ballsaction/custom_data")
    parser.add_argument('--output_dir', type=str, default="/home/pilot/Desktop/ballsaction/extracted_numpy_clips")
    args = parser.parse_args()
    
    extract_dataset(args.manifest, args.data_dir, args.output_dir)
