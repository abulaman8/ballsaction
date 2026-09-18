import os
import json
import cv2
import numpy as np

def extract_soccernet_frames(data_dir, output_dir, target_fps=2, padding_sec=30, target_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    
    # Target classes specifically for SoccerNet labels
    positive_classes = {"Foul", "Penalty", "Direct free-kick", "Indirect free-kick"}
    hard_negative_classes = {"Tackle", "Clearance", "Shots on target", "Shots off target", "Corner"}
    
    # Recursively find all Labels-v2.json files
    json_paths = []
    for root, dirs, files in os.walk(data_dir):
        if "Labels-v2.json" in files:
            json_paths.append(os.path.join(root, "Labels-v2.json"))
            
    print(f"Found {len(json_paths)} annotation files in {data_dir}.")
    
    total_processed = 0
    
    for json_path in json_paths:
        game_dir = os.path.dirname(json_path)
        # Create a safe name for the game based on its folder structure (e.g. germany_bundesliga_2016-2017_...)
        game_name = os.path.relpath(game_dir, data_dir).replace("/", "_").replace("\\", "_")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        annotations = data.get('annotations', [])
        if not annotations:
            continue
            
        # Group by half to minimize video re-opening
        half_annotations = {1: [], 2: []}
        for ann in annotations:
            label = ann.get('label')
            game_time = ann.get('gameTime') # Format: "1 - 10:15"
            
            if not label or not game_time or " - " not in game_time:
                continue
                
            target_class = None
            if label in positive_classes:
                target_class = label
            elif label in hard_negative_classes:
                target_class = "None"
                
            if target_class:
                parts = game_time.split(' - ')
                half = int(parts[0])
                time_str = parts[1]
                mins, secs = map(int, time_str.split(':'))
                total_seconds = mins * 60 + secs
                
                if half in half_annotations:
                    half_annotations[half].append({
                        "label": target_class,
                        "time": total_seconds,
                        "original_label": label
                    })
                    
        # Extract from the video files for this game
        for half, events in half_annotations.items():
            if not events:
                continue
                
            video_path = os.path.join(game_dir, f"{half}_224p.mkv")
            if not os.path.exists(video_path):
                # Fallback to 720p if 224p doesn't exist
                video_path = os.path.join(game_dir, f"{half}_720p.mkv")
                
            if not os.path.exists(video_path):
                print(f"Warning: Video for half {half} not found in {game_dir}")
                continue
                
            print(f"\nProcessing {video_path} ({len(events)} targeted events)...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open {video_path}")
                continue
                
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if source_fps <= 0:
                source_fps = 25.0
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_step = max(1, int(source_fps / target_fps))
            padding_frames = int(padding_sec * source_fps)
            
            for i, event in enumerate(events):
                # SoccerNet provides one timestamp (the anchor)
                # We window 30 seconds before and 30 seconds after
                center_frame = int(event["time"] * source_fps)
                start_frame = max(0, center_frame - padding_frames)
                end_frame = min(total_frames - 1, center_frame + padding_frames)
                
                safe_class = event["label"].replace(" ", "_").replace("-", "_")
                sample_id = f"{game_name}_half{half}_{i:04d}_{safe_class}"
                
                out_path = os.path.join(output_dir, f"{sample_id}.npy")
                if os.path.exists(out_path):
                    continue # Skip if already processed
                
                # Seek to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                clip_frames = []
                current_frame = start_frame
                
                while current_frame <= end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if (current_frame - start_frame) % frame_step == 0:
                        resized = cv2.resize(frame, target_size)
                        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        clip_frames.append(resized)
                        
                    current_frame += 1
                    
                if len(clip_frames) > 0:
                    clip_array = np.array(clip_frames, dtype=np.uint8)
                    np.save(out_path, clip_array)
                    total_processed += 1
                    print(f"  Saved {sample_id}.npy -> {clip_array.shape}")
                    
            cap.release()
            
    print(f"\nSoccerNet extraction complete! Processed {total_processed} clips.")

if __name__ == "__main__":
    # Source directory for SoccerNet downloads
    data_dir = "/home/pilot/Desktop/ballsaction/soccernet_data"
    # Target directory for extracted numpy arrays
    output_dir = "/home/pilot/Desktop/ballsaction/soccernet_extracted_clips"
    
    extract_soccernet_frames(data_dir, output_dir)
