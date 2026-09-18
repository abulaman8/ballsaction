import os
import json
import csv

def process_custom_data(data_dir, output_csv):
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    mp4_files = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]
    
    rows = []
    
    # Target classes based on the English translations in the JSON
    positive_keywords = ["foul", "penalty", "free kick", "free-kick", "handball"]
    hard_negative_keywords = ["shot at goal", "defensive action", "corner", "offside", "save"]
    
    for jf in json_files:
        json_path = os.path.join(data_dir, jf)
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {jf}")
                continue
            
        video_title = data.get("video_title", "")
        
        # Find the matching MP4 by checking if it starts with the video_title from the JSON
        matching_mp4 = None
        for mp4 in mp4_files:
            if mp4.startswith(video_title):
                matching_mp4 = mp4
                break
                
        if not matching_mp4:
            print(f"Warning: No matching MP4 found for JSON: {jf}")
            continue
            
        events = data.get("events", [])
        for event in events:
            action_en = str(event.get("primary_action_en", "")).lower()
            
            target_class = None
            if any(k in action_en for k in positive_keywords):
                target_class = action_en
            elif any(k in action_en for k in hard_negative_keywords):
                target_class = "None"
                
            if target_class:
                frame_in = event.get("frame_in", 0)
                frame_out = event.get("frame_out", 0)
                
                rows.append({
                    "Video_File": matching_mp4,
                    "Target_Class": target_class,
                    "Original_Action": event.get("primary_action", ""),
                    "Original_Action_En": event.get("primary_action_en", ""),
                    "Frame_In": frame_in,
                    "Frame_Out": frame_out,
                    "Duration_Frames": event.get("duration", 0)
                })
                
    # Write the extracted samples to a CSV manifest
    fieldnames = ["Video_File", "Target_Class", "Original_Action", "Original_Action_En", "Frame_In", "Frame_Out", "Duration_Frames"]
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            
    print(f"\nSuccessfully matched JSONs to MP4s.")
    print(f"Mined {len(rows)} samples. Manifest written to {output_csv}.")

if __name__ == "__main__":
    data_dir = "/home/pilot/Desktop/ballsaction/custom_data"
    output_csv = "/home/pilot/Desktop/ballsaction/custom_extraction_manifest.csv"
    process_custom_data(data_dir, output_csv)
