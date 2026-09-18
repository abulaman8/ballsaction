import json
import csv
import argparse
import os

def parse_gametime(gametime_str):
    """Parses SoccerNet gameTime format like '1 - 01:11' into half (int) and seconds (int)."""
    parts = gametime_str.split(' - ')
    half = int(parts[0])
    time_str = parts[1]
    mins, secs = map(int, time_str.split(':'))
    total_seconds = mins * 60 + secs
    return half, total_seconds

def generate_manifest(json_path, output_csv, window_size=60):
    """
    Parses a single SoccerNet Labels-v2.json file and extracts timestamps 
    for positive classes and hard negative classes.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # UrlLocal typically represents the game directory
    game_id = data.get('UrlLocal', os.path.basename(os.path.dirname(json_path)))
    annotations = data.get('annotations', [])
    
    # Our targeted 4 actions
    positive_classes = {"Foul", "Penalty", "Direct free-kick", "Indirect free-kick"}
    
    # Actions that look similar (e.g. game stoppages or heavy action) but are not our targets
    hard_negative_classes = {"Tackle", "Clearance", "Shots on target", "Shots off target", "Corner"}
    
    half_window = window_size // 2
    
    rows = []
    
    for ann in annotations:
        label = ann.get('label')
        game_time = ann.get('gameTime')
        
        if not label or not game_time:
            continue
            
        half, total_seconds = parse_gametime(game_time)
        
        target_class = None
        if label in positive_classes:
            target_class = label
        elif label in hard_negative_classes:
            target_class = "None"
            
        if target_class:
            window_start = max(0, total_seconds - half_window)
            window_end = total_seconds + half_window
            rows.append({
                'Game_ID': game_id,
                'Half': half,
                'Timestamp_s': total_seconds,
                'Original_Label': label,
                'Target_Class': target_class,
                'Window_Start': window_start,
                'Window_End': window_end
            })
            
    # Write the extracted samples to a CSV manifest
    fieldnames = ['Game_ID', 'Half', 'Timestamp_s', 'Original_Label', 'Target_Class', 'Window_Start', 'Window_End']
    file_exists = os.path.isfile(output_csv)
    
    # Append if exists, else write header
    mode = 'a' if file_exists else 'w'
    with open(output_csv, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            
    print(f"Processed {json_path}: Mined {len(rows)} samples. Written to {output_csv}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine SoccerNet dataset for specific targeted actions.")
    parser.add_argument('--json_path', type=str, required=True, help="Path to Labels-v2.json")
    parser.add_argument('--output_csv', type=str, default="extraction_manifest.csv", help="Output CSV manifest path")
    parser.add_argument('--window_size', type=int, default=60, help="Total window size in seconds (default: 60)")
    
    args = parser.parse_args()
    generate_manifest(args.json_path, args.output_csv, args.window_size)
