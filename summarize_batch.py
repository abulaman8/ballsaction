import re

with open("batch_report.txt", "r") as f:
    content = f.read()

clips = content.strip().split("\n\n")

total_clips = 0
boosted_clips = 0
videos = set()

for clip in clips:
    if not clip.strip(): continue
    total_clips += 1
    
    # Extract video name
    match = re.search(r'\[(.*?)\]', clip)
    if match:
        videos.add(match.group(1))
        
    if "(Boosted)" in clip:
        boosted_clips += 1

print(f"Total Videos Processed: {len(videos)}")
print(f"Total Clips Generated: {total_clips}")
print(f"Total Boosted by Whistle: {boosted_clips} ({(boosted_clips/total_clips)*100:.1f}%)")
