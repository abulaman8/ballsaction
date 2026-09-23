import os
import argparse
from huggingface_hub import snapshot_download
from SoccerNet.utils import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
import subprocess

def download_hf(token, password):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soccernet_data")
    os.makedirs(data_dir, exist_ok=True)
    
    downloader = SoccerNetDownloader(LocalDirectory=data_dir)
    downloader.password = password
    
    all_games = getListGames(["train", "valid", "test"])
    bundesliga_games = [g for g in all_games if "germany_bundesliga" in g.lower()]
    other_games = [g for g in all_games if "germany_bundesliga" not in g.lower()]
    target_games = bundesliga_games + other_games
    
    print(f"Downloading Labels-v2.json from KAUST for {len(target_games)} games...")
    for game in target_games:
        # Download just the JSON labels from KAUST (fast)
        if not os.path.exists(os.path.join(data_dir, game, "Labels-v2.json")):
            downloader.downloadGame(game=game, files=["Labels-v2.json"])
            
    print(f"Generating allow-list for {len(target_games)} games for HuggingFace...")
    patterns = []
    for game in target_games:
        patterns.append(f"*{game}*")
        
    print(f"Starting extremely fast download from HuggingFace `videos-224p` branch...")
    try:
        snapshot_download(
            repo_id="SoccerNet/SoccerNet_raw_HQ",
            repo_type="dataset",
            revision="videos-224p", # EXACT REVISION
            local_dir=data_dir,
            allow_patterns=patterns,
            token=token,
            max_workers=8
        )
        print("Download complete!")
    except Exception as e:
        print(f"Download failed: {e}")
        
    # Attempt to unzip if they downloaded as password-protected .zip files
    print("Checking for zip files to extract...")
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                print(f"Extracting {zip_path}...")
                subprocess.run(["unzip", "-o", "-P", password, zip_path, "-d", root])
                # Optionally remove the zip file to save space
                # os.remove(zip_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, required=True, help="HuggingFace API token (hf_...)")
    parser.add_argument('--password', type=str, default="s0cc3rn3t", help="Password for zip extraction if needed")
    args = parser.parse_args()
    
    download_hf(args.token, args.password)
