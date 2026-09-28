import os
import torch
import multiprocessing
from dataset_fusion import DualFusionDataset

def process_clip(clip_info):
    vid_path = clip_info["video_path"]
    aud_path = clip_info["audio_path"]
    label = clip_info["label"]
    
    out_pt = vid_path.replace(".mp4", ".pt")
    if os.path.exists(out_pt):
        return
        
    try:
        # We need to instantiate the dataset here because multiprocessing can't share it easily
        # Actually, we can just write the logic here to avoid dataset class overhead
        import cv2
        import torchaudio
        import torchvision.transforms as transforms
        
        vid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
        
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)
        amp_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Load Video
        cap = cv2.VideoCapture(vid_path)
        frames = []
        while len(frames) < 40:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(vid_transform(frame))
        cap.release()
        
        while len(frames) < 40:
            if len(frames) > 0:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros((3, 224, 224)))
        video_tensor = torch.stack(frames, dim=1).half() # Convert to float16 to save space
        
        # Load Audio
        try:
            waveform, sr = torchaudio.load(aud_path)
            if waveform.shape[1] == 0:
                raise ValueError("Empty audio")
        except Exception:
            # Fallback to 5 seconds of silence if audio track is missing/corrupted
            waveform = torch.zeros((1, 5 * 16000))
            sr = 16000
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            
        # Pad waveform to exactly 5 seconds if it's too short to avoid STFT errors
        target_length = 5 * 16000
        if waveform.shape[1] < target_length:
            pad_amount = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        spec = mel_transform(waveform)
        spec = amp_to_db(spec).half() # Convert to float16
        
        if spec.shape[2] < 256:
            pad = 256 - spec.shape[2]
            spec = torch.nn.functional.pad(spec, (0, pad))
        else:
            spec = spec[:, :, :256]
            
        torch.save({
            "video": video_tensor,
            "audio": spec,
            "label": label
        }, out_pt)
        
    except Exception as e:
        print(f"Failed to process {vid_path}: {e}")

def main():
    print("Gathering clips...")
    clips = []
    
    for dataset_dir in ["foul_dataset/train", "setpiece_dataset/train"]:
        if not os.path.exists(dataset_dir): continue
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        for cls_name in subdirs:
            cls_dir = os.path.join(dataset_dir, cls_name)
            label = 0 if cls_name == "background" else 1
            for fname in os.listdir(cls_dir):
                if fname.endswith(".mp4"):
                    clips.append({
                        "video_path": os.path.join(cls_dir, fname),
                        "audio_path": os.path.join(cls_dir, fname).replace(".mp4", ".wav"),
                        "label": label
                    })
                    
    print(f"Found {len(clips)} clips to convert to PyTorch tensors.")
    pool = multiprocessing.Pool(processes=8)
    pool.map(process_clip, clips)
    pool.close()
    pool.join()
    print("Done pre-computing tensors!")

if __name__ == "__main__":
    main()
