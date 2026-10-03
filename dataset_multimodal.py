import os
import cv2
cv2.setNumThreads(0)
import torch
import torchaudio
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MultiModalBinaryDataset(Dataset):
    def __init__(self, data_dir, is_training=True, target_frames=30, max_bg_ratio=None, max_samples_per_class=None):
        self.data_dir = data_dir
        self.is_training = is_training
        self.target_frames = target_frames
        
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        pos_dir = [d for d in subdirs if d != "background"][0]
        
        self.classes = {"background": 0, pos_dir: 1}
        self.clips = []
        
        for cls_name, cls_idx in self.classes.items():
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
                
            for fname in os.listdir(cls_dir):
                if fname.endswith(".mp4"):
                    self.clips.append({
                        "video_path": os.path.join(cls_dir, fname),
                        "audio_path": os.path.join(cls_dir, fname.replace(".mp4", ".wav")),
                        "label": cls_idx
                    })
                    
        if self.is_training or (max_samples_per_class is not None or max_bg_ratio is not None):
            import random
            random.seed(42)
            pos_clips = [c for c in self.clips if c["label"] == 1]
            bg_clips = [c for c in self.clips if c["label"] == 0]
            if max_samples_per_class is not None and len(pos_clips) > max_samples_per_class:
                pos_clips = random.sample(pos_clips, max_samples_per_class)
            if max_bg_ratio is not None:
                max_bgs = int(len(pos_clips) * max_bg_ratio)
                if len(bg_clips) > max_bgs:
                    bg_clips = random.sample(bg_clips, max_bgs)
            self.clips = pos_clips + bg_clips
                
        print(f"MultiModal Dataset initialized with {len(self.clips)} valid clips from {data_dir}.")
        
        self.to_tensor = transforms.ToTensor()
        # Native 16:9 widescreen: (224 height, 398 width)
        self.resize = transforms.Resize((224, 398))
        self.normalize = transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        self.jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
        
        self.mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)
        self.amp_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        video_path = clip_info["video_path"]
        audio_path = clip_info["audio_path"]
        label = clip_info["label"]
        
        # 1. Load Video (Native 16:9 Aspect Ratio)
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        do_flip = self.is_training and torch.rand(1).item() < 0.5
        
        if self.is_training:
            skip_frames = int(torch.randint(0, 5, (1,)).item())
            for _ in range(skip_frames):
                ret, _ = cap.read()
                if not ret: break
                
        while len(frames) < self.target_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if do_flip:
                frame = cv2.flip(frame, 1)
                
            t_frame = self.to_tensor(frame)
            t_frame = self.resize(t_frame)
            
            if self.is_training:
                t_frame = self.jitter(t_frame)
                
            t_frame = self.normalize(t_frame)
            frames.append(t_frame)
            
        cap.release()
        
        while len(frames) < self.target_frames:
            if len(frames) > 0:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros((3, 224, 398)))
                
        video_tensor = torch.stack(frames, dim=1) # (3, 30, 224, 398)
        
        # 2. Load Audio (15.0s Mel-spectrogram)
        audio_spec = torch.zeros((1, 128, 469), dtype=torch.float32)
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
            try:
                wav, sr = torchaudio.load(audio_path)
                if wav.shape[0] > 1:
                    wav = torch.mean(wav, dim=0, keepdim=True)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                    
                target_len = 240000 # 15s * 16000
                if wav.shape[1] < target_len:
                    wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
                else:
                    wav = wav[:, :target_len]
                    
                spec = self.amp_db(self.mel_tf(wav)) # (1, 128, ~469)
                if spec.shape[2] < 469:
                    spec = torch.nn.functional.pad(spec, (0, 469 - spec.shape[2]))
                else:
                    spec = spec[:, :, :469]
                audio_spec = spec
            except Exception:
                pass
                
        return video_tensor, audio_spec, torch.tensor(label, dtype=torch.long)
