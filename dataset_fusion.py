import os
import cv2
cv2.setNumThreads(0)
import torch
import torchaudio
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class DualFusionDataset(Dataset):
    def __init__(self, data_dir, is_training=True):
        self.data_dir = data_dir
        
        # Find the positive class directory dynamically
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
                    vid_path = os.path.join(cls_dir, fname)
                    aud_path = vid_path.replace(".mp4", ".wav")
                    if os.path.exists(aud_path):
                        self.clips.append({
                            "video_path": vid_path,
                            "audio_path": aud_path,
                            "label": cls_idx
                        })
                        
        print(f"Fusion Dataset initialized with {len(self.clips)} valid clips from {data_dir}.")
        
        self.vid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        
        # Load Video
        cap = cv2.VideoCapture(clip_info["video_path"])
        frames = []
        while len(frames) < 40:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.vid_transform(frame))
        cap.release()
        
        while len(frames) < 40:
            if len(frames) > 0:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros((3, 224, 224)))
        video_tensor = torch.stack(frames, dim=1)
        
        # Load Audio
        try:
            waveform, sr = torchaudio.load(clip_info["audio_path"])
            if waveform.shape[1] == 0:
                raise ValueError("Empty audio")
        except Exception:
            waveform = torch.zeros((1, 5 * 16000))
            sr = 16000
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            
        target_length = 5 * 16000
        if waveform.shape[1] < target_length:
            pad_amount = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        spec = self.mel_transform(waveform)
        spec = self.amp_to_db(spec)
        
        if spec.shape[2] < 256:
            pad = 256 - spec.shape[2]
            spec = torch.nn.functional.pad(spec, (0, pad))
        else:
            spec = spec[:, :, :256]
            
        return video_tensor, spec, torch.tensor(clip_info["label"], dtype=torch.long)
