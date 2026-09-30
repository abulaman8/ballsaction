import os
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, data_dirs, is_training=True):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        
        self.clips = []
        
        for data_dir in data_dirs:
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            if not subdirs: continue
            pos_dir = [d for d in subdirs if d != "background"][0]
            
            # Background clips (Label 0)
            bg_dir = os.path.join(data_dir, "background")
            if os.path.exists(bg_dir):
                for fname in os.listdir(bg_dir):
                    if fname.endswith(".wav"):
                        self.clips.append({"audio_path": os.path.join(bg_dir, fname), "label": 0})
                        
            # Positive clips (Label 1)
            p_dir = os.path.join(data_dir, pos_dir)
            if os.path.exists(p_dir):
                for fname in os.listdir(p_dir):
                    if fname.endswith(".wav"):
                        self.clips.append({"audio_path": os.path.join(p_dir, fname), "label": 1})
        
        print(f"Unified Audio Dataset initialized with {len(self.clips)} valid clips.")
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        
        try:
            waveform, sr = torchaudio.load(clip_info["audio_path"])
            if waveform.shape[1] == 0:
                raise ValueError("Empty audio")
        except Exception:
            waveform = torch.zeros((1, 10 * 16000))
            sr = 16000
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            
        target_length = 10 * 16000
        if waveform.shape[1] < target_length:
            pad_amount = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
            
        spec = self.mel_transform(waveform)
        spec = self.amp_to_db(spec)
        
        if spec.shape[2] < 313:
            pad = 313 - spec.shape[2]
            spec = torch.nn.functional.pad(spec, (0, pad))
        else:
            spec = spec[:, :, :313]
            
        return spec, torch.tensor(clip_info["label"], dtype=torch.long)
