import os
import torch
import torchaudio
from torch.utils.data import Dataset

class WhistleAudioDataset(Dataset):
    """
    Audio dataset loader for whistle classification with SpecAugment.
    SpecAugment randomly masks frequency bands and time frames to force the network
    to identify whistle harmonics even in the presence of loud crowd noise.
    """
    def __init__(self, data_dir, is_training=True):
        self.data_dir = data_dir
        self.is_training = is_training
        
        self.clips = []
        # Background clips (Label 0)
        bg_dir = os.path.join(data_dir, "background")
        if os.path.exists(bg_dir):
            for fname in os.listdir(bg_dir):
                if fname.endswith(".wav"):
                    self.clips.append({"audio_path": os.path.join(bg_dir, fname), "label": 0})
                    
        # Whistle clips (Label 1)
        w_dir = os.path.join(data_dir, "whistle")
        if os.path.exists(w_dir):
            for fname in os.listdir(w_dir):
                if fname.endswith(".wav"):
                    self.clips.append({"audio_path": os.path.join(w_dir, fname), "label": 1})
                    
        print(f"WhistleAudioDataset ({data_dir}): {len(self.clips)} clips (Whistles: {len([c for c in self.clips if c['label'] == 1])}, Backgrounds: {len([c for c in self.clips if c['label'] == 0])})")
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # SpecAugment transforms
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=25)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        try:
            waveform, sr = torchaudio.load(clip["audio_path"])
            if waveform.shape[1] == 0:
                raise ValueError("Empty audio")
        except Exception:
            waveform = torch.zeros((1, 10 * 16000))
            sr = 16000
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            
        target_len = 10 * 16000
        if waveform.shape[1] < target_len:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :target_len]
            
        spec = self.amp_to_db(self.mel_transform(waveform)) # (1, 128, 313)
        
        if self.is_training:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
            
        label = torch.tensor(clip["label"], dtype=torch.long)
        return spec, label

if __name__ == "__main__":
    ds = WhistleAudioDataset("audio_dataset_v2/train", is_training=True)
    s, y = ds[0]
    print(f"Spec shape: {s.shape}, label: {y}")
