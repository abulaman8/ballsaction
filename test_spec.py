import subprocess
import torchaudio
import matplotlib.pyplot as plt
import torch
import os

def extract_audio(video_path, start, dur, out_wav):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", 
        "-ss", str(start), 
        "-i", video_path, 
        "-t", str(dur), 
        "-q:a", "0", "-map", "a", 
        out_wav
    ]
    subprocess.run(cmd)

def plot_spec(wav_path, out_png):
    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found. Audio extraction failed?")
        return
        
    waveform, sample_rate = torchaudio.load(wav_path)
    # Average channels if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )(waveform)
    
    # Convert to log scale (dB)
    mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(mel_spectrogram[0].numpy(), aspect='auto', origin='lower', cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close()

vid = "/home/pilot/Desktop/ballsaction/soccernet_data/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv"
print("Extracting Whistle...")
extract_audio(vid, 1343, 4, "test_whistle.wav")
plot_spec("test_whistle.wav", "/home/pilot/.gemini/antigravity/brain/54051a72-f909-4f8f-afda-b38268595c9a/spectrogram_whistle.png")

print("Extracting Background...")
extract_audio(vid, 1000, 4, "test_bg.wav")
plot_spec("test_bg.wav", "/home/pilot/.gemini/antigravity/brain/54051a72-f909-4f8f-afda-b38268595c9a/spectrogram_bg.png")
print("Done!")
