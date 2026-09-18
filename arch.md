# X3D Free-kick Detector Architecture & Flow

This document details the complete end-to-end architecture and inference pipeline of the Meta X3D Medium Action Spotting model trained to detect football Set Pieces (Free-kicks and Penalties).

## 1. Core Model Architecture: Meta X3D Medium

The backbone of this system is the **X3D (Expand 3D) Medium** architecture developed by Meta (Facebook Research). 
Unlike standard 2D CNNs (like ResNet or YOLO) which only look at spatial features of individual frames, X3D uses **3D Convolutions (`(C, T, H, W)`)** to simultaneously learn spatial features (shapes/colors) and temporal features (motion/speed over time).

### Modifications
We loaded the base `x3d_m` model natively pre-trained on the Kinetics-400 human action dataset to leverage generalized motion-understanding features.
We then surgically replaced the final classification head (`self.model.blocks[5].proj`) with a custom binary `nn.Linear` layer:
- **Class 0:** Background (Open play, kickoffs, corners, throw-ins, fouls)
- **Class 1:** Set Piece (Direct free-kick, Indirect free-kick, Penalty)

### Input Specifications
The model strictly expects a 4-Dimensional Tensor (plus the batch dimension):
- **Shape:** `(Batch, Channels=3, Time=40, Height=224, Width=224)`
- **Temporal Resolution:** 40 frames extracted at precisely 2 FPS (representing exactly 20 seconds of real-time video).
- **Spatial Resolution:** 224x224 RGB.
- **Normalization:** Standard Kinetics normalizations (Mean: `[0.45, 0.45, 0.45]`, Std: `[0.225, 0.225, 0.225]`).

---

## 2. Training Pipeline & Hard Negative Mining

To achieve high precision and reduce false positives on visually similar stoppages (e.g., Kick-offs), the model was trained using an aggressive **Hard Negative Mining** strategy:

1. **Extraction Ratio:** A strict 1:4.5 class imbalance was enforced (1 Set Piece clip : 4.5 Background clips).
2. **Hard Negatives:** During background sampling, 60% of the background clips were explicitly forced to be visually confusing non-target events (Kick-offs, Corners, Throw-ins, Substitutions, Cards, Offsides, and Normal Fouls).
3. **Open Play:** The remaining 40% of the background clips were purely random open-play segments to ensure the model retained baseline knowledge of standard football motion.
4. **Weighted Loss:** The `nn.CrossEntropyLoss` was weighted `[1.0, 4.5]` to mathematically compensate for the dataset imbalance during backpropagation.
5. **Early Stopping:** Training utilized an automated Early Stopping monitor with a patience of 4 epochs based on the Validation Loss to prevent overfitting and capture the absolute best checkpoint (`x3d_best.pth`).

---

## 3. Inference Pipeline (`inference.py`)

The inference script is heavily optimized to process massive HD match files with zero overhead.

### Step A: High-Speed Decoupled Extraction
Instead of choking Python with OpenCV video stream decoding, the script spawns a silent `ffmpeg` subprocess to instantly rip the HD video into lightweight JPEGs at exactly `2 FPS` and `224x224` resolution.

### Step B: Strided Sliding Window
A custom `DataLoader` sweeps a 40-frame (20-second) window across the extracted frames. 
- **Stride:** 4 frames (2 seconds).
- This means the model generates a prediction every 2 seconds of the match.
- Inference is executed using `torch.amp.autocast('cuda')` (Mixed Precision FP16) to maximize GPU throughput.

### Step C: Non-Maximum Suppression (NMS) Cooldown
If the model detects a Set Piece (Confidence > 0.85), it enters a cooldown phase:
- It groups all positive detections that occur within 20 seconds of each other.
- It suppresses the overlaps and selects the single timestamp that generated the absolute **Maximum Probability** (e.g., 99.8%).

### Step D: Zero-Loss Extraction Routing
Using the optimized NMS timestamps, the script uses `ffmpeg -c copy` to instantly slice out the original HD source video (without degrading quality via re-encoding). 
The output clips are automatically sorted into confidence-based directories:
- `conf_95/` : Extreme Confidence (>= 95%)
- `conf_90/` : High Confidence (>= 90%)
- `conf_85/` : Medium Confidence (>= 85%)
