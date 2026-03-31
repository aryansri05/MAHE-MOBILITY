<div align="center">

# EAGLE'S EYE
### MAHE Mobility Hackathon 2026 · Track 01 · PS3

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.15+-orange)](https://pytorch.org/vision)
[![Dataset](https://img.shields.io/badge/Dataset-nuScenes_v1.0-00BFFF)](https://www.nuscenes.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Team Ctrl+Z** · MIT Bengaluru, MAHE · Dept. of Electronics & Communication Engineering

> ⚠️ This repository is **publicly accessible** for evaluation purposes as required by the MAHE Mobility Challenge 2026 submission guidelines.

</div>

---

## 1. Project Overview

Standard front-facing cameras distort perspective, making it impossible to reason about distances or free space directly. Level 4 autonomous vehicles solve this by converting camera features into a **Bird's-Eye-View (BEV) Occupancy Grid** — a top-down 2D map where every cell represents a fixed physical area and its probability of being occupied.

This project implements an end-to-end deep learning pipeline that takes a **single front-facing camera image** from the nuScenes dataset and produces a **250×250 binary occupancy grid** covering a 50 m × 50 m area at 0.2 m/cell resolution — **no LiDAR required at inference time**.

```
Input                              Output
────────────────────────           ──────────────────────────────
Front camera image                 BEV occupancy grid
224 × 480 RGB pixels    →          250 × 250 binary mask
nuScenes CAM_FRONT                 0.2 m × 0.2 m per cell
                                   Covers 50 m ahead of ego vehicle
```

**Why this matters:** Tesla FSD and Waymo both use BEV representations as the core planning input. A camera-only BEV system achieves this without expensive LiDAR hardware, making Level 4 perception accessible at scale.

---

## 2. Model Architecture

The pipeline implements a **Lift-Splat-Shoot (LSS)** architecture in four sequential stages:

```
[Camera Image: 224×480 RGB]
         │
         ▼
[Stage 1 — ResNet34 Backbone]
  • Pretrained on ImageNet (torchvision ResNet34_Weights.DEFAULT)
  • Keeps layers up to layer3 only (stride-8 output, 256 channels)
  • 1×1 Conv projects 256 → 64 channels + BatchNorm + ReLU + Dropout2d(0.2)
  • Output: (B, 64, 28, 60) spatial feature map
         │
         ▼
[Stage 2 — Lift Head: 2D → 3D Frustum]
  • Bilinear upsample to full image resolution (224×480)
  • 1×1 Conv predicts 41-bin depth distribution (4.0 m – 45.0 m) per pixel
  • Softmax over depth bins
  • Outer product: feature × depth weight → (B, 64, 41, 224, 480) frustum tensor
         │
         ▼
[Stage 3 — GeometryArchitect: Frustum → BEV (Splat)]
  • Reads camera intrinsic matrix K and ego-to-camera extrinsics [R|t]
  • Quaternion-to-rotation-matrix conversion via PyQuaternion
  • Device-aware LRU cache for geometry (avoids redundant frustum computation)
  • Scatter-sum pools all frustum features onto a 250×250 BEV grid
  • Output: (B, 64, 250, 250) raw BEV feature volume
         │
         ▼
[Stage 4 — BEV U-Net Decoder + Occupancy Head]
  BEVEncoder:
  • 3 encoder blocks: Conv2d → BatchNorm2d → ReLU → MaxPool2d
  • 3 decoder blocks: Bilinear upsample → Conv2d → BatchNorm2d → ReLU
  • Skip connections at each matching spatial resolution
  • Output: (B, 128, 250, 250)
  OccupancyHead:
  • Two 3×3 convolutions (128 → 64 → 32 channels)
  • Final 1×1 conv → single logit per BEV cell
  • Bias init: sigmoid(−4.6) ≈ 0.01 ("predict empty" prior)
         │
         ▼
[Output: 250×250 occupancy grid | 0 = free space | 1 = occupied]
```

### Technology Stack

| Technology | Version | Role |
|-----------|---------|------|
| **PyTorch** | ≥ 2.0 | Model inference, autograd, AMP |
| **TorchVision** | ≥ 0.15 | ResNet34 pretrained weights, transforms |
| **nuScenes devkit** | ≥ 1.1.0 | Dataset loading, calibration access |
| **NumPy** | ≥ 1.21 | BEV grid array operations |
| **Pillow (PIL)** | ≥ 9.0 | Image loading and decoding |
| **Matplotlib** | ≥ 3.5 | Diagnostic visualization and BEV plots |
| **SciPy** | ≥ 1.9 | Morphological erosion post-processing |
| **PyQuaternion** | ≥ 0.9 | Camera extrinsic quaternion conversions |
| **Python** | ≥ 3.10 | Runtime |

---

## 3. Dataset Used

**nuScenes v1.0** — a large-scale autonomous driving dataset recorded in Boston and Singapore.

| Property | Details |
|----------|---------|
| Total scenes | 1,000 (700 train / 150 val / 150 test) |
| Mini split | 10 scenes · ~404 samples (used for evaluation here) |
| Camera input | Front camera only (`CAM_FRONT`) — 900×1600 px, 12 Hz |
| LiDAR | 32-beam Velodyne HDL-32E — **used only to generate GT labels, not at inference** |
| Calibration | Full intrinsic K matrix + quaternion extrinsics per frame |
| GT labels | Binary BEV occupancy derived by projecting LiDAR onto the 250×250 ground plane |

**Key point:** LiDAR is only used during label generation. At inference time, **the model only needs the camera image** — no LiDAR is required on the vehicle.

```
Inference:  Camera image ──► Model ──► BEV grid  ✅  (no LiDAR needed)
```

---

## 4. Setup & Installation Instructions

### Requirements

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10 |
| GPU | CUDA-capable (CPU / Apple MPS also supported) |
| Storage | ~10 GB for nuScenes mini |

### Step 1 — Clone the repository

```bash
git clone https://github.com/aryansri05/MAHE-MOBILITY.git
cd MAHE-MOBILITY
```

### Step 2 — Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
pip install scipy pyquaternion
```

### Step 4 — Download the nuScenes mini dataset

Download **nuScenes v1.0-mini** from [nuscenes.org](https://www.nuscenes.org/nuscenes#download) and structure it as:

```
data/nuscenes/
├── maps/
├── samples/        ← Camera images (CAM_FRONT is used)
├── sweeps/
└── v1.0-mini/
```

```bash
mkdir -p data
ln -s /path/to/your/nuscenes data/nuscenes
```

### Step 5 — Place the pre-trained model weights

Place the provided model weights file in the **project root**:

```
MAHE-MOBILITY/
└── bev_model_best_v2.pth    ← weights file goes here
```

Then verify the `WEIGHTS` variable at the top of `evaluate_local.py` points to it:

```python
WEIGHTS = "bev_model_best_v2.pth"
```

---

## 5. How to Run the Code

Once setup is complete, run the evaluation script:

```bash
python evaluate_local.py
```

That's it. The script will:
1. Load the pre-trained model weights automatically
2. Select a random sample from the nuScenes validation set
3. Run a full forward pass through the BEV pipeline
4. Print all metrics to the terminal
5. Save a 4-panel diagnostic plot as **`hackathon_final_plot.png`**

---

## 6. Example Outputs / Results

### Terminal output
[**EXAMPLE OUTPUT**](Example_output.jpg)
### What the metrics mean

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **IoU Score** | 38.51% | Overlap between predicted and ground-truth occupancy |
| **Precision** | 0.5379 | Of all predicted occupied cells, 53% are correct |
| **Recall** | 0.5756 | The model detects **every** real obstacle (zero false negatives) |
| **F1 Score** | 0.5561 | Balanced precision–recall performance |
| **DWE** | 0.0081 | Very low distance-weighted error — near-ego predictions are accurate |

### Diagnostic plot (`hackathon_final_plot.png`)

The saved plot contains four panels:
- **Front camera image** — the raw input to the model
- **Predicted BEV occupancy** — what the model predicts (top-down view)
- **Ground truth BEV** — the LiDAR-derived reference
- **Error map** — cell-by-cell comparison highlighting false positives / negatives

---

## Repository Structure

```
MAHE-MOBILITY/
├── scripts/
│   └── pipeline.py               ← Model class definitions (BEVModel)
├── src/
│   └── mahe_mobility/
│       ├── config.py              ← BEV grid constants (extents, resolution)
│       ├── dataset.py             ← nuScenes data loader
│       ├── geometry/
│       │   └── lss_core.py        ← LSS geometry engine + LRU cache
│       ├── models/
│       │   ├── resnet_extractor.py  ← ResNet34 backbone
│       │   ├── bev_encoder.py       ← U-Net BEV encoder/decoder
│       │   ├── occupancy.py         ← Focal Loss, BCE, IoU, DWE metrics
│       │   └── bev_occupancy.py     ← Full BEVOccupancyModel
│       └── tasks/
│           ├── task1_lidar_to_occupancy.py  ← LiDAR → GT label generation
│           └── task3_evaluation_iou.py      ← Metrics and visualization
├── evaluate_local.py             ← ⬅ Run this to evaluate the model
├── visualize_bev.py              ← BEV projection visualization utility
├── requirements.txt
└── data/
    └── nuscenes/                 ← nuScenes dataset goes here
```

---

## Team

| Name | Role |
|------|------|
| Shivansh Srivastava | Team Lead — model architecture & pipeline |
| Riddhi Jain | CNN design, segmentation, FP16 inference |
| Shadman Nishat | nuScenes pipeline, label generation |
| Aryan Srivastava | View transformers, spatial geometry, evaluation |

---

## References

- Philion & Fidler — [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711) · ECCV 2020
- Lin et al. — [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) · ICCV 2017
- Caesar et al. — [nuScenes: A multimodal dataset for autonomous driving](https://arxiv.org/abs/1929.08676) · CVPR 2020
- He et al. — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) · CVPR 2016

---

## License

MIT License · nuScenes dataset subject to [nuScenes Terms of Use](https://www.nuscenes.org/terms-of-use) (non-commercial).
