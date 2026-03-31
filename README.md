<div align="center">

# MAHE-MOBILITY: Camera-to-BEV Occupancy Pipeline
### MAHE Mobility Hackathon 2026 · Track 01 · PS3

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.15+-orange)](https://pytorch.org/vision)
[![Dataset](https://img.shields.io/badge/Dataset-nuScenes_v1.0-00BFFF)](https://www.nuscenes.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Team Ctrl+Z** · MIT Bengaluru, MAHE · Dept. of Electronics & Communication Engineering

</div>

---

## What This Project Does

This pipeline takes a **single front-facing camera image** and predicts which areas around the vehicle are physically occupied — expressed as a **Bird's-Eye-View (BEV) Occupancy Grid** — using no LiDAR at inference time. LiDAR is only used during training to generate ground-truth labels.

```
Input                     →    Output
─────────────────────          ───────────────────────────────
Front camera image             250 × 250 BEV occupancy grid
224 × 480 RGB pixels           0.2 m/cell resolution
nuScenes CAM_FRONT             Covers 50 m × 50 m around ego vehicle
```

---

## Technology Stack

### Core Frameworks

| Technology | Version | Role |
|-----------|---------|------|
| **Python** | ≥ 3.10 | Runtime |
| **PyTorch** | ≥ 2.0 | Deep learning framework, autograd, AMP |
| **TorchVision** | ≥ 0.15 | ResNet34 pretrained weights, image transforms |
| **nuScenes devkit** | ≥ 1.1.0 | Dataset loading, calibration, LiDAR access |
| **NumPy** | ≥ 1.21 | Array operations, BEV grid computations |
| **Pillow (PIL)** | ≥ 9.0 | Image loading and decoding |
| **Matplotlib** | ≥ 3.5 | Diagnostic visualizations and BEV plots |
| **SciPy** | ≥ 1.9 | Morphological erosion for prediction post-processing |
| **PyQuaternion** | ≥ 0.9 | Quaternion-to-rotation-matrix conversions for camera extrinsics |

---

### Model Architecture

The pipeline implements a **Lift-Splat-Shoot (LSS)** architecture in four stages:

```
[Camera Image 224×480 RGB]
         │
         ▼
[Stage 1: ResNet34 Backbone]
  • Pretrained on ImageNet (torchvision ResNet34_Weights.DEFAULT)
  • Strips avgpool + fc, keeps up to layer3 (stride-8 output)
  • 1×1 Conv projects 256 → 64 channels + BatchNorm + ReLU + Dropout2d(0.2)
  • Output: (B, 64, 28, 60) feature map
         │
         ▼
[Stage 2: Lift Head — 2D → 3D Frustum]
  • Bilinear upsample back to full image resolution (224×480)
  • 1×1 Conv predicts 41-bin depth distribution (d=4.0m to d=45.0m)
  • Softmax over depth bins
  • Outer product: feature × depth weight → (B, 64, 41, 224, 480) frustum
         │
         ▼
[Stage 3: GeometryArchitect — LSS Voxel Pool (Splat)]
  • Uses camera intrinsic matrix K and ego-to-camera extrinsic [R|t]
  • Extrinsics derived via quaternion-to-rotation-matrix using PyQuaternion
  • Device-aware LRU cache for geometry: avoids recomputing frustum mappings
  • Scatter-sum pool frustum features onto 250×250 BEV grid
  • Output: (B, 64, 250, 250) BEV feature volume
         │
         ▼
[Stage 4: BEV U-Net Decoder]
  BEVEncoder (3 encoder + 3 decoder blocks with skip connections):
  • Encoder: Conv2d → BatchNorm2d → ReLU → MaxPool2d (×3)
  • Decoder: Bilinear upsample → Conv2d → BatchNorm2d → ReLU (×3)
  • Skip connections at matching spatial resolutions
  • Output: (B, 128, 250, 250)
         │
         ▼
[OccupancyHead]
  • Two 3×3 convolutions (128 → 64 → 32 channels)
  • Final 1×1 conv → single logit per BEV cell
  • Bias initialized to −4.6 (sigmoid(−4.6) ≈ 0.01, "predict empty" prior)
         │
         ▼
[Output: (B, 1, 250, 250) occupancy logits → sigmoid → binary mask]
```

---

### Loss Functions

The training loss is a weighted combination of two terms designed for the severe class imbalance (~95% empty cells) in BEV occupancy:

| Loss | Formula | Why |
|------|---------|-----|
| **Focal Loss** | `FL(p_t) = −α_t · (1−p_t)^γ · log(p_t)` | Down-weights easy negatives (empty cells), forces focus on rare occupied cells. α=0.25, γ=2.0 |
| **Weighted BCE** | `BCEWithLogitsLoss(pos_weight=20.0)` | Class-balanced baseline; penalises false negatives 20× more than false positives |
| **Total** | `L = 1.0 × Focal + 0.5 × BCE` | Combined for stability + hard-example focus |

---

### Data Augmentation

Applied only during training to improve generalization across weather, lighting, and sensor conditions:

| Augmentation | Parameters | Simulates |
|-------------|-----------|-----------|
| `ColorJitter` | brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1 | Day/night, fog, glare |
| `RandomGrayscale` | p=0.1 | Colour-deprived conditions (IR/night) |
| `RandomAutocontrast` | p=0.3 | Camera auto-exposure variation |
| `GaussianBlur` | kernel=3×3, σ∈[0.1, 1.5] | Rain on lens, focus issues |
| `AddGaussianNoise` | mean=0.0, std=0.03 | Sensor/thermal/dark-current noise |
| `ImageNet Normalize` | mean=[0.485, 0.456, 0.406] | Standard ResNet normalization |

---

### Training Techniques

| Technique | Detail |
|-----------|--------|
| **Optimizer** | AdamW, lr=2e-4, weight_decay=1e-4 |
| **LR Scheduler** | CosineAnnealingLR |
| **Mixed Precision** | `torch.amp.autocast` + `GradScaler` (CUDA only) |
| **Gradient Clipping** | `max_norm=35.0` |
| **Gradient Accumulation** | 4 steps (effective batch size = 4 at physical BS=1) |
| **Gradient Checkpointing** | Triple-block (Backbone / Frustum / Geometry) — saves ~5GB VRAM |
| **Geometry Caching** | LRU cache on frustum-to-BEV mapping (device-aware) |
| **Memory Management** | `PYTORCH_ALLOC_CONF=expandable_segments:True` + `empty_cache()` |

---

### Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Occupancy IoU** | `|pred ∩ gt| / |pred ∪ gt|` over binary BEV masks |
| **Precision** | `TP / (TP + FP)` |
| **Recall** | `TP / (TP + FN)` |
| **F1 Score** | `2 × (P × R) / (P + R)` |
| **Distance-Weighted Error (DWE)** | Weighted MAE where cells closer to ego vehicle are penalized up to 50× more than distant ones |

---

## Repository Structure

```
MAHE-MOBILITY/
├── scripts/
│   └── pipeline.py               ← End-to-end training script
├── src/
│   └── mahe_mobility/
│       ├── config.py              ← BEV grid constants (X_MIN/MAX, RESOLUTION, etc.)
│       ├── dataset.py             ← NuScenes data loader + image augmentation pipeline
│       ├── geometry/
│       │   └── lss_core.py        ← Lift-Splat-Shoot geometry engine + LRU cache
│       ├── models/
│       │   ├── resnet_extractor.py  ← ResNet34 backbone (pretrained, ImageNet)
│       │   ├── bev_encoder.py       ← U-Net BEV encoder/decoder
│       │   ├── occupancy.py         ← Focal Loss, Weighted BCE, IoU, DWE metrics
│       │   └── bev_occupancy.py     ← Full BEVOccupancyModel (encoder + head)
│       └── tasks/
│           ├── task1_lidar_to_occupancy.py  ← LiDAR → GT occupancy label generation
│           └── task3_evaluation_iou.py      ← Visualisation + benchmark metrics
├── evaluate_local.py             ← Run local inference on a saved model
├── visualize_bev.py              ← BEV projection visualization utility
├── requirements.txt
└── data/
    └── nuscenes/                 ← Place your nuScenes dataset here
```

---

## Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10 |
| GPU VRAM | ≥ 16 GB (Kaggle T4 supported) |
| Storage | ~10 GB for nuScenes mini, ~300 GB for trainval |

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/aryansri05/MAHE-MOBILITY.git
cd MAHE-MOBILITY
```

---

### Step 2 — Set up your Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install scipy pyquaternion   # Required for evaluation script
```

> **On Kaggle**, many dependencies are pre-installed. Run `pip install nuscenes-devkit` if it is missing.

---

### Step 3 — Download and link the nuScenes dataset

Download from [nuscenes.org](https://www.nuscenes.org/nuscenes#download). Place it as follows:

```
data/nuscenes/
├── maps/
├── samples/           ← Camera images (CAM_FRONT used)
├── sweeps/            ← LiDAR sweeps (only for label generation)
└── v1.0-mini/         ← Use for quick testing
    or v1.0-trainval/  ← Use for full training
```

```bash
mkdir -p data
ln -s /path/to/your/nuscenes data/nuscenes
```

---

### Step 4 — Train the model

```bash
# Quick test — nuScenes mini (~404 samples, ~2 hours on a T4 for 20 epochs)
python scripts/pipeline.py \
    --dataroot ./data/nuscenes \
    --version v1.0-mini \
    --epochs 20

# Full training — nuScenes trainval (700 scenes)
python scripts/pipeline.py \
    --dataroot ./data/nuscenes \
    --version v1.0-trainval \
    --epochs 20
```

The training script automatically:
- Resumes from `bev_model_latest.pth` if found (crash/timeout recovery)
- Saves `bev_model_latest.pth` (full checkpoint) after every epoch
- Saves `bev_model_best_v2.pth` whenever a new best validation IoU is achieved

---

### Step 5 — Evaluate the trained model

```bash
python evaluate_local.py
```

Edit the `WEIGHTS` variable at the top of `evaluate_local.py` to point to your `.pth` file.

**Expected output:**
```
📊 FINAL METRICS:
   IoU Score  : 27.51%
   Precision  : 0.5301
   Recall     : 1.0000
   F1 Score   : 0.6929
   DWE        : 0.0054  (lower is better)

✅ Diagnostic plot saved as 'hackathon_final_plot.png'
```

---

## Checkpointing & Recovery

| File | When Saved | Contents |
|------|-----------|----------|
| `bev_model_latest.pth` | End of every epoch | Model weights + optimizer state + epoch number |
| `bev_model_best_v2.pth` | When val IoU improves | Model weights only (best performing) |

To resume after a crash: simply re-run `pipeline.py`. It auto-detects the latest checkpoint.

---

## Training on Kaggle

```python
# In a Kaggle notebook cell:
!git clone https://github.com/aryansri05/MAHE-MOBILITY.git
%cd MAHE-MOBILITY
!pip install nuscenes-devkit pyquaternion -q

!python scripts/pipeline.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-mini \
    --epochs 20
```

> The script automatically sets `PYTORCH_ALLOC_CONF=expandable_segments:True` and uses Triple-Block Gradient Checkpointing to prevent CUDA OOM on 16GB T4 GPUs.

---

## Dataset

**nuScenes** — autonomous driving dataset recorded in Boston and Singapore.

| Property | Details |
|----------|---------|
| Total scenes | 1,000 (700 train / 150 val / 150 test) |
| Mini split | 10 scenes · ~404 samples |
| Input | Front camera (CAM_FRONT) only |
| LiDAR use | Training only (GT label generation) — not needed at inference |
| GT labels | Binary BEV occupancy derived from LiDAR point clouds |
| Calibration | Full intrinsic K matrix + quaternion extrinsics per frame |

---

## Team

| Name | Role |
|------|------|
| Shivansh Srivastava | Team Lead — architecture, training pipeline, memory optimization |
| Riddhi Jain | CNN design, segmentation, FP16 inference |
| Shadman Nishat | nuScenes pipeline, label generation |
| Aryan Srivastava | View transformers, spatial geometry, evaluation |

---

## References

- Philion & Fidler — [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711) · ECCV 2020
- Lin et al. — [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) · ICCV 2017
- Caesar et al. — [nuScenes: A multimodal dataset for autonomous driving](https://arxiv.org/abs/1929.08676) · CVPR 2020
- He et al. — [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385) · CVPR 2016

---

## License

MIT License · nuScenes dataset subject to [nuScenes Terms of Use](https://www.nuscenes.org/terms-of-use) (non-commercial).

> ⚠️ This repository is publicly accessible for evaluation purposes as required by the MAHE Mobility Challenge 2026 submission guidelines.
