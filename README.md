<div align="center">

# 🚗 MAHE-MOBILITY
### Camera-Only Bird's-Eye-View Occupancy Grid Pipeline
#### MAHE Mobility Hackathon 2026 · Track 01 · PS3

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![nuScenes](https://img.shields.io/badge/Dataset-nuScenes-00BFFF)](https://www.nuscenes.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MAHE](https://img.shields.io/badge/Institution-MAHE%20Bengaluru-orange)](https://manipal.edu)

**Team Ctrl+Z** · Dept. of Electronics & Communication Engineering · MIT Bengaluru

</div>

---

## 📌 Overview

This project implements an **end-to-end camera-to-BEV deep learning pipeline** for autonomous driving perception. The model takes a single front-facing camera image and produces a top-down **Bird's-Eye-View (BEV) Occupancy Grid** — mapping which physical areas around the vehicle are occupied — **without any LiDAR at inference time**.

```
┌──────────────────────┐         ┌──────────────────────────────────┐
│  Front Camera Image  │  ─────► │   BEV Occupancy Grid (250×250)   │
│   224 × 480 pixels   │         │   0.2 m/cell · 50 m range ahead  │
└──────────────────────┘         └──────────────────────────────────┘
         Input                              Output
```

**Why this matters:** Level 4 autonomous vehicles need a top-down representation of their surroundings to plan safe paths. BEV grids derived from cameras alone can replace expensive LiDAR sensors ($10,000+) for this task.

---

## 🖼️ Sample Output

The pipeline produces a 4-panel diagnostic evaluation plot showing exactly how the model performs on a real nuScenes scene:

![BEV Occupancy Evaluation](hackathon_final_plot.png)

> *Left to Right: Ground Truth (LiDAR), Predicted Probability Heatmap, Binary Prediction (threshold=0.5), Error Map (TP/FP/FN)*

### Evaluation Metrics (Validation Run)

| Metric | Score | Notes |
|--------|-------|-------|
| **IoU Score** | **38.51%** | Primary accuracy metric |
| **Precision** | 0.5379 | True positive rate on predicted occupied cells |
| **Recall** | 0.5756 | Coverage of actual occupied cells |
| **F1 Score** | 0.5561 | Harmonic mean of Precision & Recall |
| **DWE** | 0.0081 | Distance-Weighted Error (lower is better) |

---

## 🏗️ Architecture

The pipeline uses a **Lift-Splat-Shoot (LSS)** architecture with three key stages:

```
[Camera Image 224×480]
         │
         ▼
 ┌───────────────────┐
 │  ResNet34 Backbone │  ← Extracts rich 2D semantic features
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐
 │    Lift Head       │  ← Predicts 41-bin depth distribution per pixel
 │  (Depth Estimator) │     Generates a 3D frustum of weighted features
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐
 │ GeometryArchitect │  ← Projects 3D frustum → BEV using camera
 │  (LSS Projection) │     intrinsics (K) and extrinsics [R|t]
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐
 │  BEV U-Net Decoder│  ← Refines the 250×250 BEV feature map
 └────────┬──────────┘
          │
          ▼
 [250×250 Occupancy Grid]
```

### Key Configuration

| Parameter | Value |
|-----------|-------|
| Input resolution | 224 × 480 px |
| Depth bins | 41 (range: 4–45 m) |
| BEV grid size | 250 × 250 cells |
| BEV resolution | 0.2 m/cell |
| BEV coverage | ±25 m lateral, 0–50 m ahead |
| Loss function | BCE + Dice + Depth TV (λ=0.05) |

---

## 📁 Repository Structure

```
MAHE-MOBILITY/
├── scripts/
│   └── pipeline.py              ← ⭐ Main training script (run this to train)
├── src/
│   └── mahe_mobility/
│       ├── config.py            ← BEV grid constants (X_MIN, Y_MAX, RESOLUTION...)
│       ├── dataset.py           ← NuScenes data loader + BEV augmentation
│       ├── geometry/
│       │   └── lss_core.py      ← Lift-Splat-Shoot geometry engine + caching
│       ├── models/
│       │   ├── resnet_extractor.py   ← ResNet34 backbone feature extractor
│       │   ├── occupancy.py          ← Loss functions (BCE + Dice + Depth TV)
│       │   └── bev_occupancy.py      ← BEV U-Net decoder head
│       └── tasks/
│           ├── task1_lidar_to_occupancy.py  ← LiDAR → GT label generator
│           └── task3_evaluation_iou.py      ← Metrics + visualization
├── evaluate_local.py            ← ⭐ Run inference on a saved model
├── visualize_bev.py             ← Visualize the BEV projection geometry
├── requirements.txt
└── data/
    └── nuscenes/                ← Place your nuScenes dataset here
```

---

## ⚡ Quick Start

### Requirements

| Dependency | Version |
|------------|---------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0.0 |
| GPU VRAM | ≥ 16 GB (Kaggle T4 / local GPU) |
| nuScenes | v1.0-mini (testing) or v1.0-trainval (full) |

---

### 1. Clone the repository

```bash
git clone https://github.com/aryansri05/MAHE-MOBILITY.git
cd MAHE-MOBILITY
```

---

### 2. Set up the Python environment

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

---

### 3. Prepare the nuScenes dataset

Download nuScenes from [nuscenes.org](https://www.nuscenes.org/nuscenes#download) and place or symlink it:

```bash
mkdir -p data
ln -s /path/to/your/nuscenes data/nuscenes
```

Expected structure:

```
data/nuscenes/
├── maps/
├── samples/             ← Camera images stored here
├── sweeps/              ← LiDAR sweeps (training only)
├── v1.0-mini/           ← For quick testing (404 samples)
└── v1.0-trainval/       ← For full training (~700k frames)
```

---

### 4. Train the model

```bash
# Quick test — mini split (~2 hours on a T4 GPU)
python scripts/pipeline.py \
    --dataroot ./data/nuscenes \
    --version v1.0-mini \
    --epochs 20

# Full training — trainval split
python scripts/pipeline.py \
    --dataroot ./data/nuscenes \
    --version v1.0-trainval \
    --epochs 20
```

**What happens during training:**
- Progress is logged every 10 batches with live loss and peak GPU memory usage
- `bev_model_latest.pth` — saved after every epoch (full checkpoint for resumability)
- `bev_model_best_v2.pth` — saved whenever a new best validation IoU is achieved
- If training is interrupted, simply re-run the same command — it auto-resumes from the latest checkpoint

---

### 5. Evaluate a trained model

```bash
python evaluate_local.py
```

This runs inference on a random validation sample and prints:
- IoU Score, Precision, Recall, F1 Score, Distance-Weighted Error
- Saves a 4-panel diagnostic plot as `hackathon_final_plot.png`

> To use a specific weights file, change the `WEIGHTS` variable at the top of `evaluate_local.py`:
> ```python
> WEIGHTS = "bev_model_best_v2.pth"   # ← change this
> ```

---

## 🖥️ Running on Kaggle

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

The script automatically:
- Sets `PYTORCH_ALLOC_CONF=expandable_segments:True` to prevent memory fragmentation
- Uses **Gradient Checkpointing** (Triple-Block) to fit within 16 GB VRAM at full 224×480 resolution
- Uses **Gradient Accumulation** (4 steps) for training stability at batch size 1

---

## 💾 Saved Files Reference

| File | When Saved | Contents |
|------|-----------|----------|
| `bev_model_latest.pth` | After every epoch | Full checkpoint: weights + optimizer state + epoch number |
| `bev_model_best_v2.pth` | When val IoU improves | Model weights only (best performing) |
| `hackathon_final_plot.png` | After `evaluate_local.py` | 4-panel BEV diagnostic visualization |

---

## 🧪 Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable Accuracy-First deployment build |
| `update-focal-loss` | Experimental Focal Loss integration |
| `experiment` | Active experimentation sandbox |
| `integration/person-b` | Teammate integration branch |

---

## 👥 Team

| Name | Role |
|------|------|
| Shivansh Srivastava | Team Lead — model architecture & training pipeline |
| Riddhi Jain | CNN design, segmentation, FP16 inference |
| Shadman Nishat | nuScenes pipeline, label generation |
| Aryan Srivastava | View transformers, spatial geometry, evaluation |

---

## 📚 References

- Philion & Fidler — [Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711) · ECCV 2020
- Caesar et al. — [nuScenes: A multimodal dataset for autonomous driving](https://arxiv.org/abs/1929.08676) · CVPR 2020
- He et al. — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) · CVPR 2016

---

## 📄 License

MIT License · nuScenes dataset subject to [nuScenes Terms of Use](https://www.nuscenes.org/terms-of-use) (non-commercial research only).

> ⚠️ This repository is publicly accessible for evaluation purposes as required by the MAHE Mobility Hackathon 2026 submission guidelines.
