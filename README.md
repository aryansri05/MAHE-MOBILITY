<div align="center">

# Camera-to-BEV Occupancy Grid
### MAHE Mobility Challenge 2026 · Track 01 · PS3

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-nuScenes-00BFFF)](https://www.nuscenes.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Team Ctrl+Z** · MIT Bengaluru, MAHE · Dept. of Electronics & Communication Engineering

</div>

---

## 1. Project Overview

Standard front-facing cameras distort perspective, making it impossible to reason about distances or free space directly. Level 4 autonomous vehicles solve this by converting camera features into a **Bird's-Eye-View (BEV) Occupancy Grid** — a top-down 2D map where every cell represents a fixed physical area and its probability of being occupied.

This project implements an end-to-end deep learning pipeline that takes a **single front-facing camera image** from the nuScenes dataset and produces a **200×200 binary occupancy grid** covering a 20 m × 20 m area at 10 cm/pixel resolution — **no LiDAR required at inference time**.

```
Input                          Output
──────────────────────────     ──────────────────────────────
Front camera image             BEV occupancy grid
900 × 1600 RGB pixels    →     200 × 200 binary mask
nuScenes front cam             10 cm × 10 cm per cell
```

---

## 2. Example Output

![BEV Occupancy Evaluation](bev_results.png)

> **Left to right:** Ground truth (LiDAR) · Prediction probability map · Thresholded binary prediction · Error map (green = correct free, teal = correct occupied, red = false alarm, orange = missed obstacle)

---

## 3. Results

```
FINAL METRICS:
  IoU Score  : 38.51%
  Precision  : 0.5379
  Recall     : 0.5756
  F1 Score   : 0.5561
  DWE        : 0.0081  (lower is better)
```

### Benchmark comparison

| Method | Backbone | Input | IoU ↑ | DWE ↓ | FPS |
|--------|----------|-------|-------|-------|-----|
| VPN (baseline) | ResNet-50 | Front cam | 0.31 | 0.38 | 22 |
| PON | ResNet-50 | Front cam | 0.35 | 0.34 | 18 |
| LSS (original) | EfficientNet | 6 cameras | 0.47 | 0.29 | 25 |
| **Ours** | **EfficientNet-B4** | **Front cam only** | **~0.39** | **~0.008** | **>15** |

---

## 4. Model Architecture

```
[Camera Image]  →  [EfficientNet-B4]  →  [LSS Transformer]  →  [U-Net]  →  [BEV Mask]
 900×1600 RGB       + FPN backbone       Lift·Splat·Shoot     Decoder    200×200
```

### Stage 1 — Backbone: EfficientNet-B4 + FPN
Extracts multi-scale feature maps from the input image resized to **256×704**.
- EfficientNet-B4 produces feature maps at `1/8`, `1/16`, `1/32` of input resolution
- FPN fuses all three scales
- ~28M total parameters

### Stage 2 — View Transformer: Lift-Splat-Shoot (LSS)

| Step | What it does |
|------|-------------|
| **Lift** | For every pixel, predict a depth distribution over 64 bins (4 m – 45 m) |
| **Splat** | Project 3D frustum points onto the ground plane using camera extrinsics |
| **Shoot** | Output a spatially aligned 200×200 BEV feature map |

### Stage 3 — BEV Decoder: U-Net
3 encoder + 3 decoder blocks with skip connections → binary occupancy mask

### Stage 4 — Loss Function
```
L = 0.7 × BCE(pos_weight=3.0)  +  0.3 × Dice
```

---

## 5. Setup & Running (macOS / Linux)

### Requirements

| Dependency | Version |
|-----------|---------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0.0 |
| nuScenes devkit | ≥ 1.1.0 |

### Step 1 — Clone & install

```bash
git clone https://github.com/Ctrl-Z-Team/camera-bev-occupancy.git
cd MAHE-MOBILITY-1
pip3 install -r requirements.txt
```

### Step 2 — Download nuScenes mini dataset

1. Register at [nuscenes.org](https://www.nuscenes.org/nuscenes#download)
2. Download **Mini split** (~4 GB) → you get `v1.0-mini.tgz`
3. Set up the data folder:

```bash
mkdir -p ~/MAHE-MOBILITY-1/data/nuscenes

# Option A — extract from .tgz (recommended)
tar -xzf ~/Downloads/v1.0-mini.tgz -C ~/MAHE-MOBILITY-1/data/nuscenes/

# Option B — copy existing folder
sudo chmod -R 755 ~/Downloads/v1.0-mini/
sudo cp -r ~/Downloads/v1.0-mini/ ~/MAHE-MOBILITY-1/data/nuscenes/
```

Expected structure:
```
data/nuscenes/
└── v1.0-mini/
    ├── maps/
    ├── samples/
    ├── sweeps/
    └── v1.0-mini/        ← JSON files (category.json etc.) live here
```

### Step 3 — Run the pipeline

> ⚠️ **Must run from inside `src/` with `PYTHONPATH` set — this is required.**

```bash
cd ~/MAHE-MOBILITY-1/src
PYTHONPATH=. python3 ../scripts/pipeline.py --dataroot ../data/nuscenes --version v1.0-mini
```

### Step 4 — Local evaluation only

```bash
cd ~/MAHE-MOBILITY-1/src
PYTHONPATH=. python3 ../evaluate_local.py
```

---

## 6. Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'mahe_mobility'` | Run from `src/` with `PYTHONPATH=. python3 ...` |
| `Database version not found: ./data/nuscenes/v1.0-mini` | Add `--dataroot ../data/nuscenes --version v1.0-mini` |
| `PermissionError: category.json` | Run `sudo chmod -R 755 ~/MAHE-MOBILITY-1/data/nuscenes/` |
| `FileNotFoundError: category.json` | Dataset folder is empty — re-extract from `.tgz` |
| `zsh: command not found: pip` | Use `pip3` or `python3 -m pip` instead |

---

## 7. Repository Structure

```
MAHE-MOBILITY-1/
├── scripts/
│   └── pipeline.py
├── src/
│   └── mahe_mobility/
│       ├── geometry/
│       │   └── lss_core.py
│       ├── models/
│       │   ├── bev_encoder.py
│       │   ├── bev_occupancy.py
│       │   ├── occupancy.py
│       │   └── resnet_extractor.py
│       ├── tasks/
│       │   ├── task1_lidar_to_occupancy.py
│       │   ├── task2_distance_weighted_loss.py
│       │   └── task3_evaluation_iou.py
│       ├── utils/
│       │   └── geometry_extractor.py
│       ├── config.py
│       └── dataset.py
├── evaluate_local.py
├── requirements.txt
└── README.md
```

---

## 8. Team

| Name | Role |
|------|------|
| Shivansh Srivatsava | Team Lead — model architecture & training |
| Riddhi Jain | CNN design, segmentation, FP16 inference |
| Shadman Nishat | nuScenes pipeline, label generation |
| Aryan Srivatsava | View transformers, spatial geometry, evaluation |

---

## References

- Philion & Fidler — [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711) · ECCV 2020
- Liu et al. — [BEVFusion](https://arxiv.org/abs/2205.13542) · ICRA 2023
- Caesar et al. — [nuScenes](https://arxiv.org/abs/1929.08676) · CVPR 2020
- Tan & Le — [EfficientNet](https://arxiv.org/abs/1905.11946) · ICML 2019

---

## License

MIT License · nuScenes dataset subject to [nuScenes Terms of Use](https://www.nuscenes.org/terms-of-use) (non-commercial).

> ⚠️ This repository is publicly accessible for evaluation purposes as required by the MAHE Mobility Challenge 2026 submission guidelines.
