<div align="center">

# MAHE-MOBILITY: Camera-to-BEV Occupancy Pipeline
### MAHE Mobility Hackathon 2026 · Track 01 · PS3

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-nuScenes-00BFFF)](https://www.nuscenes.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Team Ctrl+Z** · MIT Bengaluru, MAHE · Dept. of Electronics & Communication Engineering

</div>

---

## What This Project Does

This pipeline takes a **single front-facing camera image** from the nuScenes dataset and predicts which areas around the vehicle are physically occupied — expressed as a **Bird's-Eye-View (BEV) Occupancy Grid** — without using any LiDAR at inference time.

```
Input                     →    Output
─────────────────────          ───────────────────────────────
Front camera image             250 × 250 BEV occupancy grid
224 × 480 RGB pixels           0.2 m / cell resolution
nuScenes front cam             50 m range (0–50 m ahead)
```

---

## Repository Structure

```
MAHE-MOBILITY/
├── scripts/
│   └── pipeline.py          ← Main training script
├── src/
│   └── mahe_mobility/
│       ├── config.py         ← BEV grid constants
│       ├── dataset.py        ← NuScenes data loader + BEV augmentation
│       ├── geometry/
│       │   └── lss_core.py   ← Lift-Splat-Shoot geometry engine
│       ├── models/
│       │   ├── resnet_extractor.py  ← ResNet34 feature backbone
│       │   ├── occupancy.py         ← Loss functions (BCE + Dice + Depth TV)
│       │   └── bev_occupancy.py     ← BEV decoder (U-Net head)
│       └── tasks/
│           ├── task1_lidar_to_occupancy.py  ← LiDAR → GT label generation
│           └── task3_evaluation_iou.py      ← Metrics + visualization
├── evaluate_local.py        ← Run local inference on a saved model
├── visualize_bev.py         ← Visualize the BEV projection
├── requirements.txt
└── data/
    └── nuscenes/            ← Place your nuScenes dataset here
```

---

## Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10 |
| GPU VRAM | ≥ 16 GB (Kaggle T4 supported) |
| nuScenes dataset | v1.0-mini (quick test) or v1.0-trainval (full) |

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
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **On Kaggle**, dependencies are pre-installed. Simply run `pip install nuscenes-devkit` if missing.

---

### Step 3 — Download and place the nuScenes dataset

Download the **nuScenes** dataset from [nuscenes.org](https://www.nuscenes.org/nuscenes#download).

Place or symlink it so the structure looks like this:

```
data/nuscenes/
├── maps/
├── samples/           ← Camera images
├── sweeps/            ← LiDAR sweeps (only needed for training)
└── v1.0-mini/         ← Use this for quick testing
    └── v1.0-trainval/ ← Use this for full training
```

```bash
mkdir -p data
ln -s /path/to/your/nuscenes data/nuscenes
```

---

### Step 4 — Train the model

```bash
# Quick test on the mini split (~404 samples, ~2 hrs on a T4 GPU for 20 epochs)
python scripts/pipeline.py \
    --dataroot ./data/nuscenes \
    --version v1.0-mini \
    --epochs 20

# Full training on trainval split
python scripts/pipeline.py \
    --dataroot ./data/nuscenes \
    --version v1.0-trainval \
    --epochs 20
```

The training script will automatically:
- Resume from `bev_model_latest.pth` if it already exists (crash recovery)
- Save `bev_model_latest.pth` after every epoch
- Save `bev_model_best_v2.pth` whenever a new best IoU is achieved

> **Kaggle Users:** The script sets `PYTORCH_ALLOC_CONF=expandable_segments:True` and uses Gradient Checkpointing automatically to prevent CUDA Out of Memory errors on 16GB GPUs.

---

### Step 5 — Evaluate the trained model (local)

Once training is complete (or you have a `.pth` file), run:

```bash
python evaluate_local.py
```

This will:
1. Load the saved model weights
2. Run inference on a random validation sample
3. Print metrics to the terminal
4. Save a 4-panel diagnostic plot as `hackathon_final_plot.png`

**Sample output:**
```
📊 FINAL METRICS:
   IoU Score  : 27.51%
   Precision  : 0.5301
   Recall     : 1.0000
   F1 Score   : 0.6929
   DWE        : 0.0054  (lower is better)

✅ Diagnostic plot saved as 'hackathon_final_plot.png'
```

> **Note:** To use a specific weights file, edit the `WEIGHTS` variable at the top of `evaluate_local.py`.

---

## Model Architecture

The pipeline uses a **Lift-Splat-Shoot (LSS)** architecture:

```
[Camera Image 224×480]
        ↓
[ResNet34 Backbone]     ← Extracts 2D feature maps
        ↓
[Lift Head]             ← Predicts a 41-bin depth distribution per pixel
        ↓
[GeometryArchitect]     ← Projects features into 3D space using camera intrinsics/extrinsics
        ↓
[Voxel Pool (Splat)]    ← Collapses the 3D frustum onto a 2D BEV grid
        ↓
[BEV U-Net Decoder]     ← Refines the BEV feature map
        ↓
[Output: 250×250 Occupancy Grid]
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image resolution | 224 × 480 | Input to the backbone |
| Depth bins | 41 | LSS frustum discretisation (4 m – 45 m) |
| BEV grid | 250 × 250 | 0.2 m/cell, covers −25 to +25 m laterally, 0–50 m ahead |
| Batch size | 1 (physical) | Gradient accumulation over 4 steps for effective BS=4 |
| Learning rate | 2e-4 | AdamW with weight decay 1e-4 |
| Loss | BCE + Dice + Depth TV | λ_tv = 0.05 |

---

## Training on Kaggle

The recommended way to train on Kaggle:

```python
# In a Kaggle notebook cell:
!git clone https://github.com/aryansri05/MAHE-MOBILITY.git
%cd MAHE-MOBILITY
!git submodule update --init --recursive 2>/dev/null || true
!pip install nuscenes-devkit pyquaternion -q

!python scripts/pipeline.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-mini \
    --epochs 20
```

Saved checkpoints (`bev_model_latest.pth` and `bev_model_best_v2.pth`) will appear in `/kaggle/working/`.

---

## Checkpointing & Recovery

The training script is designed to survive Kaggle session timeouts:

- **`bev_model_latest.pth`** — Full checkpoint saved after every epoch. Contains model weights, optimizer state, and epoch number.
- **`bev_model_best_v2.pth`** — Best-performing model by validation IoU.

To **resume** a crashed session, simply re-run `pipeline.py` with the same arguments. It will auto-detect `bev_model_latest.pth` and pick up from where it left off.

---

## Dataset

**nuScenes** — autonomous driving dataset recorded in Boston and Singapore.

| Property | Details |
|----------|---------|
| Scenes | 1,000 total (700 train / 150 val / 150 test) |
| Mini split | 10 scenes / ~404 samples |
| Camera | Front camera only used as model input |
| LiDAR | Used only to generate ground-truth BEV labels (not needed at inference) |
| GT labels | Binary BEV occupancy derived from LiDAR point clouds |

---

## Team

| Name | Role |
|------|------|
| Shivansh Srivastava | Team Lead — architecture & training pipeline |
| Riddhi Jain | CNN design, segmentation, FP16 inference |
| Shadman Nishat | nuScenes pipeline, label generation |
| Aryan Srivastava | View transformers, spatial geometry, evaluation |

---

## References

- Philion & Fidler — [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711) · ECCV 2020
- Caesar et al. — [nuScenes](https://arxiv.org/abs/1929.08676) · CVPR 2020

---

## License

MIT License · nuScenes dataset subject to [nuScenes Terms of Use](https://www.nuscenes.org/terms-of-use) (non-commercial).

> ⚠️ This repository is publicly accessible for evaluation purposes as required by the MAHE Mobility Challenge 2026 submission guidelines.
