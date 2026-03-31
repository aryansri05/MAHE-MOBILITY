# Mahe Mobility - BEV Occupancy Pipeline

This repository contains an end-to-end framework for predicting 2D Bird's-Eye View (BEV) occupancy grids directly from autonomous vehicle camera feeds (nuScenes dataset), utilizing cross-view attention and lifted frustums.

This repository was heavily refactored from a multi-person hackathon prototype into a cleanly structured Python project ready for GitHub publication.

## 🏗 Repository Structure

```
mahe_mobility/
├── README.md
├── requirements.txt         # Project pip dependencies
├── src/mahe_mobility/       # Core Python library
│   ├── config.py            # Global constants and grid dimensions
│   ├── dataset.py           # nuScenes dataloader
│   ├── geometry/            # LSS (Lift-Splat-Shoot) geometry engine (extrinsics/frustums)
│   ├── models/              # Neural networks (ResNet backbone, BEV Encoders, Occupancy Head)
│   ├── tasks/               # Core LiDAR and BEV task modules (Task 1, 2, 3)
│   └── utils/               # Geometry extraction utils
├── scripts/                 # Execution scripts
│   └── pipeline.py          # E2E Training logic and architecture binding
```

## 🛠 Installation

Requirements can be heavily environment-dependent when working with PyTorch + CUDA, but generally, a python `venv` suffices:

```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (Install your system's corresponding PyTorch version first)
pip install -r requirements.txt
```

## 🚀 Running the Pipeline

Ensure that you have the [nuScenes](https://www.nuscenes.org/) dataset available and mapped.

To run the end-to-end training pipeline combining 2D feature extraction, 3D metric lifting, and BEV space flattening:

```bash
# Add src to pythonpath so custom module imports resolve
export PYTHONPATH=src
python scripts/pipeline.py
```
