from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from nuscenes.nuscenes import NuScenes

# ── your teammates' modules ──────────────────────────────────
from mahe_mobility.models.resnet_extractor import ResNetFeatureExtractor  # 2-D backbone
from mahe_mobility.geometry.lss_core import (  # LSS geometry engine
    CameraConfig,
    BEVGridConfig,
    DepthConfig,
    GeometryArchitect,
    DepthPrecomputer,
)
from mahe_mobility.tasks.task1_lidar_to_occupancy import load_lidar_ego_frame, lidar_to_occupancy
from mahe_mobility.tasks.task2_distance_weighted_loss import DistanceWeightedBCELoss
from mahe_mobility.models.bev_occupancy import BEVOccupancyModel
from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION
from mahe_mobility.dataset import NuScenesFrontCameraDataset


# =============================================================
#  Helper: quaternion (w,x,y,z) → 3×3 rotation matrix
# =============================================================


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """NuScenes quaternion (w,x,y,z) → (3,3) rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ]
    ).reshape(3, 3)


def build_ego2cam(translation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    Build a 4×4 ego→camera matrix from NuScenes calibration.

    NuScenes stores the camera-to-ego transform, so we invert it.

    Parameters
    ----------
    translation : (3,)  cam-to-ego translation
    rotation    : (4,)  cam-to-ego quaternion (w,x,y,z)

    Returns
    -------
    ego2cam : (4, 4)
    """
    R = quat_to_rot(rotation)
    cam2ego = torch.eye(4, dtype=torch.float32)
    cam2ego[:3, :3] = R
    cam2ego[:3, 3] = translation
    return torch.linalg.inv(cam2ego)  # ego→cam


# =============================================================
#  Lift Head — 2-D features → depth-distributed frustum features
# =============================================================


class LiftHead(nn.Module):
    """
    Expand (B, C, feat_H, feat_W) → (B, C, D, img_H, img_W).

    1. Upsample spatial dims to (img_H, img_W).
    2. Predict a D-bin depth distribution per pixel (softmax).
    3. Outer-product: feature × depth_weight.
    """

    def __init__(self, in_channels: int, depth_cfg: DepthConfig, cam_cfg: CameraConfig):
        super().__init__()
        self.D = depth_cfg.d_steps
        self.img_h = cam_cfg.image_h
        self.img_w = cam_cfg.image_w
        self.depth_head = nn.Conv2d(in_channels, self.D, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # 1. Upsample to full image resolution
        feat_up = F.interpolate(
            feat, size=(self.img_h, self.img_w), mode="bilinear", align_corners=False
        )
        # 2. Depth distribution
        depth_dist = self.depth_head(feat_up).softmax(dim=1)  # (B, D, H, W)
        # 3. Outer product → (B, C, D, H, W)
        return feat_up.unsqueeze(2) * depth_dist.unsqueeze(1)


# =============================================================
#  BEVModel — full end-to-end nn.Module
# =============================================================


class BEVModel(nn.Module):
    """
    End-to-end BEV perception model.

    Forward signature matches what your dataset returns:
        images      : (B, 3, img_H, img_W)
        intrinsics  : (B, 3, 3)
        translation : (B, 3)   cam-to-ego translation
        rotation    : (B, 4)   cam-to-ego quaternion (w,x,y,z)

    Returns
        bev : (B, out_channels, bev_H, bev_W)
    """

    def __init__(
        self,
        out_channels: int = 64,
        cam_cfg: CameraConfig = CameraConfig(),
        bev_cfg: BEVGridConfig = BEVGridConfig(),
        depth_cfg: DepthConfig = DepthConfig(),
    ):
        super().__init__()
        self.cam_cfg = cam_cfg
        self.bev_cfg = bev_cfg
        self.depth_cfg = depth_cfg

        # Stage 1 — 2-D feature extractor (ResNet backbone)
        self.feature_extractor = ResNetFeatureExtractor(out_channels=out_channels)

        # Stage 2 — Lift: 2-D → 3-D frustum features
        self.lift_head = LiftHead(
            in_channels=out_channels, depth_cfg=depth_cfg, cam_cfg=cam_cfg
        )

        # Stage 3 — Geometry: build with identity ego2cam as placeholder;
        # precomp is rebuilt each forward() with real extrinsics
        self.geometry = GeometryArchitect(
            cam_cfg, bev_cfg, depth_cfg, ego2cam=torch.eye(4)
        )
        self.occupancy_model = BEVOccupancyModel(lift_channels=out_channels)

    def forward(
        self,
        images: torch.Tensor,  # (B, 3, H, W)
        intrinsics: torch.Tensor,  # (B, 3, 3)
        translation: torch.Tensor,  # (B, 3)
        rotation: torch.Tensor,  # (B, 4)
    ) -> torch.Tensor:

        device = images.device

        # ── Stage 1: 2-D feature extraction ─────────────────────────
        feat_2d = self.feature_extractor(images)  # (B, C, feat_H, feat_W)

        # ── Stage 2: Lift to 3-D ─────────────────────────────────────
        feat_3d = self.lift_head(feat_2d)  # (B, C, D, img_H, img_W)

        # ── Stage 3: Update BEV index map with real extrinsic ────────
        # NuScenes extrinsics are fixed per rig, so sample [0] is used
        # for the whole batch. For variable extrinsics, loop per sample.
        ego2cam = build_ego2cam(
            translation[0].cpu(),
            rotation[0].cpu(),
        ).to(device)

        with torch.no_grad():
            self.geometry.precomp = DepthPrecomputer(
                self.geometry.frustum_gen,
                self.bev_cfg,
                ego2cam,
            ).to(device)

        # ── Stage 3: Splat into BEV ──────────────────────────────────
        bev_raw = self.geometry(feat_3d)  # (B, C, bev_H, bev_W)
        logits = self.occupancy_model(bev_raw)  # (B, 1, bev_H, bev_W)
        return logits


# =============================================================
#  Training loop
# =============================================================


def train_pipeline(
    dataroot: str = "./data/nuscenes",
    version: str = "v1.0-mini",
    batch_size: int = 4,
    num_epochs: int = 1,
    out_channels: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Initializing Full Pipeline on: {device}")

    # Configs — image size must match dataset resize
    cam_cfg = CameraConfig(image_h=224, image_w=480)
    bev_cfg = BEVGridConfig(
        x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION
    )
    depth_cfg = DepthConfig()

    # Dataset & loader
    dataset = NuScenesFrontCameraDataset(dataroot=dataroot, version=version)
    
    # OPTIMIZED LOADER: Uses CPU threads to fetch data while GPU does math
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    # Model
    model = BEVModel(
        out_channels=out_channels, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg
    ).to(device)

    criterion = DistanceWeightedBCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Starting Training...")
    for epoch in range(num_epochs):
        for batch_idx, (images, intrinsics, trans, rot, gt_occupancy) in enumerate(dataloader):
            # NO MORE BATCH LIMIT HERE! It will process the whole dataset.
            images = images.to(device)
            intrinsics = intrinsics.to(device)
            trans = trans.to(device)
            rot = rot.to(device)
            gt_occupancy = gt_occupancy.to(device)

            # Forward — correctly passes all 4 args into BEVModel.forward()
            bev = model(images, intrinsics, trans, rot)

            # Replaced dummy loss with the actual DistanceWeightedBCELoss
            loss = criterion(bev, gt_occupancy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"  Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"BEV shape: {tuple(bev.shape)}  "
                f"Loss: {loss.item():.4f}"
            )

    print("\n✅ Training complete.")
    
    # Generate visualization for the last batch
    print("Generating post-training visualizations...")
    try:
        from mahe_mobility.tasks.task3_evaluation_iou import visualise_error_map
        # Take the first sample from the batch to visualize
        pred_probs = torch.sigmoid(bev[0, 0]).detach().cpu().numpy()
        gt = gt_occupancy[0, 0].detach().cpu().numpy()
        
        visualise_error_map(pred_probs, gt, save_path="pipeline_result.png")
        print("✅ Visualization saved to pipeline_result.png")
    except Exception as e:
        print(f"Failed to generate visualize image: {e}")
        
    # --- ADDED: EXPORT MODEL WEIGHTS ---
    torch.save(model.state_dict(), "bev_model_final.pth")
    print("💾 Model weights successfully saved to bev_model_final.pth")
    # -----------------------------------
    
    return model


# =============================================================
#  Entry point
# =============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mahe Mobility BEV Training")
    parser.add_argument("--dataroot", type=str, default="./data/nuscenes", help="Path to nuScenes dataset")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="nuScenes dataset version")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    args = parser.parse_args()

    print("=" * 60)
    print("LSS Full Pipeline — NuScenes → 2D → 3D Lift → BEV")
    print("=" * 60)
    model = train_pipeline(
        dataroot=args.dataroot,
        version=args.version,
        num_epochs=args.epochs
    )