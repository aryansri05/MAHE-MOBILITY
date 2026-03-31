from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for Kaggle/headless servers
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
from mahe_mobility.models.occupancy import OccupancyCriterion, occupancy_iou, distance_weighted_error
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
        self.last_depth_dist = depth_dist.detach()  # cache for visualization
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

    # Dataset & Val Split — augmentations applied only to train set
    full_dataset = NuScenesFrontCameraDataset(dataroot=dataroot, version=version, train=True)
    total_samples = len(full_dataset)
    val_size = max(1, int(0.1 * total_samples))
    train_size = total_samples - val_size

    # Split indices with a fixed seed for reproducibility
    train_dataset, _ = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    # Validation set uses clean (no-augmentation) transforms
    val_full_dataset = NuScenesFrontCameraDataset(dataroot=dataroot, version=version, train=False)
    _, val_dataset = torch.utils.data.random_split(
        val_full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    print(f"📊 Dataset split: {train_size} train, {val_size} validation samples.")


    # Model
    model = BEVModel(
        out_channels=out_channels, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg
    ).to(device)

    # Replaced basic loss with Focal + Distance Weighted BCE
    criterion = OccupancyCriterion(
        focal_alpha=0.25,
        focal_gamma=2.0,
        pos_weight=20.0,
        lambda_focal=1.0,
        lambda_bce=0.5,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    import os
    checkpoint_path = "bev_checkpoint_v2.pth"
    best_model_path = "bev_model_best_v2.pth"
    start_epoch = 0
    best_iou = 0.0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        # Since we might save dicts or mixed objects, weights_only=False is safer for optimizers/schedulers
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"✅ Resumed training from epoch {start_epoch} (Best IoU: {best_iou:.2f}%)")

    print("Starting Training...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for batch_idx, (images, intrinsics, trans, rot, gt_occupancy) in enumerate(train_loader):
            # NO MORE BATCH LIMIT HERE! It will process the whole dataset.
            images = images.to(device)
            intrinsics = intrinsics.to(device)
            trans = trans.to(device)
            rot = rot.to(device)
            gt_occupancy = gt_occupancy.to(device)

            # Forward — correctly passes all 4 args into BEVModel.forward()
            bev_logits = model(images, intrinsics, trans, rot)

            # Replaced dummy loss with OccupancyCriterion (Focal Loss + DistWeighted BCE)
            loss_dict = criterion(bev_logits, gt_occupancy.float())
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            optimizer.step()

            # Train tracking metrics
            with torch.no_grad():
                probs = torch.sigmoid(bev_logits)
                mask = probs > 0.5
                iou = occupancy_iou(mask, gt_occupancy.bool())

            print(
                f"  Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"(Focal: {loss_dict['focal_loss']:.4f}, BCE: {loss_dict['bce_loss']:.4f})  "
                f"Train-IoU: {iou.item()*100:.2f}%"
            )

            # Save checkpoint frequently to avoid data loss
            if (batch_idx + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_iou': best_iou,
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"  💾 [Checkpoint] Saved mid-epoch progress at batch {batch_idx + 1}")

        # --- VALIDATION LOOP ---
        print("\n  🔍 Running Validation...")
        model.eval()
        val_loss, val_iou, val_dwe = 0.0, 0.0, 0.0
        with torch.no_grad():
            for v_images, v_intrinsics, v_trans, v_rot, v_gt in val_loader:
                v_images = v_images.to(device)
                v_intrinsics = v_intrinsics.to(device)
                v_trans = v_trans.to(device)
                v_rot = v_rot.to(device)
                v_gt = v_gt.to(device)
                
                v_logits = model(v_images, v_intrinsics, v_trans, v_rot)
                v_loss_dict = criterion(v_logits, v_gt.float())
                val_loss += v_loss_dict["loss"].item()
                
                v_probs = torch.sigmoid(v_logits)
                v_mask = v_probs > 0.5
                val_iou += occupancy_iou(v_mask, v_gt.bool()).item()
                val_dwe += distance_weighted_error(v_probs, v_gt).item()
                
        num_val_batches = len(val_loader)
        val_loss /= num_val_batches
        val_iou = (val_iou / num_val_batches) * 100
        val_dwe /= num_val_batches
        
        print(f"  📊 Val Loss: {val_loss:.4f}  |  Val IoU: {val_iou:.2f}%  |  Val DWE: {val_dwe:.4f}")
        
        # Track and save best model based strictly on validation IoU
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"  🌟 New Best Val Model! IoU jumped to {best_iou:.2f}% - saved to {best_model_path}\n")

        # Step the LR scheduler
        scheduler.step()

        # End of epoch checkpoint
        torch.save({
            'epoch': epoch + 1,  # If epoch finished, next loop it will start from epoch+1
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"💾 [Checkpoint] Saved end-of-epoch checkpoint for epoch {epoch + 1} (LR: {scheduler.get_last_lr()[0]:.6f})")

    print("\n✅ Training complete.")

    # Generate visualization from the last validation batch
    print("Generating post-training visualizations...")
    try:
        from mahe_mobility.tasks.task3_evaluation_iou import visualise_error_map
        # Use the last val batch for a clean, unbiased visualization
        pred_probs = torch.sigmoid(v_logits[0, 0]).detach().cpu().numpy()
        gt = v_gt[0, 0].detach().cpu().numpy()

        visualise_error_map(pred_probs, gt, save_path="pipeline_result.png")
        print("✅ Visualization saved to pipeline_result.png")
    except Exception as e:
        print(f"Failed to generate visualization: {e}")
        
    # --- ADDED: EXPORT MODEL WEIGHTS ---
    torch.save(model.state_dict(), "bev_model_final_v2.pth")
    print("💾 Model weights successfully saved to bev_model_final_v2.pth")
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