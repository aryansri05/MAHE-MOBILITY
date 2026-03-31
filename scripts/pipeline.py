from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

# ── internal modules ──────────────────────────────────
from mahe_mobility.models.resnet_extractor import ResNetFeatureExtractor
from mahe_mobility.geometry.lss_core import (
    CameraConfig,
    BEVGridConfig,
    DepthConfig,
    GeometryArchitect,
    DepthPrecomputer,
)
from mahe_mobility.tasks.task1_lidar_to_occupancy import load_lidar_ego_frame, lidar_to_occupancy
from mahe_mobility.models.occupancy import (
    OccupancyCriterion,
    occupancy_iou,
    distance_weighted_error,
    find_optimal_threshold,
)
from mahe_mobility.models.bev_occupancy import BEVOccupancyModel
from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION
from mahe_mobility.dataset import NuScenesFrontCameraDataset

# T4 VRAM shields
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# =============================================================
#  Helper: quaternion (w,x,y,z) → 4×4 ego2cam matrix
# =============================================================

def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),   2*(x*z + y*w),
        2*(x*y + z*w),   1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),   2*(y*z + x*w),   1 - 2*(x*x + y*y),
    ]).reshape(3, 3)

def build_ego2cam(translation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    R = quat_to_rot(rotation)
    cam2ego = torch.eye(4, dtype=torch.float32)
    cam2ego[:3, :3] = R
    cam2ego[:3,  3] = translation
    return torch.linalg.inv(cam2ego)

# =============================================================
#  Lift Head
# =============================================================

class LiftHead(nn.Module):
    def __init__(self, in_channels: int, depth_cfg: DepthConfig, cam_cfg: CameraConfig):
        super().__init__()
        self.D = depth_cfg.d_steps
        self.img_h, self.img_w = cam_cfg.image_h, cam_cfg.image_w
        self.depth_head = nn.Conv2d(in_channels, self.D, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_up = F.interpolate(feat, size=(self.img_h, self.img_w), mode="bilinear", align_corners=False)
        depth_dist = self.depth_head(feat_up).softmax(dim=1)
        return feat_up.unsqueeze(2) * depth_dist.unsqueeze(1), depth_dist

# =============================================================
#  BEVModel — Triple-Block Checkpointed
# =============================================================

class BEVModel(nn.Module):
    def __init__(
        self,
        out_channels: int = 64,
        cam_cfg: CameraConfig = CameraConfig(),
        bev_cfg: BEVGridConfig = BEVGridConfig(),
        depth_cfg: DepthConfig = DepthConfig(),
    ):
        super().__init__()
        self.cam_cfg, self.bev_cfg, self.depth_cfg = cam_cfg, bev_cfg, depth_cfg
        self.feature_extractor = ResNetFeatureExtractor(out_channels=out_channels)
        self.lift_head   = LiftHead(in_channels=out_channels, depth_cfg=depth_cfg, cam_cfg=cam_cfg)
        self.geometry    = GeometryArchitect(cam_cfg, bev_cfg, depth_cfg, ego2cam=torch.eye(4))
        self.occupancy_model = BEVOccupancyModel(lift_channels=out_channels)

    def forward(self, images, intrinsics, translation, rotation):
        device = images.device

        # Block 1 — Backbone
        feat_2d = torch.utils.checkpoint.checkpoint(
            self.feature_extractor, images, use_reentrant=False
        )

        # Block 2 — Lift (depth prediction + frustum generation)
        feat_3d, depth_probs = torch.utils.checkpoint.checkpoint(
            self.lift_head, feat_2d, use_reentrant=False
        )

        # Block 3 — Geometry / Splat
        def _splat(f3d, k, t, r):
            e2c = build_ego2cam(t[0].cpu(), r[0].cpu()).to(device)
            return self.geometry(f3d, k, e2c.unsqueeze(0).expand(f3d.shape[0], -1, -1))

        bev_raw = torch.utils.checkpoint.checkpoint(
            _splat, feat_3d, intrinsics, translation, rotation, use_reentrant=False
        )

        logits = self.occupancy_model(bev_raw)
        return logits, depth_probs

# =============================================================
#  Training pipeline — IoU Surgery Edition
# =============================================================

def train_pipeline(
    dataroot: str = "./data/nuscenes",
    version: str = "v1.0-mini",
    batch_size: int = 1,
    accumulation_steps: int = 4,
    num_epochs: int = 20,
    out_channels: int = 64,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"\n🔪 IoU Surgery Pipeline | device={device} | eff_bs={batch_size*accumulation_steps}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    cam_cfg   = CameraConfig(image_h=224, image_w=480)
    bev_cfg   = BEVGridConfig(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION)
    depth_cfg = DepthConfig()   # 41 depth bins

    full_dataset = NuScenesFrontCameraDataset(dataroot=dataroot, version=version, train=True)
    val_full     = NuScenesFrontCameraDataset(dataroot=dataroot, version=version, train=False)

    total_samples = len(full_dataset)
    val_size, train_size = int(0.1 * total_samples), int(0.9 * total_samples)

    train_dataset, _ = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    _, val_dataset = torch.utils.data.random_split(
        val_full, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = BEVModel(out_channels=out_channels, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg).to(device)

    # IoU Surgery Loss Stack
    criterion = OccupancyCriterion(
        lambda_lovasz=1.0,    # PRIMARY: direct Jaccard optimisation
        lambda_focal=0.5,     # hard-example mining
        lambda_boundary=0.3,  # Sobel edge sharpening
        lambda_bce=0.2,       # calibration anchor
        lambda_depth=1.0,
        lambda_tv=0.05,
    ).to(device)

    # ── VRAM Shield: 8-bit AdamW ─────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4, weight_decay=1e-4)
        print("✅ bitsandbytes 8-bit AdamW loaded (VRAM shield active)")
    except ImportError:
        print("⚠️  bitsandbytes not found — falling back to standard AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_model_path      = "bev_model_best_surgery.pth"
    latest_checkpoint    = "bev_model_latest.pth"
    start_epoch, best_iou, best_threshold = 0, 0.0, 0.5

    # Auto-resume
    if os.path.exists(latest_checkpoint):
        print(f"🔄 Resuming from {latest_checkpoint}...")
        ckpt = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch    = ckpt["epoch"] + 1
        best_iou       = ckpt.get("best_iou", 0.0)
        best_threshold = ckpt.get("best_threshold", 0.5)
        print(f"   Starting from epoch {start_epoch+1}, best IoU so far: {best_iou:.2f}%")

    print("\n📐 Loss Stack: Lovász + Focal + Sobel Boundary + BCE\n")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)   # "Exhale" protocol

        for batch_idx, (images, intrinsics, trans, rot, gt_occ, gt_depth) in enumerate(train_loader):
            images, intrinsics = images.to(device), intrinsics.to(device)
            trans, rot = trans.to(device), rot.to(device)
            gt_occ, gt_depth = gt_occ.to(device), gt_depth.to(device)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits, pred_depth = model(images, intrinsics, trans, rot)
                loss_dict = criterion(logits, gt_occ.float(), pred_depth, gt_depth)
                loss = loss_dict["loss"] / accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)   # delete, not zero
                    if (batch_idx + 1) % 60 == 0:
                        torch.cuda.empty_cache()            # flush reserved trash
            else:
                loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if (batch_idx + 1) % 10 == 0:
                total_loss = loss.item() * accumulation_steps
                print(
                    f"  [{epoch+1}/{num_epochs}][{batch_idx+1}] "
                    f"loss={total_loss:.4f} | "
                    f"lovász={loss_dict['lovasz_loss'].item():.4f} "
                    f"boundary={loss_dict['boundary_loss'].item():.4f} "
                    f"focal={loss_dict['focal_loss'].item():.4f}"
                )

        # ── VALIDATION — Dynamic F1 Thresholding ───────────────────────
        model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        all_probs, all_gt = [], []
        with torch.no_grad():
            for v_imgs, v_k, v_t, v_r, v_occ, v_depth in val_loader:
                v_imgs, v_k = v_imgs.to(device), v_k.to(device)
                v_t, v_r   = v_t.to(device), v_r.to(device)
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    v_logits, _ = model(v_imgs, v_k, v_t, v_r)
                all_probs.append(torch.sigmoid(v_logits).cpu())
                all_gt.append(v_occ.cpu())

        all_probs_t = torch.cat(all_probs, dim=0)  # (N, 1, H, W)
        all_gt_t    = torch.cat(all_gt,    dim=0)

        # Sweep 0.10 → 0.90 to find the F1-maximising threshold
        best_threshold, best_f1 = find_optimal_threshold(all_probs_t, all_gt_t)

        # Compute IoU at the dynamic threshold
        pred_mask = (all_probs_t > best_threshold).bool()
        avg_v_iou = occupancy_iou(pred_mask, all_gt_t.bool()).item() * 100

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"✅ Epoch {epoch+1} | Val IoU: {avg_v_iou:.2f}% | "
            f"F1: {best_f1:.4f} @ threshold={best_threshold:.2f}"
        )

        scheduler.step()

        # Save full checkpoint every epoch
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_iou":             best_iou,
            "best_threshold":       best_threshold,
        }, latest_checkpoint)
        print(f"💾 Checkpoint: {os.path.abspath(latest_checkpoint)}")

        if avg_v_iou > best_iou:
            best_iou = avg_v_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"⭐ Best model saved → {os.path.abspath(best_model_path)} (IoU: {best_iou:.2f}%)")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="./data/nuscenes")
    parser.add_argument("--version",  type=str, default="v1.0-mini")
    parser.add_argument("--epochs",   type=int, default=20)
    args = parser.parse_args()
    train_pipeline(dataroot=args.dataroot, version=args.version, num_epochs=args.epochs)