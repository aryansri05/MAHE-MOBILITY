"""
MAHE-MOBILITY: Absolute Dominance Accuracy Refactor
====================================================
Branch: experiment

Upgrades:
  1. Focal + Dice + Depth composite loss (AccuracyCriterion)
  2. Logarithmic depth binning (in lss_core.py)
  3. CosineAnnealingWarmRestarts + Stochastic Weight Averaging (SWA)
  4. Hard Example Mining (HEM) via WeightedRandomSampler
  5. Surgical VRAM Reclamation: 8-bit AdamW, set_to_none, triple-block checkpointing
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler

# ── VRAM Reclamation: expand allocator segments ────────────────────
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── internal modules ──────────────────────────────────────────────
from mahe_mobility.models.resnet_extractor import ResNetFeatureExtractor
from mahe_mobility.geometry.lss_core import (
    CameraConfig, BEVGridConfig, DepthConfig,
    GeometryArchitect, DepthPrecomputer,
)
from mahe_mobility.tasks.task1_lidar_to_occupancy import load_lidar_ego_frame, lidar_to_occupancy
from mahe_mobility.models.occupancy import occupancy_iou, distance_weighted_error
from mahe_mobility.models.bev_occupancy import BEVOccupancyModel
from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION
from mahe_mobility.dataset import NuScenesFrontCameraDataset


# =============================================================
#  Helper: quaternion (w,x,y,z) → 4×4 ego2cam matrix
# =============================================================

def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y),
    ]).reshape(3, 3)

def build_ego2cam(translation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    R = quat_to_rot(rotation)
    cam2ego = torch.eye(4, dtype=torch.float32)
    cam2ego[:3, :3] = R
    cam2ego[:3, 3] = translation
    return torch.linalg.inv(cam2ego)


# =============================================================
#  1. AccuracyCriterion — Focal + Dice + Depth Supervision
# =============================================================

class AccuracyCriterion(nn.Module):
    """
    Composite loss for maximum BEV occupancy accuracy.

    L = λ_focal · FocalLoss(α=0.25, γ=2.0)
      + λ_dice  · DiceLoss
      + λ_depth · DepthKLLoss

    - FocalLoss: down-weights the 95% easy empty cells, forces focus on occupied cells.
    - DiceLoss:  penalises boundary imprecision geometrically.
    - DepthKLLoss: KL divergence between predicted depth dist and LiDAR-binned GT.
    """
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        lambda_focal: float = 1.0,
        lambda_dice: float = 0.5,
        lambda_depth: float = 0.3,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.lf = lambda_focal
        self.ld = lambda_dice
        self.ldp = lambda_depth
        self.smooth = smooth

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        weight = alpha_t * (1 - p_t) ** self.gamma
        return (weight * bce).mean()

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()

    def depth_loss(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between predicted depth softmax dist and one-hot GT bins.
        Only computed for pixels where LiDAR GT exists (gt_depth > 0).
        """
        valid = (gt_depth > 0).squeeze(1) if gt_depth.dim() == 4 else (gt_depth > 0)
        if not valid.any():
            return pred_depth.sum() * 0.0

        # pred_depth: (B, D, H, W) — already softmaxed
        # gt_depth:   (B, H, W)   — integer bin index (1-indexed, 0=no point)
        gt_idx = gt_depth.long()
        if gt_idx.dim() == 4:
            gt_idx = gt_idx.squeeze(1)

        # Clamp gt to valid bin range
        D = pred_depth.shape[1]
        gt_idx = (gt_idx - 1).clamp(0, D - 1)  # convert to 0-indexed

        # Gather predicted probability at GT bin
        pred_at_gt = pred_depth.permute(0, 2, 3, 1)  # (B, H, W, D)
        pred_at_gt = pred_at_gt[valid]               # (N, D)
        gt_bins = gt_idx[valid]                      # (N,)

        # Build a one-hot target with slight label smoothing
        target = torch.zeros_like(pred_at_gt)
        target.scatter_(1, gt_bins.unsqueeze(1), 0.9)
        target = target + 0.1 / D  # label smoothing

        # KL divergence: stable via log-softmax
        log_pred = torch.log(pred_at_gt.clamp(min=1e-8))
        return F.kl_div(log_pred, target, reduction="batchmean")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
    ) -> dict:
        l_focal = self.focal_loss(logits, targets)
        l_dice = self.dice_loss(logits, targets)
        l_depth = self.depth_loss(pred_depth, gt_depth)
        total = self.lf * l_focal + self.ld * l_dice + self.ldp * l_depth
        return {
            "loss": total,
            "focal_loss": l_focal.item(),
            "dice_loss": l_dice.item(),
            "depth_loss": l_depth.item(),
        }


# =============================================================
#  Lift Head — depth-distributed frustum features
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
#  BEVModel — Triple-Block Checkpointed forward
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
        self.lift_head = LiftHead(in_channels=out_channels, depth_cfg=depth_cfg, cam_cfg=cam_cfg)
        self.geometry = GeometryArchitect(cam_cfg, bev_cfg, depth_cfg, ego2cam=torch.eye(4))
        self.occupancy_model = BEVOccupancyModel(lift_channels=out_channels)

    def forward(self, images, intrinsics, translation, rotation):
        device = images.device

        # Block 1: ResNet backbone
        feat_2d = torch.utils.checkpoint.checkpoint(
            self.feature_extractor, images, use_reentrant=False
        )

        # Block 2: Depth lifting
        def _frustum(f2d):
            return self.lift_head(f2d)
        feat_3d, depth_probs = torch.utils.checkpoint.checkpoint(
            _frustum, feat_2d, use_reentrant=False
        )

        # Block 3: Geometry / voxel pool
        def _bev(f3d, k, t, r):
            e2c = build_ego2cam(t[0].cpu(), r[0].cpu()).to(device)
            return self.geometry(f3d, k, e2c.unsqueeze(0).expand(f3d.shape[0], -1, -1))
        bev_raw = torch.utils.checkpoint.checkpoint(
            _bev, feat_3d, intrinsics, translation, rotation, use_reentrant=False
        )

        logits = self.occupancy_model(bev_raw)
        return logits, depth_probs


# =============================================================
#  Training pipeline — Absolute Dominance Edition
# =============================================================

def train_pipeline(
    dataroot: str = "./data/nuscenes",
    version: str = "v1.0-mini",
    batch_size: int = 1,
    accumulation_steps: int = 4,
    num_epochs: int = 20,
    out_channels: int = 64,
    swa_start_epoch: int = 12,
):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\n⚡ Absolute Dominance Refactor | Device: {device} | Eff. BS: {batch_size * accumulation_steps}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    cam_cfg = CameraConfig(image_h=224, image_w=480)
    bev_cfg = BEVGridConfig(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION)
    depth_cfg = DepthConfig()  # 41 log-spaced bins

    full_dataset = NuScenesFrontCameraDataset(dataroot=dataroot, version=version, train=True)
    val_full     = NuScenesFrontCameraDataset(dataroot=dataroot, version=version, train=False)

    total   = len(full_dataset)
    val_sz  = int(0.1 * total)
    train_sz = total - val_sz

    train_dataset, _ = torch.utils.data.random_split(
        full_dataset, [train_sz, val_sz], generator=torch.Generator().manual_seed(42)
    )
    _, val_dataset = torch.utils.data.random_split(
        val_full, [train_sz, val_sz], generator=torch.Generator().manual_seed(42)
    )

    # ── 4. Hard Example Mining: initialise uniform sample weights ──
    sample_weights = torch.ones(len(train_dataset))

    def make_loader(weights):
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=2, pin_memory=True)

    train_loader = make_loader(sample_weights)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

    model     = BEVModel(out_channels=out_channels, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg).to(device)
    criterion = AccuracyCriterion(lambda_focal=1.0, lambda_dice=0.5, lambda_depth=0.3).to(device)

    # ── 5. Surgical VRAM: 8-bit AdamW ─────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4, weight_decay=1e-4)
        print("✅ 8-bit AdamW active")
    except ImportError:
        print("⚠️  bitsandbytes not found — using standard AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    # ── 3a. CosineAnnealingWarmRestarts ───────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # ── 3b. Stochastic Weight Averaging ───────────────────────────
    swa_model     = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=5e-5, anneal_epochs=5)
    swa_active    = False

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_model_path     = "bev_model_best_experiment.pth"
    latest_ckpt_path    = "bev_model_latest_experiment.pth"
    start_epoch, best_iou = 0, 0.0

    # Auto-resume
    if os.path.exists(latest_ckpt_path):
        print(f"🔄 Resuming from {latest_ckpt_path}...")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt.get("best_iou", 0.0)
        sample_weights = ckpt.get("sample_weights", sample_weights)
        print(f"   Resumed from Epoch {start_epoch + 1}")

    print("\n🔥 Training: Focal + Dice + Depth | Log-Depth Bins | SWA | HEM\n")
    for epoch in range(start_epoch, num_epochs):

        # ── Activate SWA after swa_start_epoch ────────────────────
        if epoch == swa_start_epoch and not swa_active:
            print(f"⭐ SWA activated at epoch {epoch + 1}")
            swa_active = True

        # Rebuild loader with updated HEM weights each epoch
        if epoch > start_epoch:
            train_loader = make_loader(sample_weights)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []

        for batch_idx, (images, intrinsics, trans, rot, gt_occ, gt_depth) in enumerate(train_loader):
            images, intrinsics = images.to(device), intrinsics.to(device)
            trans, rot         = trans.to(device), rot.to(device)
            gt_occ, gt_depth   = gt_occ.to(device), gt_depth.to(device)

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
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            raw_loss = loss.item() * accumulation_steps
            epoch_losses.append(raw_loss)

            if (batch_idx + 1) % 10 == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0
                print(
                    f"  [{epoch+1}/{num_epochs}][{batch_idx+1}] "
                    f"Loss: {raw_loss:.4f} | "
                    f"Focal: {loss_dict['focal_loss']:.4f} | "
                    f"Dice: {loss_dict['dice_loss']:.4f} | "
                    f"Depth: {loss_dict['depth_loss']:.4f} | "
                    f"VRAM: {mem:.2f}GB"
                )

        # ── Scheduler step ────────────────────────────────────────
        if swa_active:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step(epoch + 1)

        # ── 4. HEM: update sample weights based on epoch avg loss ─
        # Use per-batch losses to weight next epoch's sampler
        # Cap multiplier at 5× to prevent degenerate specialisation
        if epoch_losses:
            losses_t = torch.tensor(epoch_losses, dtype=torch.float32)
            losses_t = losses_t / (losses_t.mean() + 1e-8)
            losses_t = losses_t.clamp(max=5.0)
            # Trim or pad to match dataset size
            n = len(sample_weights)
            if len(losses_t) >= n:
                sample_weights = losses_t[:n]
            else:
                sample_weights = losses_t.repeat((n // len(losses_t)) + 1)[:n]
            sample_weights = (sample_weights + 0.1).clamp(min=0.1)  # floor weight

        # ── Validation ────────────────────────────────────────────
        eval_model = swa_model if swa_active else model
        eval_model.eval()
        v_iou = 0.0

        if device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for v_imgs, v_k, v_t, v_r, v_occ, v_depth in val_loader:
                v_imgs, v_k = v_imgs.to(device), v_k.to(device)
                v_t, v_r    = v_t.to(device), v_r.to(device)
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    v_logits, _ = eval_model(v_imgs, v_k, v_t, v_r)
                    v_mask = torch.sigmoid(v_logits) > 0.5
                v_iou += occupancy_iou(v_mask, v_occ.to(device).bool()).item()

        avg_v_iou = (v_iou / len(val_loader)) * 100
        print(f"\n✅ Epoch {epoch+1} | Val IoU: {avg_v_iou:.2f}% | Best: {best_iou:.2f}%\n")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ── Save full checkpoint every epoch ──────────────────────
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_iou": best_iou,
            "sample_weights": sample_weights,
        }, latest_ckpt_path)
        print(f"💾 Checkpoint saved → {os.path.abspath(latest_ckpt_path)}")

        if avg_v_iou > best_iou:
            best_iou = avg_v_iou
            torch.save(eval_model.state_dict(), best_model_path)
            print(f"⭐ New Best saved → {os.path.abspath(best_model_path)} (IoU: {best_iou:.2f}%)")

    # ── Final: update SWA batch norms if SWA was used ─────────────
    if swa_active and device.type == "cuda":
        print("\n🔁 Updating SWA batch norms...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), "bev_model_swa_final.pth")
        print(f"💾 SWA model saved → bev_model_swa_final.pth")

    torch.save(model.state_dict(), "bev_model_experiment_final.pth")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="./data/nuscenes")
    parser.add_argument("--version",  type=str, default="v1.0-mini")
    parser.add_argument("--epochs",   type=int, default=20)
    parser.add_argument("--swa_start", type=int, default=12)
    args = parser.parse_args()
    train_pipeline(
        dataroot=args.dataroot,
        version=args.version,
        num_epochs=args.epochs,
        swa_start_epoch=args.swa_start,
    )