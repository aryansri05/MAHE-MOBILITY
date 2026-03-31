from __future__ import annotations

import sys
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from nuscenes.nuscenes import NuScenes

# ── Ensure src is in path ──────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Internal Modules ───────────────────────────────────────
from mahe_mobility.models.resnet_extractor import ResNetFeatureExtractor
from mahe_mobility.geometry.lss_core import (
    CameraConfig,
    BEVGridConfig,
    DepthConfig,
    GeometryArchitect,
    DepthPrecomputer,
)
from mahe_mobility.tasks.task1_lidar_to_occupancy import load_lidar_ego_frame, lidar_to_occupancy
from mahe_mobility.models.occupancy import occupancy_iou, distance_weighted_error
from mahe_mobility.models.bev_occupancy import BEVOccupancyModel
from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION
from mahe_mobility.dataset import NuScenesFrontCameraDataset

# Set allocation config for stability on T4
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# =============================================================
#  Custom Loss Functions
# =============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        inputs = torch.clamp(inputs, 1e-6, 1.0 - 1e-6)
        bce = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_weight * focal_weight * bce
        if self.reduction == 'mean': return loss.mean()
        return loss.sum()

class DiceLoss(nn.Module):
    """Dice Loss — directly optimizes for IoU/F1 overlap."""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# =============================================================
#  Geometry Helpers
# =============================================================

def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """NuScenes quaternion (w,x,y,z) → (3,3) rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]).reshape(3, 3)

def build_ego2cam(translation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    R = quat_to_rot(rotation)
    cam2ego = torch.eye(4, dtype=torch.float32)
    cam2ego[:3, :3] = R
    cam2ego[:3, 3] = translation
    return torch.linalg.inv(cam2ego)

# =============================================================
#  Model Components
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
        self.last_depth_dist = depth_dist.detach()
        return feat_up.unsqueeze(2) * depth_dist.unsqueeze(1), depth_dist

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
        
        # ── Triple-Block Checkpointing (Memory Safety) ────
        def get_feat_2d(img):
            return self.feature_extractor(img)
            
        feat_2d = torch.utils.checkpoint.checkpoint(get_feat_2d, images, use_reentrant=False)
        
        def get_frustum(f2d):
            return self.lift_head(f2d)
            
        feat_3d, depth_probs = torch.utils.checkpoint.checkpoint(get_frustum, feat_2d, use_reentrant=False)
        
        def get_bev(f3d, k_mat, trans, rot):
            # Sample index 0 extrinsics for the batch
            e2c = build_ego2cam(trans[0].cpu(), rot[0].cpu()).to(device)
            # Recompute precomp indexing (non-differentiable)
            with torch.no_grad():
                self.geometry.precomp = DepthPrecomputer(self.geometry.frustum_gen, self.bev_cfg, e2c).to(device)
            return self.geometry(f3d)
            
        bev_raw = torch.utils.checkpoint.checkpoint(get_bev, feat_3d, intrinsics, translation, rotation, use_reentrant=False)
        
        logits = self.occupancy_model(bev_raw)
        return logits, depth_probs

# =============================================================
#  Training pipeline
# =============================================================

def train_pipeline(
    dataroot: str = "./data/nuscenes",
    version: str = "v1.0-mini",
    batch_size: int = 2,
    num_epochs: int = 1,
    out_channels: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\n🚀 Initializing BEV Training on: {device}")

    cam_cfg = CameraConfig(image_h=224, image_w=480)
    bev_cfg = BEVGridConfig(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION)
    depth_cfg = DepthConfig() # 41 steps

    full_dataset = NuScenesFrontCameraDataset(dataroot=dataroot, version=version, train=True)
    val_full = NuScenesFrontCameraDataset(dataroot=dataroot, version=version, train=False)
    
    total_samples = len(full_dataset)
    val_size = int(0.1 * total_samples)
    train_size = total_samples - val_size

    # Split with fixed seed
    train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    _, val_dataset = torch.utils.data.random_split(val_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"📊 Dataset split: {train_size} train, {val_size} validation samples.")

    # Model, Loss, Optimizer
    model = BEVModel(out_channels=out_channels, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg).to(device)
    
    criterion_focal = FocalLoss(alpha=0.85, gamma=2.0).to(device)
    criterion_dice  = DiceLoss(smooth=1.0).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Paths
    best_model_path = "bev_model_best_v2.pth"
    latest_checkpoint_path = "bev_checkpoint_v2.pth"
    
    start_epoch = 0
    best_iou = 0.0

    if os.path.exists(latest_checkpoint_path):
        print(f"🔄 Resuming from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"   Starting from Epoch {start_epoch + 1}")

    print("Starting Training Loop...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for batch_idx, (images, intrinsics, trans, rot, gt_occ, gt_depth) in enumerate(train_loader):
            images, intrinsics = images.to(device), intrinsics.to(device)
            trans, rot = trans.to(device), rot.to(device)
            gt_occ = gt_occ.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits, _ = model(images, intrinsics, trans, rot)
                probs = torch.sigmoid(logits)
                # Combined Loss: Focal for balancing, Dice for shape
                loss = criterion_focal(probs, gt_occ.float()) + criterion_dice(probs, gt_occ.float())

            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
                optimizer.step()

            torch.cuda.empty_cache()

            if (batch_idx + 1) % 10 == 0:
                with torch.no_grad():
                    mask = probs > 0.5
                    iou = occupancy_iou(mask, gt_occ.bool())
                print(f"  [{epoch+1}/{num_epochs}][{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}  Train-IoU: {iou.item()*100:.2f}%")

        # --- VALIDATION LOOP ---
        print("\n🔍 Running Validation...")
        model.eval()
        val_iou, val_dwe = 0.0, 0.0
        with torch.no_grad():
            for v_imgs, v_k, v_t, v_r, v_occ, _ in val_loader:
                v_imgs, v_k = v_imgs.to(device), v_k.to(device)
                v_t, v_r, v_occ = v_t.to(device), v_r.to(device), v_occ.to(device)
                
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    v_logits, _ = model(v_imgs, v_k, v_t, v_r)
                    v_probs = torch.sigmoid(v_logits)
                    v_mask = v_probs > 0.5
                
                val_iou += occupancy_iou(v_mask, v_occ.bool()).item()
                val_dwe += distance_weighted_error(v_probs, v_occ).item()
        
        avg_v_iou = (val_iou / len(val_loader)) * 100
        avg_v_dwe = val_dwe / len(val_loader)
        print(f"✅ Epoch {epoch+1} Complete. Val IoU: {avg_v_iou:.2f}% | Val DWE: {avg_v_dwe:.4f}")
        
        scheduler.step()

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
        }, latest_checkpoint_path)

        if avg_v_iou > best_iou:
            best_iou = avg_v_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 New Best Model! Saved to {best_model_path} (IoU: {best_iou:.2f}%)")

    # Export final model
    torch.save(model.state_dict(), "bev_model_final_v2.pth")
    print("💾 Final model saved to bev_model_final_v2.pth")
    
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="./data/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    train_pipeline(dataroot=args.dataroot, version=args.version, num_epochs=args.epochs)