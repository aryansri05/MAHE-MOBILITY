from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from nuscenes.nuscenes import NuScenes
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

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
from mahe_mobility.models.occupancy import OccupancyCriterion, occupancy_iou, distance_weighted_error
from mahe_mobility.models.bev_occupancy import BEVOccupancyModel
from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION
from mahe_mobility.dataset import NuScenesFrontCameraDataset

# Set allocation config for stability on T4
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# =============================================================
#  Helper: quaternion (w,x,y,z) → 3×3 rotation matrix
# =============================================================
# Kaggle Path Correction: ensure we aren't training in a nested "MAHE-MOBILITY/MAHE-MOBILITY" dir
if os.getcwd().endswith("MAHE-MOBILITY/MAHE-MOBILITY"):
    parent = os.path.dirname(os.getcwd())
    print(f"⚠️ Nested directory detected. Shifting CWD up to: {parent}")
    os.chdir(parent)

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
#  Lift Head — 2-D features → depth-distributed frustum features
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
        
        # ── Triple-Block Checkpointing (Memory Safety) ────
        def get_feat_2d(img):
            return self.feature_extractor(img)
            
        feat_2d = torch.utils.checkpoint.checkpoint(get_feat_2d, images, use_reentrant=False)
        
        def get_frustum(f2d):
            return self.lift_head(f2d)
            
        feat_3d, depth_probs = torch.utils.checkpoint.checkpoint(get_frustum, feat_2d, use_reentrant=False)
        
        def get_bev(f3d, k_mat, trans, rot):
            e2c = build_ego2cam(trans[0].cpu(), rot[0].cpu()).to(device)
            return self.geometry(f3d, k_mat, e2c.unsqueeze(0).expand(f3d.shape[0], -1, -1))
            
        bev_raw = torch.utils.checkpoint.checkpoint(get_bev, feat_3d, intrinsics, translation, rotation, use_reentrant=False)
        
        logits = self.occupancy_model(bev_raw)
        return logits, depth_probs

# =============================================================
#  Training pipeline
# =============================================================

def train_pipeline(
    dataroot: str = "./data/nuscenes",
    version: str = "v1.0-mini",
    batch_size: int = 1, 
    accumulation_steps: int = 4, 
    num_epochs: int = 20,
    out_channels: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\n🚀 Accuracy-First Deployment (Eff. BS: {batch_size * accumulation_steps})")

    cam_cfg = CameraConfig(image_h=224, image_w=480)
    bev_cfg = BEVGridConfig(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION)
    depth_cfg = DepthConfig() # 41 steps

    full_dataset = NuScenesFrontCameraDataset(dataroot=dataroot, version=version, train=True)
    val_full = NuScenesFrontCameraDataset(dataroot=dataroot, version=version, train=False)
    
    total_samples = len(full_dataset)
    val_size = int(0.1 * total_samples)
    train_size = total_samples - val_size

    train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    _, val_dataset = torch.utils.data.random_split(val_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = BEVModel(out_channels=out_channels, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg).to(device)
    criterion = OccupancyCriterion(lambda_depth=1.0, lambda_tv=0.05).to(device)
    
    if bnb is not None:
        print("💎 Using bitsandbytes 8-bit AdamW optimizer.")
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=2e-4, weight_decay=1e-4)
    else:
        print("⚠️ bitsandbytes not found, falling back to standard AdamW.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Paths
    best_model_path = "bev_model_best_v2.pth"
    latest_checkpoint_path = "bev_model_latest.pth"
    
    start_epoch = 0
    best_iou = 0.0

    # 🔄 AUTO-RESUME Logic
    if os.path.exists(latest_checkpoint_path):
        print(f"🔄 Found existing checkpoint: {latest_checkpoint_path}. Resuming...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"   Resuming from Epoch {start_epoch+1}")

    print("Starting Training Loop...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
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
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if (batch_idx + 1) % 10 == 0:
                print(f"  [{epoch+1}/{num_epochs}][{batch_idx+1}] Loss: {loss.item()*accumulation_steps:.4f} (Depth: {loss_dict['depth_loss']:.4f})")

        # --- VALIDATION (Dynamic Thresholding Hook) ---
        model.eval()
        thresholds = torch.linspace(0.1, 0.9, 9).to(device)
        # Confusion matrix components for each threshold: [num_thresholds]
        tps = torch.zeros(len(thresholds), device=device)
        fps = torch.zeros(len(thresholds), device=device)
        fns = torch.zeros(len(thresholds), device=device)
        
        with torch.no_grad():
            for v_imgs, v_k, v_t, v_r, v_occ, v_depth in val_loader:
                v_imgs, v_k = v_imgs.to(device), v_k.to(device)
                v_t, v_r = v_t.to(device), v_r.to(device)
                v_occ = v_occ.to(device).bool()
                
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    v_logits, _ = model(v_imgs, v_k, v_t, v_r)
                    v_probs = torch.sigmoid(v_logits)
                    
                    for i, t in enumerate(thresholds):
                        v_mask = v_probs > t
                        tps[i] += (v_mask & v_occ).sum()
                        fps[i] += (v_mask & ~v_occ).sum()
                        fns[i] += (~v_mask & v_occ).sum()
        
        # Calculate F1 and IoU for all thresholds
        precision = tps / (tps + fps + 1e-6)
        recall = tps / (tps + fns + 1e-6)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        ious = tps / (tps + fps + fns + 1e-6)
        
        best_idx = torch.argmax(f1_scores)
        opt_threshold = thresholds[best_idx].item()
        max_f1 = f1_scores[best_idx].item()
        best_iou_at_f1 = ious[best_idx].item() * 100
        
        print(f"✅ Epoch {epoch+1} complete.")
        print(f"   🎯 Optimal Threshold: {opt_threshold:.2f} | Max F1: {max_f1:.4f} | IoU: {best_iou_at_f1:.2f}%")
        
        avg_v_iou = best_iou_at_f1 # Using IoU at optimal F1 for saving
        
        # 💾 SAVE FULL CHECKPOINT EVERY EPOCH
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
        }, latest_checkpoint_path)
        print(f"💾 Saved latest checkpoint to: {os.path.abspath(latest_checkpoint_path)}")

        if avg_v_iou > best_iou:
            best_iou = avg_v_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"⭐ New Best Model Saved to: {os.path.abspath(best_model_path)} (IoU: {best_iou:.2f}%)")

    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="./data/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train_pipeline(dataroot=args.dataroot, version=args.version, num_epochs=args.epochs)