from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
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
from mahe_mobility.models.occupancy import OccupancyCriterion, occupancy_iou, distance_weighted_error
from mahe_mobility.models.bev_occupancy import BEVOccupancyModel
from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION
from mahe_mobility.dataset import NuScenesFrontCameraDataset

# Set allocation config for stability on T4
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# =============================================================
#  Helper: quaternion (w,x,y,z) → 3×3 rotation matrix
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
#  BEVModel — full end-to-end nn.Module
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
        
        # 🧪 ACCURACY PUSH: Using Gradient Checkpointing for 2D backbone
        def custom_forward(img):
            f2d = self.feature_extractor(img)
            f3d, dp = self.lift_head(f2d)
            return f3d, dp

        # Wrap in checkpointing to save ~5GB of VRAM
        feat_3d, depth_probs = torch.utils.checkpoint.checkpoint(
            custom_forward, images, use_reentrant=False
        )
        
        # Extrinsics handling
        ego2cam = build_ego2cam(translation[0].cpu(), rotation[0].cpu()).to(device)
        
        # Voxel pool (cached internally)
        bev_raw = self.geometry(feat_3d, intrinsics, ego2cam.unsqueeze(0).expand(feat_3d.shape[0], -1, -1))
        
        logits = self.occupancy_model(bev_raw)
        return logits, depth_probs

# =============================================================
#  Training pipeline
# =============================================================

def train_pipeline(
    dataroot: str = "./data/nuscenes",
    version: str = "v1.0-mini",
    batch_size: int = 1, # BS=1 for absolute stability
    accumulation_steps: int = 4, # Effective BS = 4
    num_epochs: int = 20,
    out_channels: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\n🚀 Accuracy-First: Initializing Pipeline on: {device} (Eff. BS: {batch_size * accumulation_steps})")

    cam_cfg = CameraConfig(image_h=224, image_w=480)
    bev_cfg = BEVGridConfig(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION)
    depth_cfg = DepthConfig() # Uses default 41 steps for Accuracy

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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    checkpoint_path = "bev_checkpoint_v2.pth"
    best_model_path = "bev_model_best_v2.pth"
    best_iou = 0.0

    print("Starting Training with ACTIVATION CHECKPOINTING & MODEL CHECKPOINTING...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
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
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if (batch_idx + 1) % 10 == 0:
                mem_used = torch.cuda.max_memory_allocated() / 1024**3 if device.type=="cuda" else 0
                print(f"  [{epoch+1}/{num_epochs}][{batch_idx+1}] Loss: {loss.item()*accumulation_steps:.4f} (Depth: {loss_dict['depth_loss']:.4f}) | peak: {mem_used:.2f}GB")

            # 💾 ACCURACY PUSH: Frequent Model Checkpointing
            if (batch_idx + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                }, checkpoint_path)

        # --- VALIDATION ---
        model.eval()
        v_iou = 0.0
        with torch.no_grad():
            for v_imgs, v_k, v_t, v_r, v_occ, v_depth in val_loader:
                v_imgs, v_k = v_imgs.to(device), v_k.to(device)
                v_t, v_r = v_t.to(device), v_r.to(device)
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    v_logits, _ = model(v_imgs, v_k, v_t, v_r)
                    v_mask = torch.sigmoid(v_logits) > 0.5
                v_iou += occupancy_iou(v_mask, v_occ.to(device).bool()).item()
        
        avg_v_iou = (v_iou / len(val_loader)) * 100
        print(f"✅ Epoch {epoch+1} complete. Val IoU: {avg_v_iou:.2f}%")
        if avg_v_iou > best_iou:
            best_iou = avg_v_iou
            torch.save(model.state_dict(), best_model_path)

    torch.save(model.state_dict(), "bev_model_final_push.pth")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="./data/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train_pipeline(dataroot=args.dataroot, version=args.version, num_epochs=args.epochs)