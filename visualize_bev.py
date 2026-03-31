"""
visualize_bev.py
----------------
Produces a 3-panel visualization matching the hackathon sample output:
  [0] Camera View + GT Annotation Boxes
  [1] Depth Map (metres) — from the LSS LiftHead
  [2] BEV Occupancy Grid — model prediction from above
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from PIL import Image
from pyquaternion import Quaternion

sys.path.append('src')
sys.path.append('scripts')

from nuscenes.nuscenes import NuScenes
from mahe_mobility.geometry.lss_core import CameraConfig, BEVGridConfig, DepthConfig
from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION
from mahe_mobility.dataset import NuScenesFrontCameraDataset
from pipeline import BEVModel

# ImageNet un-normalisation (reverses the Normalize transform for display)
_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])


def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor back to a displayable HWC uint8 image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)   # CHW → HWC
    img = img * _STD + _MEAN                         # un-normalise
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def project_box_to_2d(box, cam_intrinsic: np.ndarray, img_w: int, img_h: int):
    """
    Project a NuScenes Box's 8 3D corners into 2D pixel coords.
    Returns (x_min, y_min, x_max, y_max) or None if box is behind camera.
    """
    corners_3d = box.corners()              # (3, 8) in camera frame
    # Check that the box is in front of the camera
    if corners_3d[2].min() < 0.1:
        return None
    # Project using intrinsic matrix
    corners_2d = cam_intrinsic @ corners_3d  # (3, 8)
    corners_2d[:2] /= corners_2d[2]         # perspective divide
    xs, ys = corners_2d[0], corners_2d[1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    # Clip to image and discard tiny/oob boxes
    if x_max < 0 or y_max < 0 or x_min > img_w or y_min > img_h:
        return None
    return (max(0, x_min), max(0, y_min),
            min(img_w, x_max), min(img_h, y_max))


def get_depth_map(model: BEVModel, depth_cfg: DepthConfig) -> np.ndarray:
    """
    Compute per-pixel expected depth (metres) from the cached LiftHead
    depth distribution. Returns an (H, W) float array.
    """
    depth_dist = model.lift_head.last_depth_dist  # (1, D, H, W)
    depth_bins = torch.linspace(
        depth_cfg.d_min, depth_cfg.d_max, depth_cfg.d_steps,
        device=depth_dist.device
    )                                              # (D,)
    # Expected depth = sum over bins of bin_value * probability
    expected = (depth_dist[0] * depth_bins.view(-1, 1, 1)).sum(dim=0)  # (H, W)
    return expected.cpu().numpy()


def run_visualization(sample_idx: int = 0, save_path: str = "bev_sample.png", weights: str = "bev_model_best_v2.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🧠 Loading model onto {device} from '{weights}'...")

    cam_cfg   = CameraConfig(image_h=224, image_w=480)
    bev_cfg   = BEVGridConfig(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION)
    depth_cfg = DepthConfig()

    # --- Load model ---
    model = BEVModel(out_channels=64, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg)
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    # bev_model_best_v2.pth is a plain state_dict; bev_checkpoint dicts have a 'model_state_dict' key
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # plain state_dict from best-model saves
    model.to(device)
    model.eval()
    print("✅ Weights loaded!")

    # --- Load ONE sample ---
    dataset = NuScenesFrontCameraDataset(dataroot="./data/nuscenes", version="v1.0-mini", train=False)
    nusc    = dataset.nusc
    sample  = dataset.samples[sample_idx]

    image_tensor, intrinsics, trans, rot, gt_occupancy = dataset[sample_idx]

    with torch.no_grad():
        bev_logits = model(
            image_tensor.unsqueeze(0).to(device),
            intrinsics.unsqueeze(0).to(device),
            trans.unsqueeze(0).to(device),
            rot.unsqueeze(0).to(device),
        )

    # --- Panel data ---
    cam_img_np   = unnormalize(image_tensor)              # HWC uint8
    depth_map    = get_depth_map(model, depth_cfg)         # H×W metres
    bev_probs    = torch.sigmoid(bev_logits[0, 0]).cpu().numpy()  # H×W [0,1]
    bev_binary   = (bev_probs > 0.5).astype(float)

    # --- NuScenes GT annotation boxes on camera ---
    cam_token    = sample["data"]["CAM_FRONT"]
    cam_data     = nusc.get("sample_data", cam_token)
    calib        = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    cam_K        = np.array(calib["camera_intrinsic"])    # (3,3)
    cam_t        = np.array(calib["translation"])
    cam_R        = Quaternion(calib["rotation"]).rotation_matrix

    boxes_2d = []
    for box in nusc.get_boxes(cam_token):
        # Transform 3D box from global → cam frame
        # Step 1: global → ego using ego pose
        ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
        box.translate(-np.array(ego_pose["translation"]))
        box.rotate(Quaternion(ego_pose["rotation"]).inverse)
        # Step 2: ego → camera using calibrated sensor
        box.translate(-cam_t)
        box.rotate(Quaternion(calib["rotation"]).inverse)
        result = project_box_to_2d(box, cam_K, cam_img_np.shape[1], cam_img_np.shape[0])
        if result:
            boxes_2d.append(result)

    # ── Build the 3-panel figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    label_kw = dict(color="white", fontsize=8)

    # Panel 0: Camera View + Boxes
    axes[0].imshow(cam_img_np)
    for (x0, y0, x1, y1) in boxes_2d:
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.5, edgecolor="#00ff88", facecolor="none"
        )
        axes[0].add_patch(rect)
    axes[0].set_title(f"Camera View + Detections (sample {sample_idx})", color="white", fontsize=10)
    axes[0].axis("off")

    # Panel 1: Depth Map
    im1 = axes[1].imshow(depth_map, cmap="plasma", vmin=depth_cfg.d_min, vmax=depth_cfg.d_max)
    axes[1].set_title(f"Depth Map (metres) (sample {sample_idx})", color="white", fontsize=10)
    axes[1].axis("off")
    cbar = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("depth (m)", color="white", fontsize=8)

    # Panel 2: BEV Occupancy Grid
    bev_extent = [X_MIN, X_MAX, Y_MIN, Y_MAX]
    axes[2].imshow(
        np.flipud(bev_binary), cmap="plasma",
        extent=bev_extent, origin="lower", vmin=0, vmax=1
    )
    axes[2].set_title(f"BEV Occupancy Grid (sample {sample_idx})", color="white", fontsize=10)
    axes[2].set_xlabel("X lateral (m)", **label_kw)
    axes[2].set_ylabel("Z forward (m)", **label_kw)
    axes[2].tick_params(colors="white")
    axes[2].plot(0, Y_MIN, marker="^", color="cyan", markersize=8, zorder=5, label="ego")
    axes[2].legend(fontsize=7, labelcolor="white", facecolor="#1a1a1a")

    plt.suptitle("BEV Occupancy Visualization", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Visualization saved to '{save_path}'")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--out",    type=str, default="bev_sample.png", help="Output image path")
    parser.add_argument("--weights", type=str, default="bev_model_best_v2.pth", help="Path to model weights")
    args = parser.parse_args()
    run_visualization(sample_idx=args.sample, save_path=args.out, weights=args.weights)
