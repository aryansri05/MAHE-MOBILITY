"""
evaluate_local.py
-----------------
Official evaluation script for MAHE Mobility BEV Perception.
Produces a comprehensive 7-panel diagnostic visualization and calculates IoU metrics.

Usage:
    python evaluate_local.py --sample 4 --out results.png
    python evaluate_local.py (runs random sample)
"""

import sys
import os
from scipy.ndimage import binary_erosion
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from pyquaternion import Quaternion

sys.path.append('src')
sys.path.append('scripts')

from mahe_mobility.geometry.lss_core import CameraConfig, BEVGridConfig, DepthConfig
from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION, OCC_THRESHOLD
from mahe_mobility.dataset import NuScenesFrontCameraDataset
from pipeline import BEVModel

# Constants
WEIGHTS = "bev_model_final_v2 (1).pth"  # Best ResNet34 model from Kaggle training
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])
BEV_EXTENT = [X_MIN, X_MAX, Y_MIN, Y_MAX]

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * _STD + _MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def project_box_to_2d(box, cam_K, img_w, img_h):
    corners = box.corners()
    if corners[2].min() < 0.1:
        return None
    c2d = cam_K @ corners
    c2d[:2] /= c2d[2]
    xs, ys = c2d[0], c2d[1]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    if x1 < 0 or y1 < 0 or x0 > img_w or y0 > img_h:
        return None
    return (max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1))

def get_depth_map(model, depth_cfg):
    if not hasattr(model.lift_head, 'last_depth_dist'):
        return None
    depth_dist = model.lift_head.last_depth_dist
    bins = torch.linspace(depth_cfg.d_min, depth_cfg.d_max, depth_cfg.d_steps, device=depth_dist.device)
    return (depth_dist[0] * bins.view(-1, 1, 1)).sum(0).cpu().numpy()

def style_ax(ax, title, color="white"):
    ax.set_title(title, color=color, fontsize=9, pad=5)
    ax.tick_params(colors=color, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_facecolor("#060610")

def add_ego_marker(ax):
    ax.plot(0, Y_MIN + 0.5, marker="^", color="cyan", markersize=7, zorder=6, clip_on=False)

# ─────────────────────────────────────────────────────────────────────────────
#  Core Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(sample_idx=None, save_path="hackathon_final_plot.png", weights=WEIGHTS):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🧠 Loading model from '{weights}' onto {device}...")

    cam_cfg = CameraConfig(image_h=224, image_w=480)
    bev_cfg = BEVGridConfig(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION)
    depth_cfg = DepthConfig()
    model = BEVModel(out_channels=64, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg)

    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()
    print("✅ Weights loaded!")

    dataset = NuScenesFrontCameraDataset(dataroot="./data/nuscenes", version="v1.0-mini", train=False)
    
    if sample_idx is None:
        import random
        sample_idx = random.randint(0, len(dataset) - 1)
        print(f"📸 Picking random sample: {sample_idx}")
    else:
        print(f"📸 Using specified sample: {sample_idx}")

    nusc = dataset.nusc
    sample_data = dataset.samples[sample_idx]
    # Dataset returns: img, intrinsics, translation, rotation, occupancy, depth
    img_t, intrinsics, trans, rot, gt_occ, gt_depth = dataset[sample_idx]

    with torch.no_grad():
        # AMP: float16 on CUDA for speed boost, no-op on CPU/MPS
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) \
            if device.type == 'cuda' else torch.amp.autocast(device_type='cpu', enabled=False)

        with autocast_ctx:
            bev_logits = model(
                img_t.unsqueeze(0).to(device),
                intrinsics.unsqueeze(0).to(device),
                trans.unsqueeze(0).to(device),
                rot.unsqueeze(0).to(device)
            )
        bev_logits = bev_logits.float()  # cast back to float32 for metric computation

    # Process outputs
    cam_img = unnormalize(img_t)
    depth_map = get_depth_map(model, depth_cfg)
    pred_probs = torch.sigmoid(bev_logits[0, 0]).cpu().numpy()
    
    # ── Thresholding & Morphological Refinement ───────────────
    preds_binary = (pred_probs > OCC_THRESHOLD).astype(np.uint8)
    # Erosion strips thin false-positive edges
    kernel = np.ones((3, 3), bool)
    preds_binary = binary_erosion(preds_binary, structure=kernel).astype(np.uint8)
    
    gt = gt_occ[0].numpy()
    gt_d_map = gt_depth.numpy() 

    # ── Calculate Metrics ─────────────────────────────────────
    gt_binary = gt.astype(np.uint8)
    intersection = np.logical_and(preds_binary, gt_binary).sum()
    union = np.logical_or(preds_binary, gt_binary).sum()
    iou_score = (intersection / union) * 100 if union > 0 else 0.0

    TP = (preds_binary * gt_binary).sum()
    FP = (preds_binary * (1 - gt_binary)).sum()
    FN = ((1 - preds_binary) * gt_binary).sum()
    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1_score  = 2 * (precision * recall) / (precision + recall + 1e-6)

    # ── Distance-Weighted Error (20× penalty in 0–10 m zone) ─
    errors = np.abs(preds_binary.astype(float) - gt_binary.astype(float))
    weights_mat = np.ones_like(errors, dtype=float)
    ten_meter_mark = int(errors.shape[0] * 0.8)  # bottom 20% of grid ≈ 0–10 m
    weights_mat[ten_meter_mark:, :] = 20.0
    dwe = (errors * weights_mat).sum() / weights_mat.sum()

    print(f"\n📊 FINAL METRICS:")
    print(f"   IoU Score  : {iou_score:.2f}%")
    print(f"   Precision  : {precision:.4f}")
    print(f"   Recall     : {recall:.4f}")
    print(f"   F1 Score   : {f1_score:.4f}")
    print(f"   DWE        : {dwe:.4f}  (lower is better)")

    # ── Error Map for Visualization ──────────────────────────
    err = np.zeros_like(gt, dtype=int)
    pb, gb = (preds_binary > 0.5), (gt_binary > 0.5)
    err[pb & gb] = 1   # TP
    err[pb & ~gb] = 2  # FP
    err[~pb & gb] = 3  # FN

    # ── NuScenes 2D Box Projections ──────────────────────────
    cam_token = sample_data["data"]["CAM_FRONT"]
    cam_sd = nusc.get("sample_data", cam_token)
    calib = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    cam_K = np.array(calib["camera_intrinsic"])
    cam_t = np.array(calib["translation"])

    boxes_2d = []
    for box in nusc.get_boxes(cam_token):
        ego_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])
        box.translate(-np.array(ego_pose["translation"]))
        box.rotate(Quaternion(ego_pose["rotation"]).inverse)
        box.translate(-cam_t)
        box.rotate(Quaternion(calib["rotation"]).inverse)
        res = project_box_to_2d(box, cam_K, cam_img.shape[1], cam_img.shape[0])
        if res: boxes_2d.append(res)

    # ── Plotting ──────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 11), facecolor="#0a0a14")
    gs = gridspec.GridSpec(2, 12, figure=fig, hspace=0.45, wspace=0.35, left=0.04, right=0.97, top=0.91, bottom=0.06)

    ax_cam = fig.add_subplot(gs[0, 0:3])
    ax_depth = fig.add_subplot(gs[0, 3:6])
    ax_depth_gt = fig.add_subplot(gs[0, 6:9])
    ax_bev = fig.add_subplot(gs[0, 9:12])
    ax_gt = fig.add_subplot(gs[1, 0:3])
    ax_prob = fig.add_subplot(gs[1, 3:6])
    ax_bin = fig.add_subplot(gs[1, 6:9])
    ax_err = fig.add_subplot(gs[1, 9:12])

    # Panels
    ax_cam.imshow(cam_img)
    for (x0, y0, x1, y1) in boxes_2d:
        ax_cam.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1.5, edgecolor="#00ff88", facecolor="none"))
    style_ax(ax_cam, "Camera View + Detections")
    ax_cam.axis("off")

    if depth_map is not None:
        im1 = ax_depth.imshow(depth_map, cmap="plasma", vmin=depth_cfg.d_min, vmax=depth_cfg.d_max)
        cb1 = plt.colorbar(im1, ax=ax_depth, fraction=0.035, pad=0.03)
        cb1.ax.tick_params(colors="white", labelsize=7)
    style_ax(ax_depth, "Predicted Depth (LSS Head)")
    ax_depth.axis("off")

    im_gt_d = ax_depth_gt.imshow(gt_d_map, cmap="plasma", vmin=0, vmax=depth_cfg.d_steps)
    cb_gt_d = plt.colorbar(im_gt_d, ax=ax_depth_gt, fraction=0.035, pad=0.03)
    cb_gt_d.ax.tick_params(colors="white", labelsize=7)
    style_ax(ax_depth_gt, "Ground Truth Depth (LiDAR)")
    ax_depth_gt.axis("off")

    ax_bev.set_facecolor("#040408")
    orows, ocols = np.where(preds_binary > 0.5)
    ax_bev.scatter(X_MIN + (ocols+0.5)*RESOLUTION, Y_MIN + (orows+0.5)*RESOLUTION, s=4, c="#f0e040", marker="s", linewidths=0)
    ax_bev.set_xlim(X_MIN, X_MAX); ax_bev.set_ylim(Y_MIN, Y_MAX)
    add_ego_marker(ax_bev); ax_bev.grid(True, color="#111", linewidth=0.4)
    style_ax(ax_bev, "BEV Occupancy Grid")

    ax_gt.imshow(np.flipud(gt), cmap="gray", extent=BEV_EXTENT, origin="lower")
    add_ego_marker(ax_gt); style_ax(ax_gt, "Ground Truth (LiDAR)")

    im4 = ax_prob.imshow(np.flipud(pred_probs), cmap="hot", extent=BEV_EXTENT, origin="lower", vmin=0, vmax=1)
    plt.colorbar(im4, ax=ax_prob, fraction=0.046).ax.tick_params(colors="white", labelsize=6)
    add_ego_marker(ax_prob); style_ax(ax_prob, "Prediction Probability")

    ax_bin.imshow(np.flipud(preds_binary), cmap="gray", extent=BEV_EXTENT, origin="lower")
    add_ego_marker(ax_bin); style_ax(ax_bin, f"Eroded Binary Prediction (IoU={iou_score:.1f}%)")

    err_cmap = matplotlib.colors.ListedColormap(["#0a0a18", "#00c49b", "#e63946", "#f4a261"])
    ax_err.imshow(np.flipud(err), cmap=err_cmap, extent=BEV_EXTENT, origin="lower", vmin=0, vmax=3)
    legend_patches = [
        mpatches.Patch(color="#0a0a18", label="TN"), mpatches.Patch(color="#00c49b", label="TP"),
        mpatches.Patch(color="#e63946", label="FP"), mpatches.Patch(color="#f4a261", label="FN")
    ]
    ax_err.legend(handles=legend_patches, loc="upper right", fontsize=6, labelcolor="white", facecolor="#111")
    add_ego_marker(ax_err); style_ax(ax_err, "Error Map")

    plt.suptitle(f"MAHE Mobility BEV Evaluation | Sample {sample_idx} | IoU {iou_score:.2f}% | DWE {dwe:.4f}", color="white", fontsize=13, y=0.97)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Results saved to '{save_path}'")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Specific sample index")
    parser.add_argument("--out", type=str, default="hackathon_final_plot.png", help="Output filename")
    parser.add_argument("--weights", type=str, default=WEIGHTS, help="Weights path")
    args = parser.parse_args()
    
    run_evaluation(sample_idx=args.sample, save_path=args.out, weights=args.weights)
