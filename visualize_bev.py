"""
visualize_bev.py
----------------
Comprehensive 7-panel BEV visualization combining:
  TOP ROW (3 panels):
    [0] Camera View + GT Annotation Boxes
    [1] Depth Map (metres) from LSS LiftHead
    [2] BEV Occupancy Grid — scatter dot style

  BOTTOM ROW (4 panels):
    [3] Ground Truth occupancy (from LiDAR)
    [4] Prediction probabilities (heatmap)
    [5] Binary prediction (threshold=0.5)
    [6] Error map — TP / FP / FN / TN colour-coded
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from pyquaternion import Quaternion

sys.path.append('src')
sys.path.append('scripts')

from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader
from mahe_mobility.geometry.lss_core import CameraConfig, BEVGridConfig, DepthConfig
from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION, GRID_H, GRID_W, OCC_THRESHOLD
from mahe_mobility.dataset import NuScenesFrontCameraDataset
from pipeline import BEVModel

WEIGHTS   = "bev_model_best_v2.pth"
_MEAN     = np.array([0.485, 0.456, 0.406])
_STD      = np.array([0.229, 0.224, 0.225])
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
    depth_dist = model.lift_head.last_depth_dist          # (1, D, H, W)
    bins = torch.linspace(depth_cfg.d_min, depth_cfg.d_max,
                          depth_cfg.d_steps, device=depth_dist.device)
    return (depth_dist[0] * bins.view(-1, 1, 1)).sum(0).cpu().numpy()


def style_ax(ax, title, color="white"):
    ax.set_title(title, color=color, fontsize=9, pad=5)
    ax.tick_params(colors=color, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_facecolor("#060610")


def add_ego_marker(ax):
    ax.plot(0, Y_MIN + 0.5, marker="^", color="cyan",
            markersize=7, zorder=6, clip_on=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def run_visualization(sample_idx: int = 0,
                      save_path: str  = "bev_sample.png",
                      weights: str    = WEIGHTS):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🧠  Loading model [{weights}] on {device}...")

    cam_cfg   = CameraConfig(image_h=224, image_w=480)
    bev_cfg   = BEVGridConfig(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION)
    depth_cfg = DepthConfig()

    model = BEVModel(out_channels=64, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg)
    ckpt  = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt)
    model.to(device).eval()
    print("✅  Weights loaded!")

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = NuScenesFrontCameraDataset(dataroot="./data/nuscenes", version="v1.0-mini", train=False)
    nusc    = dataset.nusc
    sample  = dataset.samples[sample_idx]

    img_t, intrinsics, trans, rot, gt_occ = dataset[sample_idx]

    with torch.no_grad():
        bev_logits = model(
            img_t.unsqueeze(0).to(device),
            intrinsics.unsqueeze(0).to(device),
            trans.unsqueeze(0).to(device),
            rot.unsqueeze(0).to(device),
        )

    # ── Derived arrays ────────────────────────────────────────────────────────
    cam_img    = unnormalize(img_t)
    depth_map  = get_depth_map(model, depth_cfg)
    pred_probs = torch.sigmoid(bev_logits[0, 0]).cpu().numpy()
    pred_bin   = (pred_probs > OCC_THRESHOLD).astype(float)
    gt         = gt_occ[0].numpy()

    # Error map: 0=TN, 1=TP, 2=FP, 3=FN
    err = np.zeros_like(gt, dtype=int)
    pb  = pred_bin.astype(bool)
    gb  = (gt > 0.5)
    err[pb  &  gb] = 1
    err[pb  & ~gb] = 2
    err[~pb &  gb] = 3

    # IoU for title
    inter = (pb & gb).sum()
    union = (pb | gb).sum()
    iou   = inter / union * 100 if union > 0 else 0.0
    print(f"📊  IoU: {iou:.2f}%")

    # ── NuScenes GT boxes ─────────────────────────────────────────────────────
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data  = nusc.get("sample_data", cam_token)
    calib     = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    cam_K     = np.array(calib["camera_intrinsic"])
    cam_t     = np.array(calib["translation"])

    boxes_2d = []
    for box in nusc.get_boxes(cam_token):
        ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
        box.translate(-np.array(ego_pose["translation"]))
        box.rotate(Quaternion(ego_pose["rotation"]).inverse)
        box.translate(-cam_t)
        box.rotate(Quaternion(calib["rotation"]).inverse)
        r = project_box_to_2d(box, cam_K, cam_img.shape[1], cam_img.shape[0])
        if r:
            boxes_2d.append(r)

    # ── BEV scatter coords ────────────────────────────────────────────────────
    occ_rows, occ_cols = np.where(pred_bin > 0.5)
    bev_x = X_MIN + (occ_cols + 0.5) * RESOLUTION
    bev_y = Y_MIN + (occ_rows + 0.5) * RESOLUTION

    gt_rows, gt_cols = np.where(gt > 0.5)
    gt_x = X_MIN + (gt_cols + 0.5) * RESOLUTION
    gt_y = Y_MIN + (gt_rows + 0.5) * RESOLUTION

    # ── Figure layout: 2 rows ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 11), facecolor="#0a0a14")
    gs  = gridspec.GridSpec(
        2, 12, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.04, right=0.97, top=0.91, bottom=0.06
    )

    # Top row — 3 × 4-col panels
    ax_cam   = fig.add_subplot(gs[0, 0:4])
    ax_depth = fig.add_subplot(gs[0, 4:8])
    ax_bev   = fig.add_subplot(gs[0, 8:12])

    # Bottom row — 4 × 3-col panels
    ax_gt    = fig.add_subplot(gs[1, 0:3])
    ax_prob  = fig.add_subplot(gs[1, 3:6])
    ax_bin   = fig.add_subplot(gs[1, 6:9])
    ax_err   = fig.add_subplot(gs[1, 9:12])

    lbl_kw = dict(color="#aaa", fontsize=7)

    # ── Panel 0: Camera View + GT Boxes ──────────────────────────────────────
    ax_cam.imshow(cam_img)
    for (x0, y0, x1, y1) in boxes_2d:
        ax_cam.add_patch(patches.Rectangle(
            (x0, y0), x1-x0, y1-y0,
            linewidth=1.5, edgecolor="#00ff88", facecolor="none"
        ))
    style_ax(ax_cam, f"Camera View + Detections (sample {sample_idx})")
    ax_cam.axis("off")

    # ── Panel 1: Depth Map ────────────────────────────────────────────────────
    im1 = ax_depth.imshow(depth_map, cmap="plasma",
                          vmin=depth_cfg.d_min, vmax=depth_cfg.d_max)
    style_ax(ax_depth, f"Depth Map (metres) (sample {sample_idx})")
    ax_depth.axis("off")
    cb1 = plt.colorbar(im1, ax=ax_depth, fraction=0.035, pad=0.03)
    cb1.ax.tick_params(colors="white", labelsize=7)
    cb1.set_label("depth (m)", color="white", fontsize=7)

    # ── Panel 2: BEV Scatter Grid (dot style like reference image) ────────────
    ax_bev.set_facecolor("#040408")
    ax_bev.scatter(bev_x, bev_y, s=4, c="#f0e040", marker="s",
                   linewidths=0, alpha=0.9, label="predicted")
    ax_bev.set_xlim(X_MIN, X_MAX)
    ax_bev.set_ylim(Y_MIN, Y_MAX)
    ax_bev.set_xlabel("X lateral (m)", **lbl_kw)
    ax_bev.set_ylabel("Z forward (m)", **lbl_kw)
    add_ego_marker(ax_bev)
    ax_bev.grid(True, color="#111", linewidth=0.4)
    style_ax(ax_bev, f"BEV Occupancy Grid (sample {sample_idx})")

    # ── Panel 3: Ground Truth ─────────────────────────────────────────────────
    ax_gt.imshow(np.flipud(gt), cmap="gray",
                 extent=BEV_EXTENT, origin="lower", vmin=0, vmax=1)
    ax_gt.set_xlabel("X (m)", **lbl_kw)
    ax_gt.set_ylabel("Y (m)", **lbl_kw)
    add_ego_marker(ax_gt)
    style_ax(ax_gt, "Ground Truth (LiDAR)")

    # ── Panel 4: Prediction Probability ──────────────────────────────────────
    im4 = ax_prob.imshow(np.flipud(pred_probs), cmap="hot",
                          extent=BEV_EXTENT, origin="lower", vmin=0, vmax=1)
    ax_prob.set_xlabel("X (m)", **lbl_kw)
    ax_prob.set_ylabel("Y (m)", **lbl_kw)
    add_ego_marker(ax_prob)
    plt.colorbar(im4, ax=ax_prob, fraction=0.046).ax.tick_params(colors="white", labelsize=6)
    style_ax(ax_prob, "Prediction (probability)")

    # ── Panel 5: Binary Prediction ────────────────────────────────────────────
    ax_bin.imshow(np.flipud(pred_bin), cmap="gray",
                  extent=BEV_EXTENT, origin="lower", vmin=0, vmax=1)
    ax_bin.set_xlabel("X (m)", **lbl_kw)
    ax_bin.set_ylabel("Y (m)", **lbl_kw)
    add_ego_marker(ax_bin)
    style_ax(ax_bin, f"Prediction (threshold={OCC_THRESHOLD})  IoU={iou:.1f}%")

    # ── Panel 6: Error Map ────────────────────────────────────────────────────
    err_cmap = matplotlib.colors.ListedColormap(
        ["#0a0a18", "#00c49b", "#e63946", "#f4a261"]   # TN, TP, FP, FN
    )
    ax_err.imshow(np.flipud(err), cmap=err_cmap,
                  extent=BEV_EXTENT, origin="lower", vmin=0, vmax=3)
    ax_err.set_xlabel("X (m)", **lbl_kw)
    ax_err.set_ylabel("Y (m)", **lbl_kw)
    add_ego_marker(ax_err)
    legend_patches = [
        mpatches.Patch(color="#0a0a18", label="TN — correct free"),
        mpatches.Patch(color="#00c49b", label="TP — correct occupied"),
        mpatches.Patch(color="#e63946", label="FP — false alarm"),
        mpatches.Patch(color="#f4a261", label="FN — missed obstacle"),
    ]
    ax_err.legend(handles=legend_patches, loc="upper right",
                  fontsize=6, labelcolor="white",
                  facecolor="#111", edgecolor="#444")
    style_ax(ax_err, "Error Map")

    # ── Title & Save ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"MAHE Mobility — BEV Occupancy Pipeline  |  Sample {sample_idx}  |  IoU: {iou:.2f}%",
        color="white", fontsize=13, fontweight="bold", y=0.97
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅  Saved → '{save_path}'")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MAHE Mobility BEV Visualizer")
    parser.add_argument("--sample",  type=int, default=4,               help="Sample index (0–403)")
    parser.add_argument("--out",     type=str, default="bev_sample.png", help="Output image path")
    parser.add_argument("--weights", type=str, default=WEIGHTS,          help="Model weights file")
    args = parser.parse_args()
    run_visualization(sample_idx=args.sample, save_path=args.out, weights=args.weights)
