"""
evaluate_local.py
-----------------
Runs a local evaluation of the trained BEV model using the best saved weights.
Outputs the IoU score and saves a 4-panel diagnostic plot.
"""

import sys
from scipy.ndimage import binary_erosion
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('src')
sys.path.append('scripts')

from mahe_mobility.geometry.lss_core import CameraConfig, BEVGridConfig, DepthConfig
from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION
from mahe_mobility.dataset import NuScenesFrontCameraDataset
from mahe_mobility.tasks.task3_evaluation_iou import visualise_error_map
from pipeline import BEVModel

WEIGHTS = "bev_model_final_v2 (1).pth"


def run_final_evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    import os
    weights_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS)
    print(f"🧠 Loading trained model onto {device} from '{weights_abs}'...")

    cam_cfg   = CameraConfig(image_h=224, image_w=480)
    bev_cfg   = BEVGridConfig(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION)
    depth_cfg = DepthConfig()
    model     = BEVModel(out_channels=64, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg)

    # bev_model_best_v2.pth is saved as a plain state_dict (not a checkpoint dict)
    checkpoint = torch.load(weights_abs, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print("✅ Weights loaded successfully!")

    dataset    = NuScenesFrontCameraDataset(dataroot="./data/nuscenes", version="v1.0-mini", train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("📸 Running inference on a random test sample...")
    with torch.no_grad():
        images, intrinsics, trans, rot, gt_occupancy = next(iter(dataloader))
        images = images.to(device)

        # AMP: float16 on CUDA for speed boost, no-op on CPU/MPS
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) \
            if device.type == 'cuda' else torch.amp.autocast(device_type='cpu', enabled=False)

        with autocast_ctx:
            bev_logits = model(images, intrinsics.to(device), trans.to(device), rot.to(device))
        bev_logits = bev_logits.float()  # cast back to float32 for metric computation

        pred_probs  = torch.sigmoid(bev_logits[0, 0]).cpu().numpy()
        gt          = gt_occupancy[0, 0].cpu().numpy()

        preds_binary = (pred_probs > 0.5).astype(np.uint8)
        # Morphological erosion: strips thin false-positive edges, keeps dense confident regions
        kernel = np.ones((3, 3), bool)
        preds_binary = binary_erosion(preds_binary, structure=kernel).astype(np.uint8)
        gt_binary    = gt.astype(int)

        intersection = np.logical_and(preds_binary, gt_binary).sum()
        union        = np.logical_or(preds_binary, gt_binary).sum()
        iou_score    = (intersection / union) * 100 if union > 0 else 0.0

        # ── Precision / Recall / F1 ──────────────────────────────
        TP = (preds_binary * gt_binary).sum()
        FP = (preds_binary * (1 - gt_binary)).sum()
        FN = ((1 - preds_binary) * gt_binary).sum()
        precision = TP / (TP + FP + 1e-6)
        recall    = TP / (TP + FN + 1e-6)
        f1_score  = 2 * (precision * recall) / (precision + recall + 1e-6)

        # ── Distance-Weighted Error (20× penalty in 0–10 m zone) ─
        errors = np.abs(preds_binary.astype(float) - gt_binary.astype(float))
        weights = np.ones_like(errors, dtype=float)
        ten_meter_mark = int(errors.shape[0] * 0.8)  # bottom 20% ≈ 0–10 m
        weights[ten_meter_mark:, :] = 20.0
        dwe = (errors * weights).sum() / weights.sum()

        print(f"\n📊 FINAL METRICS:")
        print(f"   IoU Score  : {iou_score:.2f}%")
        print(f"   Precision  : {precision:.4f}")
        print(f"   Recall     : {recall:.4f}")
        print(f"   F1 Score   : {f1_score:.4f}")
        print(f"   DWE        : {dwe:.4f}  (lower is better)")

        visualise_error_map(pred_probs, gt, save_path="hackathon_final_plot.png")
        print("\n✅ Diagnostic plot saved as 'hackathon_final_plot.png'")


if __name__ == "__main__":
    run_final_evaluation()