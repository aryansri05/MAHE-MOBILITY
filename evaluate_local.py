"""
evaluate_local.py
-----------------
Runs a local evaluation of the trained BEV model using the best saved weights.
Outputs the IoU score and saves a 4-panel diagnostic plot.
"""

import sys
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

WEIGHTS = "bev_model_best_v2.pth"  # Best ResNet34 model from Kaggle training


def run_final_evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🧠 Loading trained model onto {device} from '{WEIGHTS}'...")

    cam_cfg   = CameraConfig(image_h=224, image_w=480)
    bev_cfg   = BEVGridConfig(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION)
    depth_cfg = DepthConfig()
    model     = BEVModel(out_channels=64, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg)

    # bev_model_best_v2.pth is saved as a plain state_dict (not a checkpoint dict)
    checkpoint = torch.load(WEIGHTS, map_location=device, weights_only=False)
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

        bev_logits = model(images, intrinsics.to(device), trans.to(device), rot.to(device))

        pred_probs  = torch.sigmoid(bev_logits[0, 0]).cpu().numpy()
        gt          = gt_occupancy[0, 0].cpu().numpy()

        preds_binary = (pred_probs > 0.5).astype(int)
        gt_binary    = gt.astype(int)

        intersection = np.logical_and(preds_binary, gt_binary).sum()
        union        = np.logical_or(preds_binary, gt_binary).sum()
        iou_score    = (intersection / union) * 100 if union > 0 else 0.0

        print(f"\n📊 FINAL METRICS:")
        print(f"   IoU Score: {iou_score:.2f}%")

        visualise_error_map(pred_probs, gt, save_path="hackathon_final_plot.png")
        print("\n✅ Diagnostic plot saved as 'hackathon_final_plot.png'")


if __name__ == "__main__":
    run_final_evaluation()
