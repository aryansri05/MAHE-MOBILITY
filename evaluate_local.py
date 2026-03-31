import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import DataLoader

# Add folders to path so we can find the modules
sys.path.append('src')
sys.path.append('scripts')

from mahe_mobility.geometry.lss_core import CameraConfig, BEVGridConfig, DepthConfig
from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION
from mahe_mobility.dataset import NuScenesFrontCameraDataset
from mahe_mobility.tasks.task3_evaluation_iou import visualise_error_map
from pipeline import BEVModel 

def run_final_evaluation():
    # Automatically use Apple Silicon GPU if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🧠 Loading trained brain onto {device}...")

    # 1. Rebuild the empty architecture
    cam_cfg = CameraConfig(image_h=224, image_w=480)
    bev_cfg = BEVGridConfig(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, cell_size=RESOLUTION)
    depth_cfg = DepthConfig()
    model = BEVModel(out_channels=64, cam_cfg=cam_cfg, bev_cfg=bev_cfg, depth_cfg=depth_cfg)

    # 2. Insert the trained weights from the local file
    checkpoint = torch.load("bev_checkpoint.pth", map_location=device, weights_only=True)
    
    # Handle the dictionary structure from our training script's checkpointing
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # fallback
        
    model.to(device)
    model.eval() # Set to inference mode
    print("✅ Weights loaded successfully!")

    # 3. Load the Dataset (Just 1 batch for the showcase)
    # TODO: Update dataroot to my local path if necessary
    dataset = NuScenesFrontCameraDataset(dataroot="./data/nuscenes", version="v1.0-mini")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) 

    print("📸 Taking a test image and generating BEV map...")
    with torch.no_grad(): 
        images, intrinsics, trans, rot, gt_occupancy = next(iter(dataloader))
        images = images.to(device)
        
        # 4. Make the Prediction
        bev_logits = model(images, intrinsics.to(device), trans.to(device), rot.to(device))
        
        pred_probs = torch.sigmoid(bev_logits[0, 0]).cpu().numpy()
        gt = gt_occupancy[0, 0].cpu().numpy()

        # 5. Calculate Accuracy (IoU)
        preds_binary = (pred_probs > 0.5).astype(int)
        gt_binary = gt.astype(int)
        
        intersection = np.logical_and(preds_binary, gt_binary).sum()
        union = np.logical_or(preds_binary, gt_binary).sum()
        iou_score = (intersection / union) * 100 if union > 0 else 0.0

        print(f"\n📊 FINAL METRICS:")
        print(f"-> Accuracy (IoU Score): {iou_score:.2f}%")
        
        # 6. Generate the Final Image
        visualise_error_map(pred_probs, gt, save_path="hackathon_final_plot.png")
        print("\n✅ Final image saved as 'hackathon_final_plot.png'")

if __name__ == "__main__":
    run_final_evaluation()
