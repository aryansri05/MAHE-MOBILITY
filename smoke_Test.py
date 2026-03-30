import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Handle the import based on your specific VS Code structure
try:
    from config import GRID_H, GRID_W, OCC_THRESHOLD
except ImportError:
    # Fallback if the parent-path issue persists during testing
    print("⚠️ Warning: Could not find config.py, using default 250x250 grid.")
    GRID_H, GRID_W = 250, 250
    OCC_THRESHOLD = 0.5

# Import your Person A modules
from models.bev_encoder import BEVEncoder
from models.occupancy import OccupancyHead, OccupancyCriterion

def run_smoke_test():
    """
    Verifies that Person A's Encoder and Head can process 
    data and calculate a loss without crashing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Testing Pipeline on: {device}")

    # 1. Initialize Subunit 2 & 3 Modules (Person A & B parts)
    # We assume Subunit 1 sends 64-channel features to our grid
    encoder = BEVEncoder(in_channels=64, base_channels=64).to(device)
    head = OccupancyHead(in_channels=128, hidden_ch=64).to(device)
    criterion = OccupancyCriterion().to(device)

    # 2. Mock Input Data (The "Fake" Subunit 1 features)
    # Shape: (Batch=2, Channels=64, H=250, W=250)
    dummy_bev_features = torch.randn(2, 64, GRID_H, GRID_W).to(device)

    # 3. Mock Ground Truth (The "Fake" Person B labels)
    # Binary mask: 1.0 for occupied, 0.0 for free
    dummy_gt = (torch.rand(2, 1, GRID_H, GRID_W) > 0.98).float().to(device)

    print("—> Running BEV Encoder...")
    spatial_features = encoder(dummy_bev_features) # Output: (2, 128, 250, 250)

    print("—> Running Occupancy Head...")
    logits = head(spatial_features) # Output: (2, 1, 250, 250)
    
    # Get probabilities for visualization
    probs = torch.sigmoid(logits)

    print("—> Calculating Training Loss...")
    loss_dict = criterion(logits, dummy_gt)

    print("\n✅ --- SMOKE TEST SUCCESSFUL --- ✅")
    print(f"Input Shape  : {dummy_bev_features.shape}")
    print(f"Output Shape : {logits.shape}")
    print(f"Total Loss   : {loss_dict['loss'].item():.4f}")
    
    return logits, probs

if __name__ == "__main__":
    # Run the test and capture outputs for visualization
    logits, probs = run_smoke_test()

    # 4. Visualization (Testing the 'decodability' of your grid)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(probs[0, 0].detach().cpu().numpy(), cmap='magma')
    plt.title("AI Predicted Probability Map")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    # Show a binary mask based on the threshold
    mask = (probs[0, 0] > OCC_THRESHOLD).detach().cpu().numpy()
    plt.imshow(mask, cmap='gray')
    plt.title("Thresholded Occupancy Mask")

    plt.tight_layout()
    print("\n📊 Displaying Map... (Close the window to end the script)")
    plt.show()