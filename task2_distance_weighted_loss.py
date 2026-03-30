# task2_distance_weighted_loss.py
# ─────────────────────────────────────────────────────────────────
# Person B  |  Task 2: Distance-Weighted Loss Function
#
# Why we need this:
#   Standard Binary Cross-Entropy (BCE) treats every grid cell
#   equally. But the problem statement says: errors CLOSER to the
#   ego-vehicle must be penalised MORE. A wrong prediction at 2 m
#   is far more dangerous than one at 45 m.
#
# How it works:
#   1. Build a weight mask  W[row, col]  where cells near (0,0)
#      get a HIGH weight and far cells get a LOW weight.
#   2. Multiply the per-cell BCE loss by W before averaging.
#
# Weight formula:
#   distance  = sqrt( X² + Y² )   for each cell centre
#   weight    = 1 / (distance + 1)
#              → distance = 0  → weight = 1.0  (maximum)
#              → distance = 49 → weight ≈ 0.02 (minimum)
#
# Output:  DistanceWeightedBCELoss  — drop-in PyTorch loss module
# ─────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (X_MIN, X_MAX, Y_MIN, Y_MAX,
                    RESOLUTION, GRID_W, GRID_H)


# ═════════════════════════════════════════════════════════════════
# STEP 1 — Build the distance weight mask (numpy, CPU)
# ═════════════════════════════════════════════════════════════════

def build_distance_weight_mask() -> np.ndarray:
    """
    Create a (GRID_H, GRID_W) array where each cell contains a
    weight inversely proportional to its real-world distance from
    the ego vehicle (0, 0).

    Coordinate system:
        col_idx → X axis  (left/right)
        row_idx → Y axis  (forward from car)

    Returns:
        weight_mask : np.ndarray  shape (GRID_H, GRID_W)  float32
    """
    # ── Cell centre coordinates in metres ────────────────────────
    # Centre of cell (col, row):
    #   x_centre = X_MIN + (col + 0.5) * RESOLUTION
    #   y_centre = Y_MIN + (row + 0.5) * RESOLUTION

    col_centres = X_MIN + (np.arange(GRID_W) + 0.5) * RESOLUTION   # (GRID_W,)
    row_centres = Y_MIN + (np.arange(GRID_H) + 0.5) * RESOLUTION   # (GRID_H,)

    # Broadcast into a full 2D grid
    # X[row, col]  and  Y[row, col]
    X, Y = np.meshgrid(col_centres, row_centres)   # both (GRID_H, GRID_W)

    # ── Euclidean distance from ego (0, 0) ───────────────────────
    distance = np.sqrt(X**2 + Y**2)               # (GRID_H, GRID_W)

    # ── Inverse-distance weight ──────────────────────────────────
    #   +1 avoids division by zero at the origin cell
    weight_mask = 1.0 / (distance + 1.0)          # (GRID_H, GRID_W)

    # ── Optional: normalise so the mean weight = 1.0 ─────────────
    #   This keeps the total loss magnitude comparable to plain BCE
    weight_mask = weight_mask / weight_mask.mean()

    print(f"Weight mask  shape={weight_mask.shape}")
    print(f"  max weight (near car) : {weight_mask.max():.3f}")
    print(f"  min weight (far away) : {weight_mask.min():.3f}")
    print(f"  mean weight           : {weight_mask.mean():.3f}")
    return weight_mask.astype(np.float32)


# ═════════════════════════════════════════════════════════════════
# STEP 2 — PyTorch loss module
# ═════════════════════════════════════════════════════════════════

class DistanceWeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss with distance-based cell weighting.

    Cells close to the ego vehicle contribute more to the total
    loss than distant cells, matching the hackathon metric.

    Usage:
        criterion = DistanceWeightedBCELoss()

        # pred : model output  shape (B, 1, H, W) or (B, H, W)
        #        RAW logits (before sigmoid)
        # gt   : ground truth  shape (B, 1, H, W) or (B, H, W)
        #        binary float32 (0.0 or 1.0)

        loss = criterion(pred, gt)
        loss.backward()
    """

    def __init__(self):
        super().__init__()

        # Build the weight mask once and register as a buffer
        # (buffers are moved to GPU automatically with .to(device))
        mask_np  = build_distance_weight_mask()           # (H, W)
        mask_t   = torch.from_numpy(mask_np)              # (H, W)
        self.register_buffer("weight_mask", mask_t)       # persists in state_dict

    def forward(self,
                pred_logits: torch.Tensor,
                gt:          torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits : raw model output  (B, 1, H, W) or (B, H, W)
            gt          : binary ground truth (B, 1, H, W) or (B, H, W)

        Returns:
            scalar loss tensor
        """
        # ── Normalise shapes ─────────────────────────────────────
        if pred_logits.dim() == 4:
            pred_logits = pred_logits.squeeze(1)   # (B, H, W)
        if gt.dim() == 4:
            gt = gt.squeeze(1)                     # (B, H, W)

        # ── Per-cell BCE (unreduced) ──────────────────────────────
        #   shape: (B, H, W)
        bce_per_cell = F.binary_cross_entropy_with_logits(
            pred_logits, gt, reduction="none"
        )

        # ── Apply distance weight mask ────────────────────────────
        #   weight_mask is (H, W) — broadcasts across batch dim
        weighted = bce_per_cell * self.weight_mask   # (B, H, W)

        # ── Mean over all cells and batch ────────────────────────
        return weighted.mean()


# ═════════════════════════════════════════════════════════════════
# STEP 3 — Visualise the weight mask (sanity check)
# ═════════════════════════════════════════════════════════════════

def visualise_weight_mask(mask: np.ndarray, save_path: str = None):
    """
    Plot the weight mask so you can confirm:
      - Bright (high weight) = near the ego vehicle
      - Dark  (low weight)  = far from the ego vehicle
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        mask,
        cmap   = "hot",
        origin = "lower",
        extent = [X_MIN, X_MAX, Y_MIN, Y_MAX],
    )
    plt.colorbar(im, ax=ax, label="Cell weight")

    ax.plot(0, 0, marker="^", color="cyan", markersize=10,
            label="Ego (weight=max)")
    ax.set_xlabel("X (m) → right")
    ax.set_ylabel("Y (m) → forward")
    ax.set_title("Distance-Weighted Loss Mask\n"
                 "bright = errors penalised more")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved → {save_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════
# STEP 4 — Quick integration test (no GPU needed)
# ═════════════════════════════════════════════════════════════════

def demo_loss():
    """
    Prove the loss works by comparing two fake predictions:
      - pred_good : mostly correct
      - pred_bad  : makes errors right in front of the car

    Expect:  loss_bad  >>  loss_good
    """
    print("\n── Demo: distance-weighted vs plain BCE ──")

    B = 2   # batch size
    H, W = GRID_H, GRID_W

    # Random ground truth
    gt = torch.randint(0, 2, (B, H, W)).float()

    # Good prediction: small random logit errors everywhere
    pred_good = gt * 4 - 2 + torch.randn(B, H, W) * 0.5

    # Bad prediction: deliberately wrong logits in the NEAR zone
    pred_bad  = pred_good.clone()
    near_rows = H // 10        # bottom 10% of grid = closest to car
    pred_bad[:, :near_rows, :] = -pred_bad[:, :near_rows, :]

    criterion       = DistanceWeightedBCELoss()
    plain_bce       = nn.BCEWithLogitsLoss()

    loss_good_w     = criterion(pred_good, gt)
    loss_bad_w      = criterion(pred_bad,  gt)
    loss_good_plain = plain_bce(pred_good, gt)
    loss_bad_plain  = plain_bce(pred_bad,  gt)

    print(f"  Distance-weighted loss  — good pred : {loss_good_w.item():.4f}")
    print(f"  Distance-weighted loss  — bad pred  : {loss_bad_w.item():.4f}  ← should be MUCH higher")
    print(f"  Plain BCE loss          — good pred : {loss_good_plain.item():.4f}")
    print(f"  Plain BCE loss          — bad pred  : {loss_bad_plain.item():.4f}  ← gap is smaller")

    ratio_weighted = loss_bad_w.item()    / loss_good_w.item()
    ratio_plain    = loss_bad_plain.item()/ loss_good_plain.item()
    print(f"\n  bad/good ratio — weighted: {ratio_weighted:.2f}x   plain: {ratio_plain:.2f}x")
    print("  ✓ Weighted ratio should be higher — near-car errors are penalised more.")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Build and visualise the weight mask
    mask = build_distance_weight_mask()
    visualise_weight_mask(mask, save_path="weight_mask.png")

    # Save mask as .npy (useful for Person A's training loop)
    np.save("distance_weight_mask.npy", mask)
    print("  Mask saved → distance_weight_mask.npy")

    # Run integration demo
    demo_loss()

    print("\n── How to use in your training loop ──")
    print("""
    from task2_distance_weighted_loss import DistanceWeightedBCELoss

    criterion = DistanceWeightedBCELoss().to(device)

    for batch in dataloader:
        pred   = model(images)          # raw logits (B, 1, H, W)
        gt     = batch['occupancy']     # binary (B, 1, H, W)
        loss   = criterion(pred, gt)
        loss.backward()
        optimizer.step()
    """)