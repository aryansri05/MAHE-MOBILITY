"""
BEV Occupancy Model — Full Pipeline
=====================================
Assembles all subunits into one `nn.Module` with a clean
forward pass and training / inference helpers.

Data flow
---------
  camera image(s)
       │
       ▼  [Subunit 1 — Lift backbone — separate module, not repeated here]
  (B, C, D, img_H, img_W)   depth features
       │
       ▼  [LSS Geometry Architect — voxel pooling]
  (B, C, GRID_H, GRID_W)    raw BEV features
       │
       ▼  [Subunit 2 — BEVEncoder]
  (B, 128, GRID_H, GRID_W)  rich spatial features
       │
       ▼  [Subunit 3 — OccupancyHead]
  (B, 1,   GRID_H, GRID_W)  occupancy logits / probabilities

Training
---------
  Run train.py (example script at bottom) to iterate over nuScenes,
  log Focal + BCE losses, and validate with occ-IoU + DWE.
"""

import torch
import torch.nn as nn

from mahe_mobility.config import GRID_H, GRID_W, OCC_THRESHOLD
from mahe_mobility.models.bev_encoder import BEVEncoder
from mahe_mobility.models.occupancy import (
    OccupancyHead,
    OccupancyCriterion,
    occupancy_iou,
    distance_weighted_error,
)


class BEVOccupancyModel(nn.Module):
    """
    End-to-end BEV occupancy model (Subunits 2 + 3).

    The LSS frustum→voxelpool stage (Geometry Architect) is kept
    separate so it can be swapped for a BEVFormer / SimpleBEV
    view transformer without touching this module.

    Parameters
    ----------
    lift_channels : int   — output channels from the voxel pooler / Lift
    base_channels : int   — BEVEncoder width multiplier
    enc_out_ch    : int   — channels into the occupancy head
    head_hidden   : int   — hidden width inside the head
    """

    def __init__(
        self,
        lift_channels: int = 64,
        base_channels: int = 64,
        enc_out_ch: int = 128,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.encoder = BEVEncoder(
            in_channels=lift_channels,
            base_channels=base_channels,
            out_channels=enc_out_ch,
        )
        self.head = OccupancyHead(
            in_channels=enc_out_ch,
            hidden_ch=head_hidden,
        )

    # ------------------------------------------------------------------
    def forward(self, bev_raw: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        bev_raw : (B, lift_channels, GRID_H, GRID_W)
            Raw BEV features from the view transformer (LSS / BEVFormer).

        Returns
        -------
        logits : (B, 1, GRID_H, GRID_W)   — un-sigmoidied
        """
        feats = self.encoder(bev_raw)
        logits = self.head(feats)
        return logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        bev_raw: torch.Tensor,
        threshold: float = OCC_THRESHOLD,
    ) -> dict:
        """
        Inference wrapper. Returns a dict with probs and binary mask.
        """
        self.eval()
        logits = self.forward(bev_raw)
        probs = torch.sigmoid(logits)
        mask = probs > threshold
        return {"logits": logits, "probs": probs, "mask": mask}

    # ------------------------------------------------------------------
    def training_step(
        self,
        bev_raw: torch.Tensor,
        gt_mask: torch.Tensor,
        criterion: OccupancyCriterion,
    ) -> dict:
        """
        Single training step. Call `.backward()` on the returned loss.

        Parameters
        ----------
        bev_raw  : (B, C, GRID_H, GRID_W)
        gt_mask  : (B, 1, GRID_H, GRID_W) binary float {0, 1}
        criterion: OccupancyCriterion

        Returns
        -------
        dict with 'loss', 'focal_loss', 'bce_loss' tensors
        """
        logits = self.forward(bev_raw)
        return criterion(logits, gt_mask)

    @torch.no_grad()
    def validation_step(
        self,
        bev_raw: torch.Tensor,
        gt_mask: torch.Tensor,
        threshold: float = OCC_THRESHOLD,
    ) -> dict:
        """
        Compute eval metrics for one batch.

        Returns
        -------
        dict with 'iou' and 'dwe' scalar tensors
        """
        self.eval()
        result = self.predict(bev_raw, threshold)
        iou = occupancy_iou(result["mask"], gt_mask.bool())
        dwe = distance_weighted_error(result["probs"], gt_mask)
        return {"iou": iou, "dwe": dwe}


# ─────────────────────────────────────────────
#  Example training loop skeleton
# ─────────────────────────────────────────────


def example_training_loop():
    """
    Minimal training loop to validate the full pipeline.
    Replace the random tensors with your nuScenes DataLoader.
    """
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BEVOccupancyModel().to(device)
    criterion = OccupancyCriterion(
        focal_alpha=0.25,
        focal_gamma=2.0,
        pos_weight=20.0,
        lambda_focal=1.0,
        lambda_bce=0.5,
    )
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    B, C = 2, 64
    print(f"\nRunning on: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    for step in range(3):  # replace with real dataloader
        # ── Simulate a batch (replace with real data) ─────────────
        bev_raw = torch.randn(B, C, GRID_H, GRID_W, device=device)
        # ~5% occupancy, matching nuScenes statistics
        gt_mask = (torch.rand(B, 1, GRID_H, GRID_W, device=device) > 0.95).float()

        # ── Forward + loss ────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        losses = model.training_step(bev_raw, gt_mask, criterion)
        losses["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
        optimizer.step()

        # ── Val metrics ───────────────────────────────────────────
        metrics = model.validation_step(bev_raw, gt_mask)

        print(
            f"  step {step + 1} | "
            f"loss={losses['loss'].item():.4f}  "
            f"focal={losses['focal_loss'].item():.4f}  "
            f"bce={losses['bce_loss'].item():.4f} | "
            f"IoU={metrics['iou'].item():.4f}  "
            f"DWE={metrics['dwe'].item():.5f}"
        )

    scheduler.step()
    print("\n  ✓ Training loop smoke-test complete.")


if __name__ == "__main__":
    example_training_loop()
