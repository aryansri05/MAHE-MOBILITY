"""
Subunit 3 — Occupancy Head
Subunit 4 — Loss Functions and Evaluation Metrics
===================================================

Subunit 3
---------
A lightweight convolutional head that maps BEV encoder features
to a per-cell occupancy probability map.

  Input  : (B, C, GRID_H, GRID_W)   — from BEVEncoder
  Output : (B, 1, GRID_H, GRID_W)   — sigmoid probability, range [0, 1]

Subunit 4
---------
Loss
  • Focal loss          — handles the severe class imbalance (most BEV
                           cells are empty; occupied cells are rare).
  • Binary cross-entropy — baseline comparator, weighted by pos_weight.

Metrics (eval only, not differentiable)
  • Occupancy IoU       — intersection-over-union on the binary grid
  • Distance-weighted error — penalises mistakes close to the ego vehicle
                              more than distant ones (per challenge spec).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GRID_H, GRID_W, X_MIN, Y_MIN, RESOLUTION, OCC_THRESHOLD


# ═══════════════════════════════════════════════════════════════════
#  SUBUNIT 3 — OccupancyHead
# ═══════════════════════════════════════════════════════════════════

class OccupancyHead(nn.Module):
    """
    Binary occupancy prediction head.

    Architecture
    ------------
    Three 3×3 convolutions narrow the channel count, then a 1×1 conv
    collapses to a single logit per cell. No sigmoid here — we return
    raw logits so BCEWithLogitsLoss / Focal loss can be numerically
    stable. Call `.predict()` to get sigmoid probabilities at inference.

    Parameters
    ----------
    in_channels : int   — output channels of BEVEncoder (e.g. 128)
    hidden_ch   : int   — intermediate channel width (default 64)
    """

    def __init__(self, in_channels: int = 128, hidden_ch: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            # 3×3 conv, preserve resolution
            nn.Conv2d(in_channels, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_ch, hidden_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch // 2),
            nn.ReLU(inplace=True),

            # 1×1 projection → single logit per cell
            nn.Conv2d(hidden_ch // 2, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        # Initialise final bias so sigmoid(bias) ≈ 0.01
        # → model starts predicting "empty" for all cells,
        #    which matches the prior (most cells are empty).
        nn.init.constant_(self.head[-1].bias, -4.6)  # sigmoid(-4.6) ≈ 0.01

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feats : (B, in_channels, GRID_H, GRID_W)

        Returns
        -------
        logits : (B, 1, GRID_H, GRID_W)   raw (un-sigmoidied) logits
        """
        return self.head(feats)

    @torch.no_grad()
    def predict(
        self,
        feats:     torch.Tensor,
        threshold: float = OCC_THRESHOLD,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper for inference.

        Returns
        -------
        probs : (B, 1, GRID_H, GRID_W)   sigmoid probabilities
        mask  : (B, 1, GRID_H, GRID_W)   bool  — True = occupied
        """
        logits = self.forward(feats)
        probs  = torch.sigmoid(logits)
        mask   = probs > threshold
        return probs, mask


# ═══════════════════════════════════════════════════════════════════
#  SUBUNIT 4 — Loss Functions
# ═══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Sigmoid focal loss (Lin et al. 2017).

    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

    Why focal over plain BCE?
    -------------------------
    In a 250×250 BEV grid ≈ 62,500 cells. A typical nuScenes scene
    might have 2–5% occupied cells. Vanilla BCE lets the model "win"
    by predicting all-empty. Focal loss down-weights the easy
    negatives by (1 − p_t)^γ, forcing the model to focus on the
    hard positives (small objects, far obstacles).

    Parameters
    ----------
    alpha : float   — weight for positive class (occupied). 0.25 is
                       the default from the original focal loss paper.
    gamma : float   — focusing exponent. 2.0 is standard.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,   # (B, 1, H, W) raw logits
        targets: torch.Tensor,  # (B, 1, H, W) binary {0, 1} float
    ) -> torch.Tensor:
        """Returns scalar mean focal loss."""
        # Binary cross-entropy term (numerically stable via logsigmoid)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # (B, 1, H, W)

        # p_t: probability of the TRUE class
        p       = torch.sigmoid(logits)
        p_t     = targets * p + (1 - targets) * (1 - p)

        # α_t weighting
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal modulation
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss         = focal_weight * bce

        return loss.mean()


class WeightedBCELoss(nn.Module):
    """
    Standard BCE with logits + positive class weight.

    Simpler than focal; useful as a baseline.

    Parameters
    ----------
    pos_weight : float  — scales the positive-class BCE term.
                           A value of (neg_count / pos_count) is a
                           good starting point (~20 for nuScenes BEV).
    """

    def __init__(self, pos_weight: float = 20.0):
        super().__init__()
        self.register_buffer(
            "pos_weight", torch.tensor([pos_weight])
        )

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight.to(logits.device),
        )


# ═══════════════════════════════════════════════════════════════════
#  SUBUNIT 4 — Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def occupancy_iou(
    pred_mask:  torch.Tensor,   # (B, 1, H, W) bool
    gt_mask:    torch.Tensor,   # (B, 1, H, W) bool
    eps:        float = 1e-6,
) -> torch.Tensor:
    """
    Per-sample occupancy IoU averaged over the batch.

    IoU = |pred ∩ gt| / |pred ∪ gt|

    Returns
    -------
    iou : scalar tensor
    """
    pred = pred_mask.bool().float()
    gt   = gt_mask.bool().float()

    # Sum over spatial dims (H, W, channel)
    intersection = (pred * gt).sum(dim=(1, 2, 3))   # (B,)
    union        = (pred + gt).clamp(max=1).sum(dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def _build_distance_weight_map(device: torch.device) -> torch.Tensor:
    """
    Pre-compute a (1, 1, GRID_H, GRID_W) weight map where each cell's
    weight is inversely proportional to its distance from the ego vehicle.

    Grid layout (from config.py)
    ----------------------------
    • X axis (columns): X_MIN=-25 m to X_MAX=+25 m, ego at col 125
    • Y axis (rows)   : Y_MIN=0 m  to Y_MAX=50 m,  ego at row 0 (top)

    Weight formula:
        w(r, c) = 1 / (d_ego(r, c) + 1)

    → cells at d=0 m get weight 1.0
    → cells at d=50 m get weight ≈ 0.02
    → cells closer to the car matter ~50× more than those at 50 m
    """
    # Physical coordinates of each cell centre
    rows = torch.arange(GRID_H, dtype=torch.float32, device=device)
    cols = torch.arange(GRID_W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")  # (H, W)

    # Convert to metres in ego frame
    x_m = grid_x * RESOLUTION + X_MIN + RESOLUTION / 2
    y_m = grid_y * RESOLUTION + Y_MIN + RESOLUTION / 2

    # Euclidean distance from ego (ego is at X=0, Y=0 in ego frame)
    dist = torch.sqrt(x_m ** 2 + y_m ** 2)          # (H, W)
    weights = 1.0 / (dist + 1.0)                     # avoid div/0

    return weights.unsqueeze(0).unsqueeze(0)          # (1, 1, H, W)


@torch.no_grad()
def distance_weighted_error(
    pred_probs:  torch.Tensor,   # (B, 1, H, W) sigmoid probabilities
    gt_mask:     torch.Tensor,   # (B, 1, H, W) binary float/bool
) -> torch.Tensor:
    """
    Challenge metric: distance-weighted mean absolute error.

    DWE = Σ_{cells} w(cell) · |pred(cell) − gt(cell)|
          ──────────────────────────────────────────────
                    Σ_{cells} w(cell)

    Errors near the ego vehicle are penalised more heavily.
    Returns a scalar (mean over batch).
    """
    weights = _build_distance_weight_map(pred_probs.device)  # (1,1,H,W)
    gt_f    = gt_mask.float()

    abs_err     = (pred_probs - gt_f).abs()             # (B, 1, H, W)
    weighted    = (abs_err * weights).sum(dim=(1,2,3))   # (B,)
    normaliser  = weights.sum()

    return (weighted / normaliser).mean()


# ═══════════════════════════════════════════════════════════════════
#  Combined criterion (used in training loop)
# ═══════════════════════════════════════════════════════════════════

class OccupancyCriterion(nn.Module):
    """
    Joint training loss:  λ_focal · FocalLoss + λ_bce · WeightedBCE

    In practice, focal loss alone is often sufficient;
    the weighted BCE acts as a regulariser ensuring calibration.

    Parameters
    ----------
    focal_alpha : float
    focal_gamma : float
    pos_weight  : float   — for the BCE branch
    lambda_focal: float   — loss mixing coefficient
    lambda_bce  : float
    """

    def __init__(
        self,
        focal_alpha:  float = 0.25,
        focal_gamma:  float = 2.0,
        pos_weight:   float = 20.0,
        lambda_focal: float = 1.0,
        lambda_bce:   float = 0.5,
    ):
        super().__init__()
        self.focal      = FocalLoss(focal_alpha, focal_gamma)
        self.bce        = WeightedBCELoss(pos_weight)
        self.lf         = lambda_focal
        self.lb         = lambda_bce

    def forward(
        self,
        logits:  torch.Tensor,   # (B, 1, H, W)
        targets: torch.Tensor,   # (B, 1, H, W) binary float
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with individual loss components and the total,
        so the training loop can log each term separately.
        """
        l_focal = self.focal(logits, targets)
        l_bce   = self.bce(logits, targets)
        total   = self.lf * l_focal + self.lb * l_bce

        return {
            "loss":       total,
            "focal_loss": l_focal,
            "bce_loss":   l_bce,
        }


# ═══════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Occupancy Head + Loss/Metrics — smoke test")
    B = 2

    # ── Head ──────────────────────────────────────────────────────
    head      = OccupancyHead(in_channels=128, hidden_ch=64)
    feats     = torch.randn(B, 128, GRID_H, GRID_W)
    logits    = head(feats)
    probs, mask = head.predict(feats)

    print(f"  logits : {tuple(logits.shape)}")
    print(f"  probs  : {tuple(probs.shape)}  "
          f"[{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  mask   : {mask.sum().item()} occupied cells "
          f"(of {GRID_H*GRID_W})")

    # ── Loss ──────────────────────────────────────────────────────
    targets   = (torch.rand(B, 1, GRID_H, GRID_W) > 0.95).float()
    criterion = OccupancyCriterion()
    losses    = criterion(logits, targets)
    print(f"\n  total loss : {losses['loss'].item():.4f}")
    print(f"  focal loss : {losses['focal_loss'].item():.4f}")
    print(f"  bce loss   : {losses['bce_loss'].item():.4f}")

    # ── Metrics ───────────────────────────────────────────────────
    iou = occupancy_iou(mask, targets.bool())
    dwe = distance_weighted_error(probs, targets)
    print(f"\n  occ-IoU             : {iou.item():.4f}")
    print(f"  dist-weighted error : {dwe.item():.4f}")
    print("\n  ✓ all checks passed")