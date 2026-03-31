"""
occupancy.py — IoU Surgery Edition
====================================
Loss arsenal:
  1. LovászSoftmax   — directly optimises the Jaccard index (IoU)
  2. FocalLoss       — hard-example mining for severe class imbalance
  3. SobelBoundaryLoss — Sobel-filter edge penalty to sharpen object boundaries
  4. WeightedBCELoss — class-balanced BCE baseline

Metrics:
  • occupancy_iou          — binary IoU
  • distance_weighted_error — challenge DWE metric
  • find_optimal_threshold  — sweeps 0.1→0.9 to maximise F1 on a batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mahe_mobility.config import GRID_H, GRID_W, X_MIN, Y_MIN, RESOLUTION, OCC_THRESHOLD


# ═══════════════════════════════════════════════════════════════════
#  SUBUNIT 3 — OccupancyHead
# ═══════════════════════════════════════════════════════════════════

class OccupancyHead(nn.Module):
    """
    Binary occupancy prediction head.

    Architecture
    ------------
    Two 3×3 convolutions narrow the channel count, then a 1×1 conv
    collapses to a single logit per cell. Returns raw logits —
    call sigmoid externally for probabilities.

    Parameters
    ----------
    in_channels : int   — output channels of BEVEncoder (e.g. 128)
    hidden_ch   : int   — intermediate channel width (default 64)
    """

    def __init__(self, in_channels: int = 128, hidden_ch: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch // 2, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.head[-1].bias, -4.6)  # sigmoid(-4.6) ≈ 0.01

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats)

    @torch.no_grad()
    def predict(self, feats: torch.Tensor, threshold: float = OCC_THRESHOLD):
        logits = self.forward(feats)
        probs = torch.sigmoid(logits)
        mask = probs > threshold
        return probs, mask


# ═══════════════════════════════════════════════════════════════════
#  LOSS 1 — Lovász-Softmax (Direct IoU Optimisation)
# ═══════════════════════════════════════════════════════════════════

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient of the Lovász extension for binary losses.
    This is the core of the Lovász-Softmax algorithm (Berman et al. 2018).
    """
    p = len(gt_sorted)
    gt_sorted_f = gt_sorted.float()
    gts = gt_sorted_f.sum()
    intersection = gts - gt_sorted_f.cumsum(0)
    union = gts + (1.0 - gt_sorted_f).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszBinaryLoss(nn.Module):
    """
    Lovász-Softmax Loss for binary segmentation (Berman et al. 2018).

    Directly optimises the Jaccard index (IoU) — the primary evaluation
    metric. Standard BCE/Focal losses optimise pixel-level probability,
    which is only a proxy for IoU. This loss optimises IoU directly.

    The binary version treats each sample independently and sorts errors
    by magnitude before computing the Lovász extension of the Jaccard.

    Parameters
    ----------
    per_image : bool  — if True, average loss over images in the batch.
                        If False, flatten all pixels into one loss.
    """
    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def _lovasz_hinge_flat(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the Lovász hinge for a flattened (1D) set of logits and binary targets."""
        if len(targets) == 0:
            return logits.sum() * 0.0
        signs = 2.0 * targets.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = targets[perm]
        grad = _lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, 1, H, W) — raw logits (no sigmoid applied)
        targets : (B, 1, H, W) — binary float {0, 1}
        """
        if self.per_image:
            loss = torch.stack([
                self._lovasz_hinge_flat(
                    logits[i].view(-1),
                    targets[i].view(-1).bool()
                )
                for i in range(logits.shape[0])
            ]).mean()
        else:
            loss = self._lovasz_hinge_flat(logits.view(-1), targets.view(-1).bool())
        return loss


# ═══════════════════════════════════════════════════════════════════
#  LOSS 2 — Sobel Boundary Loss (Edge Enforcement)
# ═══════════════════════════════════════════════════════════════════

class SobelBoundaryLoss(nn.Module):
    """
    Boundary-aware loss using Sobel edge detection.

    Extracts edges from both the predicted probability map and the
    ground-truth mask using Sobel filters, then penalises their
    difference. This forces the network to draw razor-sharp, precise
    boundaries around occupied regions rather than blurry blobs.

    The Sobel edge magnitude captures high-frequency spatial transitions
    (exactly the occupied/free boundary), providing direct gradient signal
    to sharpen the output.
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels (fixed, non-trainable)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # Shape: (out_ch, in_ch, H, W)
        self.register_buffer("kernel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", sobel_y.view(1, 1, 3, 3))

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel edge magnitude for a (B, 1, H, W) tensor."""
        ex = F.conv2d(x, self.kernel_x, padding=1)
        ey = F.conv2d(x, self.kernel_y, padding=1)
        return torch.sqrt(ex ** 2 + ey ** 2 + 1e-8)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, 1, H, W) — raw logits
        targets : (B, 1, H, W) — binary float {0, 1}
        """
        probs = torch.sigmoid(logits)
        pred_edges = self._edge_map(probs)
        gt_edges = self._edge_map(targets)
        return F.l1_loss(pred_edges, gt_edges)


# ═══════════════════════════════════════════════════════════════════
#  LOSS 3 — Focal Loss (Hard-Example Mining)
# ═══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Sigmoid focal loss (Lin et al. 2017).
    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ═══════════════════════════════════════════════════════════════════
#  LOSS 4 — Weighted BCE (Calibration anchor)
# ═══════════════════════════════════════════════════════════════════

class WeightedBCELoss(nn.Module):
    """Standard BCE with positive class weight for class-imbalance handling."""
    def __init__(self, pos_weight: float = 20.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight.to(logits.device)
        )


# ═══════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def occupancy_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Binary IoU = |pred ∩ gt| / |pred ∪ gt|, averaged over batch."""
    pred = pred_mask.bool().float()
    gt = gt_mask.bool().float()
    intersection = (pred * gt).sum(dim=(1, 2, 3))
    union = (pred + gt).clamp(max=1).sum(dim=(1, 2, 3))
    return ((intersection + eps) / (union + eps)).mean()


@torch.no_grad()
def find_optimal_threshold(
    all_probs: torch.Tensor,  # (N, 1, H, W) — accumulated sigmoid probs
    all_gt: torch.Tensor,     # (N, 1, H, W) — binary ground truth
    thresholds: list = None,
) -> tuple[float, float]:
    """
    Dynamic F1 Thresholding — sweeps from 0.1 to 0.9 to find the
    exact threshold that maximises F1 Score for this dataset.

    Returns
    -------
    best_threshold : float
    best_f1        : float
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in torch.arange(0.1, 0.95, 0.05).tolist()]

    best_f1, best_thresh = 0.0, 0.5
    gt_flat = all_gt.bool().float().view(-1)

    for t in thresholds:
        pred_flat = (all_probs > t).float().view(-1)
        tp = (pred_flat * gt_flat).sum()
        fp = (pred_flat * (1 - gt_flat)).sum()
        fn = ((1 - pred_flat) * gt_flat).sum()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        if f1.item() > best_f1:
            best_f1 = f1.item()
            best_thresh = t

    return best_thresh, best_f1


def _build_distance_weight_map(device: torch.device) -> torch.Tensor:
    rows = torch.arange(GRID_H, dtype=torch.float32, device=device)
    cols = torch.arange(GRID_W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    x_m = grid_x * RESOLUTION + X_MIN + RESOLUTION / 2
    y_m = grid_y * RESOLUTION + Y_MIN + RESOLUTION / 2
    dist = torch.sqrt(x_m**2 + y_m**2)
    weights = 1.0 / (dist + 1.0)
    return weights.unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def distance_weighted_error(
    pred_probs: torch.Tensor,
    gt_mask: torch.Tensor,
) -> torch.Tensor:
    """Challenge metric: distance-weighted MAE."""
    weights = _build_distance_weight_map(pred_probs.device)
    gt_f = gt_mask.float()
    abs_err = (pred_probs - gt_f).abs()
    weighted = (abs_err * weights).sum(dim=(1, 2, 3))
    normaliser = weights.sum()
    return (weighted / normaliser).mean()


# ═══════════════════════════════════════════════════════════════════
#  Combined Criterion — IoU Surgery Edition
# ═══════════════════════════════════════════════════════════════════

class OccupancyCriterion(nn.Module):
    """
    IoU Surgery Loss Stack:

        L = λ_lovász  · LovászBinary    (direct IoU optimisation)
          + λ_focal   · FocalLoss       (hard-example mining)
          + λ_boundary· SobelBoundary   (edge sharpening)
          + λ_bce     · WeightedBCE     (calibration anchor)
          + optional depth supervision terms

    Default coefficients are tuned for low-occupancy BEV grids (~2-5%).
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float = 20.0,
        lambda_lovasz: float = 1.0,    # PRIMARY: direct IoU optimisation
        lambda_focal: float = 0.5,     # SECONDARY: hard-example mining
        lambda_boundary: float = 0.3,  # TERTIARY: edge sharpening
        lambda_bce: float = 0.2,       # ANCHOR: calibration
        lambda_depth: float = 1.0,
        lambda_tv: float = 0.05,
    ):
        super().__init__()
        self.lovasz = LovaszBinaryLoss(per_image=True)
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.boundary = SobelBoundaryLoss()
        self.bce = WeightedBCELoss(pos_weight)
        self.ll = lambda_lovasz
        self.lf = lambda_focal
        self.lb_edge = lambda_boundary
        self.lb = lambda_bce
        self.ld = lambda_depth
        self.ltv = lambda_tv

    def compute_depth_loss(self, depth_probs, gt_depth):
        mask = gt_depth > 0
        if not mask.any():
            return torch.tensor(0.0).to(depth_probs.device)
        target = (gt_depth[mask] - 1).long()
        pred = depth_probs.permute(0, 2, 3, 1)[mask]
        return F.nll_loss(torch.log(pred + 1e-9), target)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        depth_probs: torch.Tensor = None,
        gt_depth: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:

        l_lovasz = self.lovasz(logits, targets)
        l_focal = self.focal(logits, targets)
        l_boundary = self.boundary(logits, targets)
        l_bce = self.bce(logits, targets)

        total = (
            self.ll   * l_lovasz
          + self.lf   * l_focal
          + self.lb_edge * l_boundary
          + self.lb   * l_bce
        )

        d_loss = torch.tensor(0.0).to(logits.device)
        tv_loss = torch.tensor(0.0).to(logits.device)

        if depth_probs is not None:
            tv_loss = torch.abs(depth_probs[:, 1:] - depth_probs[:, :-1]).mean()
            total = total + self.ltv * tv_loss

        if depth_probs is not None and gt_depth is not None:
            d_loss = self.compute_depth_loss(depth_probs, gt_depth)
            total = total + self.ld * d_loss

        return {
            "loss": total,
            "lovasz_loss": l_lovasz,
            "focal_loss": l_focal,
            "boundary_loss": l_boundary,
            "bce_loss": l_bce,
            "depth_loss": d_loss,
            "depth_tv_loss": tv_loss,
        }