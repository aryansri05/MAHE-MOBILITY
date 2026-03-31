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
    def predict(
        self,
        feats: torch.Tensor,
        threshold: float = OCC_THRESHOLD,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(feats)
        probs = torch.sigmoid(logits)
        mask = probs > threshold
        return probs, mask


# ═══════════════════════════════════════════════════════════════════
#  SUBUNIT 4 — Loss Functions
# ═══════════════════════════════════════════════════════════════════


class FocalLoss(nn.Module):
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
        loss = focal_weight * bce
        return loss.mean()


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 20.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight.to(logits.device)
        )


class LovaszSoftmaxLoss(nn.Module):
    """
    Binary Lovasz-Softmax loss: direct surrogate for IoU.
    Reference: https://github.com/bermanmaxim/LovaszSoftmax
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)
        signs = 2.0 * targets - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = targets[perm]

        def grad(gt_sorted):
            p = len(gt_sorted)
            gts = gt_sorted.sum()
            intersection = gts - gt_sorted.cumsum(0)
            union = gts + (1 - gt_sorted).cumsum(0)
            jaccard = 1.0 - intersection / union
            if p > 1:
                jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
            return jaccard

        grad_vec = grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad_vec)
        return loss


class BinaryBoundaryLoss(nn.Module):
    """
    Penalizes misalignment of edges to force sharp boundaries.
    Uses Sobel filters to compute spatial gradients.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", sobel_x)
        self.register_buffer("kernel_y", sobel_y)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        grad_x_pred = F.conv2d(pred, self.kernel_x, padding=1)
        grad_y_pred = F.conv2d(pred, self.kernel_y, padding=1)
        grad_x_gt = F.conv2d(targets, self.kernel_x, padding=1)
        grad_y_gt = F.conv2d(targets, self.kernel_y, padding=1)
        edge_pred = torch.sqrt(grad_x_pred**2 + grad_y_pred**2 + 1e-6)
        edge_gt = torch.sqrt(grad_x_gt**2 + grad_y_gt**2 + 1e-6)
        return F.mse_loss(edge_pred, edge_gt)


# ═══════════════════════════════════════════════════════════════════
#  SUBUNIT 4 — Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def occupancy_iou(pred_mask, gt_mask, eps=1e-6):
    pred = pred_mask.bool().float()
    gt = gt_mask.bool().float()
    intersection = (pred * gt).sum(dim=(1, 2, 3))
    union = (pred + gt).clamp(max=1).sum(dim=(1, 2, 3))
    return ((intersection + eps) / (union + eps)).mean()


def _build_distance_weight_map(device):
    rows = torch.arange(GRID_H, dtype=torch.float32, device=device)
    cols = torch.arange(GRID_W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    x_m = grid_x * RESOLUTION + X_MIN + RESOLUTION / 2
    y_m = grid_y * RESOLUTION + Y_MIN + RESOLUTION / 2
    dist = torch.sqrt(x_m**2 + y_m**2)
    return (1.0 / (dist + 1.0)).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def distance_weighted_error(pred_probs, gt_mask):
    weights = _build_distance_weight_map(pred_probs.device)
    abs_err = (pred_probs - gt_mask.float()).abs()
    weighted = (abs_err * weights).sum(dim=(1, 2, 3))
    return (weighted / weights.sum()).mean()


# ═══════════════════════════════════════════════════════════════════
class OccupancyCriterion(nn.Module):
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float = 20.0,
        lambda_lovasz: float = 1.0,  # IoU Surgery: Main driver
        lambda_boundary: float = 1.0, # Accuracy Push: Sharper edges
        lambda_focal: float = 0.5,
        lambda_bce: float = 0.2,
        lambda_depth: float = 1.0,
        lambda_tv: float = 0.05
    ):
        super().__init__()
        self.lovasz = LovaszSoftmaxLoss()
        self.boundary = BinaryBoundaryLoss()
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.bce = WeightedBCELoss(pos_weight)
        self.ll, self.lbound, self.lf, self.lb, self.ld, self.ltv = \
            lambda_lovasz, lambda_boundary, lambda_focal, lambda_bce, lambda_depth, lambda_tv

    def compute_depth_loss(self, depth_probs, gt_depth):
        mask = gt_depth > 0
        if not mask.any():
            return torch.tensor(0.0).to(depth_probs.device)
        target = (gt_depth[mask] - 1).long()
        pred = depth_probs.permute(0, 2, 3, 1)[mask]
        return F.nll_loss(torch.log(pred + 1e-9), target)

    def forward(self, logits, targets, depth_probs=None, gt_depth=None):
        l_lovasz = self.lovasz(logits, targets)
        l_boundary = self.boundary(logits, targets)
        l_focal = self.focal(logits, targets)
        l_bce = self.bce(logits, targets)
        total = (self.ll * l_lovasz + self.lbound * l_boundary + self.lf * l_focal + self.lb * l_bce)
        d_loss = torch.tensor(0.0).to(logits.device)
        tv_loss = torch.tensor(0.0).to(logits.device)
        if depth_probs is not None:
            tv_loss = torch.abs(depth_probs[:, 1:] - depth_probs[:, :-1]).mean()
            total += self.ltv * tv_loss
        if depth_probs is not None and gt_depth is not None:
            d_loss = self.compute_depth_loss(depth_probs, gt_depth)
            total += self.ld * d_loss
        return {
            "loss": total, "lovasz_loss": l_lovasz, "boundary_loss": l_boundary,
            "focal_loss": l_focal, "bce_loss": l_bce, "depth_loss": d_loss, "depth_tv_loss": tv_loss
        }


if __name__ == "__main__":
    print("Occupancy Criterion (IoU Surgery) — smoke test")
    B, GRID_H, GRID_W = 2, 250, 250
    logits = torch.randn(B, 1, GRID_H, GRID_W, requires_grad=True)
    targets = (torch.rand(B, 1, GRID_H, GRID_W) > 0.95).float()
    criterion = OccupancyCriterion()
    losses = criterion(logits, targets)
    print(f"  Total Loss    : {losses['loss'].item():.4f}")
    print(f"  Lovasz Loss   : {losses['lovasz_loss'].item():.4f}")
    print(f"  Boundary Loss : {losses['boundary_loss'].item():.4f}")
    losses['loss'].backward()
    if logits.grad is not None:
        print(f"  Logits Grad   : {logits.grad.abs().mean().item():.6f}")
        print("\n  ✓ Backprop successful. Metric Surgery complete.")