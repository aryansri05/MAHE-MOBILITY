# task3_evaluation_iou.py
# ─────────────────────────────────────────────────────────────────
# Person B  |  Task 3: Evaluation Metrics — Occupancy IoU
#
# What this does:
#   1. Takes a model prediction grid (float32 probabilities or logits)
#   2. Takes the ground truth grid (binary, from Task 1)
#   3. Thresholds the prediction to binary (0 or 1)
#   4. Computes Intersection over Union (IoU) for the "occupied" class
#   5. Also computes distance-weighted IoU (penalises near-car errors)
#   6. Generates a visual diff map to show exactly where mistakes are
#
# IoU formula:
#   IoU = |prediction ∩ ground_truth| / |prediction ∪ ground_truth|
#        = true_positives / (true_positives + false_positives + false_negatives)
#
# This is THE metric the judges will use. Higher = better.
# ─────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import Dict

from mahe_mobility.config import X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION, GRID_W, GRID_H, OCC_THRESHOLD


# ═════════════════════════════════════════════════════════════════
# STEP 1 — Core IoU computation
# ═════════════════════════════════════════════════════════════════


@dataclass
class OccupancyMetrics:
    """Container for all evaluation results."""

    iou: float  # standard occupancy IoU
    iou_weighted: float  # distance-weighted IoU
    precision: float  # TP / (TP + FP)
    recall: float  # TP / (TP + FN)
    f1: float  # harmonic mean of precision & recall
    true_positives: int  # occupied cells correctly predicted
    false_positives: int  # free cells wrongly called occupied
    false_negatives: int  # occupied cells missed
    true_negatives: int  # free cells correctly predicted


def compute_occupancy_iou(
    pred_probs: np.ndarray,
    gt: np.ndarray,
    threshold: float = OCC_THRESHOLD,
) -> OccupancyMetrics:
    """
    Compute occupancy IoU and supporting metrics.

    Args:
        pred_probs : model output probabilities  shape (H, W)  float32
                     values in [0, 1]  (apply sigmoid first if logits)
        gt         : ground truth binary grid    shape (H, W)  float32
                     0.0 = free,  1.0 = occupied
        threshold  : probability above which a cell is "occupied"

    Returns:
        OccupancyMetrics dataclass
    """
    assert pred_probs.shape == gt.shape, (
        f"Shape mismatch: pred {pred_probs.shape} vs gt {gt.shape}"
    )
    assert pred_probs.shape == (GRID_H, GRID_W), (
        f"Expected ({GRID_H},{GRID_W}), got {pred_probs.shape}"
    )

    # ── Binarise prediction ───────────────────────────────────────
    pred_bin = (pred_probs >= threshold).astype(bool)
    gt_bin = (gt > 0.5).astype(bool)

    # ── Confusion matrix components ───────────────────────────────
    # Use fast bitwise logic to compute confusion matrix values directly
    TP = int((pred_bin & gt_bin).sum())  # hit
    FP = int((pred_bin & ~gt_bin).sum())  # false alarm
    FN = int((~pred_bin & gt_bin).sum())  # miss
    TN = pred_bin.size - TP - FP - FN  # correct reject

    # ── Standard IoU ─────────────────────────────────────────────
    #   IoU = TP / (TP + FP + FN)
    denominator = TP + FP + FN
    iou = TP / denominator if denominator > 0 else 0.0

    # ── Precision / Recall / F1 ───────────────────────────────────
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # ── Distance-weighted IoU ─────────────────────────────────────
    #   Same as IoU but each cell's contribution is scaled by its
    #   distance weight, so near-car errors matter more.
    iou_weighted = _weighted_iou(pred_bin, gt_bin)

    return OccupancyMetrics(
        iou=iou,
        iou_weighted=iou_weighted,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=TP,
        false_positives=FP,
        false_negatives=FN,
        true_negatives=TN,
    )


_DISTANCE_WEIGHT_MASK = None


def _weighted_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """
    Distance-weighted IoU.

    Instead of counting cells equally (each cell = 1), each cell
    contributes  weight = 1/(distance+1)  to the TP/FP/FN sums.
    """
    global _DISTANCE_WEIGHT_MASK
    if _DISTANCE_WEIGHT_MASK is None:
        # Build weight mask once (same formula as Task 2)
        col_c = X_MIN + (np.arange(GRID_W) + 0.5) * RESOLUTION
        row_c = Y_MIN + (np.arange(GRID_H) + 0.5) * RESOLUTION
        X = col_c[None, :]
        Y = row_c[:, None]
        _DISTANCE_WEIGHT_MASK = 1.0 / (np.hypot(X, Y) + 1.0)  # (H, W)

    W = _DISTANCE_WEIGHT_MASK

    w_TP = W[pred_bin & gt_bin].sum()
    w_FP = W[pred_bin & ~gt_bin].sum()
    w_FN = W[~pred_bin & gt_bin].sum()

    denom = w_TP + w_FP + w_FN
    return float(w_TP / denom) if denom > 0 else 0.0


# ═════════════════════════════════════════════════════════════════
# STEP 2 — Pretty-print results
# ═════════════════════════════════════════════════════════════════


def print_metrics(m: OccupancyMetrics, label: str = ""):
    header = f"── Evaluation Metrics {label} ──"
    print(header)
    print(f"  Occupancy IoU          : {m.iou:.4f}   ← primary judge metric")
    print(f"  Distance-weighted IoU  : {m.iou_weighted:.4f}")
    print(f"  Precision              : {m.precision:.4f}")
    print(f"  Recall                 : {m.recall:.4f}")
    print(f"  F1 score               : {m.f1:.4f}")
    print(f"  True  positives        : {m.true_positives:,}")
    print(f"  False positives        : {m.false_positives:,}  (hallucinated obstacles)")
    print(f"  False negatives        : {m.false_negatives:,}  (missed obstacles)")
    print(f"  True  negatives        : {m.true_negatives:,}")


# ═════════════════════════════════════════════════════════════════
# STEP 3 — Visual error map
# ═════════════════════════════════════════════════════════════════


def visualise_error_map(
    pred_probs: np.ndarray,
    gt: np.ndarray,
    threshold: float = OCC_THRESHOLD,
    save_path: str = None,
):
    """
    4-panel diagnostic plot:
      [0] Ground truth occupancy
      [1] Model prediction (probabilities)
      [2] Binarised prediction
      [3] Error map  — colour-coded TP / FP / FN / TN
    """
    pred_bin = (pred_probs >= threshold).astype(bool)
    gt_bin = (gt > 0.5).astype(bool)

    # Error map: 4-class integer grid
    #   0 = TN (both free)     → dark background
    #   1 = TP (both occupied) → green
    #   2 = FP (pred occ, gt free) → red
    #   3 = FN (pred free, gt occ) → orange
    error_map = np.zeros((GRID_H, GRID_W), dtype=int)
    error_map[pred_bin & gt_bin] = 1  # TP
    error_map[pred_bin & ~gt_bin] = 2  # FP
    error_map[~pred_bin & gt_bin] = 3  # FN

    cmap_err = plt.cm.colors.ListedColormap(
        [
            "#1a1a2e",  # TN  — dark blue
            "#00b09b",  # TP  — teal
            "#e63946",  # FP  — red
            "#f4a261",
        ]  # FN  — orange
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    extent = [X_MIN, X_MAX, Y_MIN, Y_MAX]
    flip = lambda g: np.flipud(g)

    # Panel 0: ground truth
    axes[0].imshow(flip(gt), cmap="gray", origin="lower", extent=extent)
    axes[0].set_title("Ground truth (LiDAR)")

    # Panel 1: prediction probabilities
    im = axes[1].imshow(
        flip(pred_probs), cmap="hot", origin="lower", extent=extent, vmin=0, vmax=1
    )
    axes[1].set_title("Prediction (probability)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Panel 2: binarised prediction
    axes[2].imshow(
        flip(pred_bin.astype(float)), cmap="gray", origin="lower", extent=extent
    )
    axes[2].set_title(f"Prediction (threshold={threshold})")

    # Panel 3: error map
    axes[3].imshow(
        flip(error_map), cmap=cmap_err, origin="lower", extent=extent, vmin=0, vmax=3
    )
    axes[3].set_title("Error map")
    patches = [
        mpatches.Patch(color="#1a1a2e", label="TN — correct free"),
        mpatches.Patch(color="#00b09b", label="TP — correct occupied"),
        mpatches.Patch(color="#e63946", label="FP — false alarm"),
        mpatches.Patch(color="#f4a261", label="FN — missed obstacle"),
    ]
    axes[3].legend(handles=patches, loc="upper right", fontsize=7)

    for ax in axes:
        ax.plot(0, 0, marker="^", color="cyan", markersize=8, zorder=5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    plt.suptitle("BEV Occupancy Evaluation", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Error map saved → {save_path}")
    plt.show()


# ═════════════════════════════════════════════════════════════════
# STEP 4 — Batch evaluation (run over many samples)
# ═════════════════════════════════════════════════════════════════


def evaluate_dataset(
    pred_list: list,  # list of (H,W) np.ndarray  — model outputs
    gt_list: list,  # list of (H,W) np.ndarray  — ground truths
    threshold: float = OCC_THRESHOLD,
) -> Dict[str, float]:
    """
    Evaluate over multiple samples and return mean metrics.

    Args:
        pred_list : list of probability grids  (one per sample)
        gt_list   : list of ground truth grids (one per sample)

    Returns:
        dict with keys: 'mean_iou', 'mean_iou_weighted', 'mean_f1'
    """
    assert len(pred_list) == len(gt_list), "Lists must be the same length"

    ious, iou_ws, f1s = [], [], []
    for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
        m = compute_occupancy_iou(pred, gt, threshold)
        ious.append(m.iou)
        iou_ws.append(m.iou_weighted)
        f1s.append(m.f1)

    results = {
        "mean_iou": float(np.mean(ious)),
        "mean_iou_weighted": float(np.mean(iou_ws)),
        "mean_f1": float(np.mean(f1s)),
        "std_iou": float(np.std(ious)),
        "n_samples": len(ious),
    }

    print(f"\n── Dataset Evaluation ({results['n_samples']} samples) ──")
    print(
        f"  Mean IoU           : {results['mean_iou']:.4f} ± {results['std_iou']:.4f}"
    )
    print(f"  Mean weighted IoU  : {results['mean_iou_weighted']:.4f}")
    print(f"  Mean F1            : {results['mean_f1']:.4f}")
    return results


# ═════════════════════════════════════════════════════════════════
# MAIN — demo with synthetic data (no nuScenes needed to test)
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── Running IoU evaluation demo with synthetic data ──\n")

    # ── Create fake ground truth ──────────────────────────────────
    rng = np.random.default_rng(42)
    gt = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    # Simulate a car ahead (Y = 10-15m, X = -2 to +2)
    r0 = int((10 - Y_MIN) / RESOLUTION)
    r1 = int((15 - Y_MIN) / RESOLUTION)
    c0 = int((-2 - X_MIN) / RESOLUTION)
    c1 = int((2 - X_MIN) / RESOLUTION)
    gt[r0:r1, c0:c1] = 1.0

    # Simulate a wall on the right (X = 20-22m, Y = 5-40m)
    rw0 = int((5 - Y_MIN) / RESOLUTION)
    rw1 = int((40 - Y_MIN) / RESOLUTION)
    cw0 = int((20 - X_MIN) / RESOLUTION)
    cw1 = int((22 - X_MIN) / RESOLUTION)
    gt[rw0:rw1, cw0:cw1] = 1.0

    # ── Create a "decent" prediction (slightly offset + noisy) ────
    pred_good = gt.copy()
    pred_good = np.roll(pred_good, shift=2, axis=0)  # 0.4m offset
    pred_good = np.clip(pred_good + rng.normal(0, 0.1, gt.shape), 0, 1)

    # ── Create a "bad" prediction (many errors near the car) ──────
    pred_bad = pred_good.copy()
    near_rows = GRID_H // 8
    # Flip predictions in the near zone
    pred_bad[:near_rows, :] = 1.0 - pred_bad[:near_rows, :]

    # ── Evaluate both ─────────────────────────────────────────────
    print("Good prediction:")
    m_good = compute_occupancy_iou(pred_good, gt)
    print_metrics(m_good, label="(good pred)")

    print("\nBad prediction (errors near car):")
    m_bad = compute_occupancy_iou(pred_bad, gt)
    print_metrics(m_bad, label="(bad pred — near car errors)")

    print(f"\n── Impact of near-car errors ──")
    print(f"  Standard IoU gap   : {m_good.iou - m_bad.iou:.4f}")
    print(
        f"  Weighted IoU gap   : {m_good.iou_weighted - m_bad.iou_weighted:.4f}  ← should be larger"
    )

    # ── Visual error map ──────────────────────────────────────────
    visualise_error_map(pred_good, gt, save_path="error_map_good.png")
    visualise_error_map(pred_bad, gt, save_path="error_map_bad.png")

    # ── Batch evaluation demo ─────────────────────────────────────
    evaluate_dataset(
        pred_list=[pred_good] * 5 + [pred_bad] * 3,
        gt_list=[gt] * 8,
    )

    print("\n── How to use with real nuScenes data ──")
    print("""
    from mahe_mobility.tasks.task1_lidar_to_occupancy import load_nuscenes, load_lidar_ego_frame, lidar_to_occupancy
    from mahe_mobility.tasks.task3_evaluation_iou import compute_occupancy_iou, print_metrics, visualise_error_map

    nusc = load_nuscenes("/data/nuscenes")

    for sample in nusc.sample[:10]:
        gt   = lidar_to_occupancy(load_lidar_ego_frame(nusc, sample["token"]))
        pred = model_predict(sample)          # your model's output
        m    = compute_occupancy_iou(pred, gt)
        print_metrics(m)
    """)
