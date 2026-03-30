# task1_lidar_to_occupancy.py
# ─────────────────────────────────────────────────────────────────
# Person B  |  Task 1: LiDAR → 2D Occupancy Ground Truth Map
#
# What this does:
#   1. Loads a nuScenes LiDAR sweep (.bin file — raw point cloud)
#   2. Transforms the points into the ego-vehicle coordinate frame
#   3. Filters out ground plane and sky points (Z filter)
#   4. Maps each surviving (X, Y) point onto a 250×250 grid cell
#   5. Marks that cell as "Occupied" (1)
#   6. Saves the binary grid as a .npy file + visualises it
#
# Output:  ground_truth_occ.npy  — shape (250, 250), dtype float32
#          0.0 = free space,  1.0 = occupied
# ─────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── nuScenes SDK imports ──────────────────────────────────────────
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

# ── Shared config ─────────────────────────────────────────────────
from config import (X_MIN, X_MAX, Y_MIN, Y_MAX,
                    RESOLUTION, GRID_W, GRID_H, Z_MIN, Z_MAX)


# ═════════════════════════════════════════════════════════════════
# STEP 1 — Load nuScenes and pick a sample
# ═════════════════════════════════════════════════════════════════

def load_nuscenes(dataroot: str, version: str = "v1.0-mini"):
    """
    Initialise the nuScenes SDK.

    Args:
        dataroot : path to your nuScenes dataset folder
        version  : 'v1.0-mini' (small) or 'v1.0-trainval' (full)

    Returns:
        NuScenes object
    """
    print(f"Loading nuScenes {version} from {dataroot} ...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    print(f"  {len(nusc.sample)} samples available.")
    return nusc


# ═════════════════════════════════════════════════════════════════
# STEP 2 — Load LiDAR points and transform to ego frame
# ═════════════════════════════════════════════════════════════════

def load_lidar_ego_frame(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """
    Load the LiDAR point cloud for one sample and return points
    expressed in the EGO-VEHICLE coordinate frame.

    nuScenes stores LiDAR data in the SENSOR frame. We need two
    transforms to bring it to ego frame:
        sensor frame  →  [sensor extrinsic]  →  ego frame

    Args:
        nusc         : NuScenes object
        sample_token : token string for the desired sample

    Returns:
        pts_ego : np.ndarray  shape (N, 3)  — (X, Y, Z) in metres
                  X = right,  Y = forward,  Z = up  (RHF convention)
    """
    sample = nusc.get("sample", sample_token)

    # ── Get the LiDAR sensor data record ─────────────────────────
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data  = nusc.get("sample_data", lidar_token)

    # ── Load raw point cloud (.bin) ───────────────────────────────
    # Each point: [X, Y, Z, intensity]  (sensor frame)
    pc = LidarPointCloud.from_file(
        os.path.join(nusc.dataroot, lidar_data["filename"])
    )
    # pc.points shape: (4, N)  — rows are X, Y, Z, intensity

    # ── Build sensor-to-ego transform matrix ─────────────────────
    cs_record   = nusc.get("calibrated_sensor",
                            lidar_data["calibrated_sensor_token"])
    sensor2ego  = transform_matrix(
        translation = cs_record["translation"],
        rotation    = Quaternion(cs_record["rotation"]),
        inverse     = False,
    )  # shape (4, 4)

    # ── Apply transform ───────────────────────────────────────────
    pc.transform(sensor2ego)          # modifies pc.points in-place

    # Return (N, 3) — drop intensity column
    pts_ego = pc.points[:3, :].T      # shape (N, 3)
    print(f"  Loaded {len(pts_ego):,} LiDAR points (ego frame).")
    return pts_ego


# ═════════════════════════════════════════════════════════════════
# STEP 3 — Filter points and build the occupancy grid
# ═════════════════════════════════════════════════════════════════

def lidar_to_occupancy(pts_ego: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) ego-frame LiDAR points into a 2D binary
    occupancy grid.

    Algorithm:
        1. Height filter  — remove ground & very high points
        2. Spatial filter — keep only points inside grid bounds
        3. Discretise     — map continuous (X,Y) → integer (col, row)
        4. Mark cells     — set grid[row, col] = 1.0

    Returns:
        grid : np.ndarray  shape (GRID_H, GRID_W)  dtype float32
               0.0 = free,  1.0 = occupied
    """
    # ── 1. Height filter ─────────────────────────────────────────
    mask_z  = (pts_ego[:, 2] > Z_MIN) & (pts_ego[:, 2] < Z_MAX)
    pts     = pts_ego[mask_z]
    print(f"  After Z filter: {len(pts):,} points remain.")

    # ── 2. Spatial filter (inside grid bounds) ───────────────────
    mask_x  = (pts[:, 0] >= X_MIN) & (pts[:, 0] < X_MAX)
    mask_y  = (pts[:, 1] >= Y_MIN) & (pts[:, 1] < Y_MAX)
    pts     = pts[mask_x & mask_y]
    print(f"  After XY filter: {len(pts):,} points in grid area.")

    # ── 3. Discretise to grid indices ────────────────────────────
    #
    #   col  =  floor( (X - X_MIN) / resolution )   ∈ [0, GRID_W)
    #   row  =  floor( (Y - Y_MIN) / resolution )   ∈ [0, GRID_H)
    #
    #   Note: row 0 = Y_MIN (closest to car front),
    #         row GRID_H-1 = Y_MAX (furthest away)
    #
    col_idx = np.floor((pts[:, 0] - X_MIN) / RESOLUTION).astype(int)
    row_idx = np.floor((pts[:, 1] - Y_MIN) / RESOLUTION).astype(int)

    # Clip to valid range (safety — should be no-op after filter)
    col_idx = np.clip(col_idx, 0, GRID_W - 1)
    row_idx = np.clip(row_idx, 0, GRID_H - 1)

    # ── 4. Fill the grid ─────────────────────────────────────────
    grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    grid[row_idx, col_idx] = 1.0

    occ_pct = 100.0 * grid.sum() / grid.size
    print(f"  Grid shape: {grid.shape}  |  {occ_pct:.1f}% occupied.")
    return grid


# ═════════════════════════════════════════════════════════════════
# STEP 4 — Visualise and save
# ═════════════════════════════════════════════════════════════════

def visualise_occupancy(grid: np.ndarray, save_path: str = None):
    """
    Plot the ground truth occupancy grid.

    The ego vehicle sits at the BOTTOM CENTRE of the image
    (Y=0 = front bumper, X=0 = centre).
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # flip vertically so Y=0 (closest) is at the bottom
    display = np.flipud(grid)

    ax.imshow(
        display,
        cmap    = "RdYlGn_r",   # red = occupied, green = free
        origin  = "lower",
        extent  = [X_MIN, X_MAX, Y_MIN, Y_MAX],
        vmin    = 0, vmax = 1,
    )

    # Mark ego vehicle position
    ax.plot(0, 0, marker="^", color="cyan", markersize=10,
            label="Ego vehicle", zorder=5)

    ax.set_xlabel("X (m) →  right")
    ax.set_ylabel("Y (m) →  forward")
    ax.set_title("Ground Truth Occupancy (LiDAR)")

    occ_patch  = mpatches.Patch(color="red",   label="Occupied")
    free_patch = mpatches.Patch(color="green", label="Free space")
    ax.legend(handles=[occ_patch, free_patch,
                        plt.Line2D([0],[0], marker="^", color="cyan",
                                   linestyle="None", label="Ego")])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved → {save_path}")
    plt.show()


def save_grid(grid: np.ndarray, path: str = "ground_truth_occ.npy"):
    np.save(path, grid)
    print(f"  Grid saved → {path}  shape={grid.shape} dtype={grid.dtype}")


# ═════════════════════════════════════════════════════════════════
# MAIN — wire it all together
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── CONFIG — edit these two lines ────────────────────────────
    DATAROOT     = "/data/nuscenes"      # path to your dataset
    NUSCENES_VER = "v1.0-mini"           # or "v1.0-trainval"
    SAMPLE_IDX   = 0                     # which sample to use (0 = first)
    # ─────────────────────────────────────────────────────────────

    nusc         = load_nuscenes(DATAROOT, NUSCENES_VER)
    sample_token = nusc.sample[SAMPLE_IDX]["token"]
    print(f"\nProcessing sample: {sample_token[:8]}...")

    # Load and transform LiDAR
    pts_ego  = load_lidar_ego_frame(nusc, sample_token)

    # Build ground truth grid
    gt_grid  = lidar_to_occupancy(pts_ego)

    # Save as .npy for Person A to compare against
    save_grid(gt_grid, "ground_truth_occ.npy")

    # Visualise
    visualise_occupancy(gt_grid, save_path="gt_occupancy.png")