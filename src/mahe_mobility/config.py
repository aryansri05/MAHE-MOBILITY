# config.py
# ─────────────────────────────────────────────────────────────────
# Shared grid definition — BOTH Person A and Person B import this.
# Never hardcode these values anywhere else.
# ─────────────────────────────────────────────────────────────────

# BEV grid bounds in metres (ego-vehicle is at origin)
X_MIN, X_MAX = -25.0, 25.0  # left / right
Y_MIN, Y_MAX = 0.0, 50.0  # front only (behind car not needed)

# Grid resolution in metres per cell
RESOLUTION = 0.2

# Derived grid size  →  250 × 250 cells
GRID_W = int((X_MAX - X_MIN) / RESOLUTION)  # 250  (X axis)
GRID_H = int((Y_MAX - Y_MIN) / RESOLUTION)  # 250  (Y axis)

# LiDAR height filter — ignore ground and very high points
Z_MIN = -2.0  # metres  (ground plane ≈ −1.7 m in nuScenes)
Z_MAX = 1.5  # metres  (crop out lamp posts, sky, etc.)

# Occupancy threshold — model logit above this → "occupied"
OCC_THRESHOLD = 0.5

print(f"Grid: {GRID_W} × {GRID_H} cells  |  resolution: {RESOLUTION} m/cell")
print(f"X ∈ [{X_MIN}, {X_MAX}]  Y ∈ [{Y_MIN}, {Y_MAX}]")
