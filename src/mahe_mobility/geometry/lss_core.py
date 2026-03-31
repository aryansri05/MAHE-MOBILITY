from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ─────────────────────────────────────────────
#  Configuration dataclasses
# ─────────────────────────────────────────────


@dataclass
class CameraConfig:
    """Intrinsic and image-plane parameters for a single camera."""

    image_h: int = 256
    image_w: int = 704
    fx: float = 458.654
    fy: float = 457.296
    cx: float = 367.215
    cy: float = 248.375


@dataclass
class BEVGridConfig:
    """Bird's-Eye-View grid extent and resolution."""

    x_min: float = -51.2
    x_max: float = 51.2
    y_min: float = -51.2
    y_max: float = 51.2
    z_min: float = -5.0
    z_max: float = 3.0
    cell_size: float = 0.2  # 20 cm × 20 cm cells


@dataclass
class DepthConfig:
    """Discrete depth bins for the frustum."""

    d_min: float = 4.0
    d_max: float = 45.0
    d_steps: int = 41 # Restore to 41 for maximum accuracy


# ─────────────────────────────────────────────
#  Helper: build intrinsic matrix
# ─────────────────────────────────────────────


def build_intrinsic_matrix(cfg: CameraConfig) -> torch.Tensor:
    K = torch.zeros(3, 3, dtype=torch.float32)
    K[0, 0] = cfg.fx
    K[1, 1] = cfg.fy
    K[0, 2] = cfg.cx
    K[1, 2] = cfg.cy
    K[2, 2] = 1.0
    return K


# ─────────────────────────────────────────────
#  TASK 1 — Frustum Generator
# ─────────────────────────────────────────────


class FrustumGenerator(nn.Module):
    def __init__(self, cam_cfg: CameraConfig, depth_cfg: DepthConfig):
        super().__init__()
        self.cam_cfg = cam_cfg
        self.depth_cfg = depth_cfg
        frustum = self._build_frustum()
        self.register_buffer("frustum", frustum)  # (D, H, W, 3)

    def _build_frustum(self) -> torch.Tensor:
        cfg = self.cam_cfg
        dcfg = self.depth_cfg
        H, W = cfg.image_h, cfg.image_w
        D = dcfg.d_steps

        K = build_intrinsic_matrix(cfg)
        K_inv = torch.linalg.inv(K)

        u = torch.arange(W, dtype=torch.float32)
        v = torch.arange(H, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")

        ones = torch.ones_like(uu)
        uvone = torch.stack([uu, vv, ones], dim=0).reshape(3, -1)
        rays = K_inv @ uvone  # (3, H·W)

        # ── Logarithmic Depth Binning ─────────────────────────────────
        # Ultra-high near-field precision: ~0.3m at 4m, ~2m at 45m
        # Formula: d_i = d_min * (d_max/d_min)^(i/(D-1))
        depths = dcfg.d_min * (dcfg.d_max / dcfg.d_min) ** (
            torch.arange(D, dtype=torch.float32) / (D - 1)
        )
        pts = rays.unsqueeze(0) * depths.reshape(D, 1, 1)  # (D, 3, H·W)
        pts = pts.permute(0, 2, 1).reshape(D, H, W, 3)

        return pts

    def forward(self, ego2cam: Optional[torch.Tensor] = None) -> torch.Tensor:
        pts = self.frustum
        if ego2cam is not None:
            cam2ego = torch.linalg.inv(ego2cam)
            R, t = cam2ego[:3, :3], cam2ego[:3, 3]
            pts_flat = pts.reshape(-1, 3)
            pts = (pts_flat @ R.T + t).reshape(*pts.shape[:3], 3)
        return pts


# ─────────────────────────────────────────────
#  TASK 3 — Depth Pre-computation
# ─────────────────────────────────────────────


class DepthPrecomputer(nn.Module):
    def __init__(
        self,
        frustum_gen: FrustumGenerator,
        bev_cfg: BEVGridConfig,
        ego2cam: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.bev_cfg = bev_cfg
        cfg = bev_cfg
        self.bev_W = round((cfg.x_max - cfg.x_min) / cfg.cell_size)
        self.bev_H = round((cfg.y_max - cfg.y_min) / cfg.cell_size)

        with torch.no_grad():
            pts_ego = frustum_gen(ego2cam)

        bev_xi, bev_yi, valid = self._pts_to_bev_indices(pts_ego)
        self.register_buffer("bev_xi", bev_xi)
        self.register_buffer("bev_yi", bev_yi)
        self.register_buffer("valid", valid)

        flat_idx = bev_yi * self.bev_W + bev_xi
        flat_idx[~valid] = 0
        self.register_buffer("flat_idx", flat_idx.long())

    def _pts_to_bev_indices(
        self, pts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.bev_cfg
        X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]

        xi = ((X - cfg.x_min) / cfg.cell_size).floor().long()
        yi = ((Y - cfg.y_min) / cfg.cell_size).floor().long()

        valid = (
            (xi >= 0)
            & (xi < self.bev_W)
            & (yi >= 0)
            & (yi < self.bev_H)
            & (Z >= cfg.z_min)
            & (Z <= cfg.z_max)
        )
        xi = xi.clamp(0, self.bev_W - 1)
        yi = yi.clamp(0, self.bev_H - 1)
        return xi, yi, valid

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (self.bev_H, self.bev_W)


# ─────────────────────────────────────────────
#  TASK 2 — Voxel Pooling ("The Splat")
# ─────────────────────────────────────────────


class VoxelPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        features: torch.Tensor,
        precomp: DepthPrecomputer,
    ) -> torch.Tensor:
        B, C, D, img_H, img_W = features.shape
        bev_H, bev_W = precomp.grid_shape

        valid_b = precomp.valid.unsqueeze(0).unsqueeze(0).float()
        features = features * valid_b

        feat_flat = features.reshape(B, C, -1)
        idx_flat = precomp.flat_idx.reshape(-1)
        N = idx_flat.shape[0]

        bev_flat = torch.zeros(
            B, C, bev_H * bev_W, dtype=features.dtype, device=features.device
        )
        idx_exp = idx_flat.unsqueeze(0).unsqueeze(0).expand(B, C, N)
        bev_flat.scatter_add_(2, idx_exp, feat_flat)

        return bev_flat.reshape(B, C, bev_H, bev_W)


# ─────────────────────────────────────────────
class GeometryArchitect(nn.Module):
    def __init__(
        self,
        cam_cfg: CameraConfig = CameraConfig(),
        bev_cfg: BEVGridConfig = BEVGridConfig(),
        depth_cfg: DepthConfig = DepthConfig(),
        ego2cam: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.frustum_gen = FrustumGenerator(cam_cfg, depth_cfg)
        self.bev_cfg, self.cam_cfg, self.depth_cfg = bev_cfg, cam_cfg, depth_cfg
        
        # Initial precomputation (standard ego frame)
        initial_ego2cam = ego2cam if ego2cam is not None else torch.eye(4)
        self.precomp = DepthPrecomputer(self.frustum_gen, bev_cfg, initial_ego2cam)
        
        self.voxel_pool = VoxelPooling()
        
        # Device-aware cache for dynamic extrinsics/augmentation
        self._cache = {}
        self._max_cache_size = 64 # Enough for a full batch + val offsets

    def _get_cache_key(self, intrinsic: torch.Tensor, extrinsic: torch.Tensor):
        """
        Hash geometric matrices with precision to avoid floating-point jitter.
        K: 6 decimals, E: 4 decimals for translation, 6 for rotation.
        """
        # Rounded tuples are reliable keys
        K_key = tuple(torch.round(intrinsic.flatten(), decimals=6).tolist())
        E_key = tuple(torch.round(extrinsic.flatten(), decimals=6).tolist())
        return (K_key, E_key)

    def forward(
        self, 
        features: torch.Tensor, 
        intrinsic: Optional[torch.Tensor] = None, 
        extrinsic: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        features : (B, C, D, img_H, img_W)
        If intrinsic/extrinsic provided, uses (or builds) cached precomputer.
        """
        device = features.device
        
        # Use current precomp if no geometry provided
        if intrinsic is None or extrinsic is None:
            return self.voxel_pool(features, self.precomp)
            
        # Per-batch caching logic
        # Note: In LSS, every sample in B typically uses the same rig calibration.
        # If heterogeneous extrinsics are needed, this should loop or batch-cache.
        # We assume batch-uniform geometry for now (standard for NuScenes rig).
        cache_key = self._get_cache_key(intrinsic[0].cpu(), extrinsic[0].cpu())
        
        if cache_key not in self._cache:
            # Build and cache on-device
            if len(self._cache) >= self._max_cache_size:
                self._cache.pop(next(iter(self._cache))) # Simple FIFO eviction
            
            new_precomp = DepthPrecomputer(
                self.frustum_gen, self.bev_cfg, extrinsic[0]
            ).to(device)
            self._cache[cache_key] = new_precomp
            
        return self.voxel_pool(features, self._cache[cache_key])

    @property
    def frustum_shape(self) -> Tuple[int, int, int]:
        return (self.depth_cfg.d_steps, self.cam_cfg.image_h, self.cam_cfg.image_w)

    @property
    def bev_shape(self) -> Tuple[int, int]:
        return self.precomp.grid_shape

    def __repr__(self) -> str:
        D, H, W = self.frustum_shape
        bH, bW = self.bev_shape
        n_valid = self.precomp.valid.sum().item()
        n_total = D * H * W
        pct = 100 * n_valid / n_total
        return (
            f"GeometryArchitect(\n"
            f"  frustum  : {D}d × {H}h × {W}w  = {n_total:,} pts\n"
            f"  valid    : {n_valid:,} pts  ({pct:.1f}% inside BEV extent)\n"
            f"  bev grid : {bH} × {bW}  @ {self.bev_cfg.cell_size * 100:.0f}cm resolution\n"
            f"  depth    : {self.depth_cfg.d_min}m → {self.depth_cfg.d_max}m"
            f"  ({self.depth_cfg.d_steps} bins)\n)"
        )


# ─────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 56)
    print("LSS Geometry Architect — smoke test")
    print("=" * 56)

    cam_cfg = CameraConfig()
    bev_cfg = BEVGridConfig()
    depth_cfg = DepthConfig()
    ego2cam = torch.eye(4)

    print("\n[1/3] Building GeometryArchitect...")
    arch = GeometryArchitect(cam_cfg, bev_cfg, depth_cfg, ego2cam)
    print(arch)

    print("\n[2/3] Frustum point cloud (first 3 depth bins, pixel [0,0]):")
    pts = arch.frustum_gen(ego2cam)
    for d in range(3):
        p = pts[d, 0, 0]
        print(f"  depth bin {d:02d}: X={p[0]:.3f}m  Y={p[1]:.3f}m  Z={p[2]:.3f}m")

    print("\n[3/3] Forward pass — splat features to BEV...")
    B, C = 2, 64
    D, H, W = arch.frustum_shape
    features = torch.randn(B, C, D, H, W)
    bev = arch(features)
    print(f"  input  : (B={B}, C={C}, D={D}, H={H}, W={W})")
    print(f"  output : {tuple(bev.shape)}")
    bH, bW = arch.bev_shape
    print(
        f"  BEV grid: {bH} × {bW} cells  @ "
        f"{bev_cfg.cell_size * 100:.0f}cm resolution "
        f"= {bev_cfg.x_max * 2:.0f}m × {bev_cfg.y_max * 2:.0f}m extent"
    )

    print("\n✓ All tasks complete.")
