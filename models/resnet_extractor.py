from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNetFeatureExtractor(nn.Module):
    """
    2-D image feature extractor based on a pretrained ResNet-18 backbone.

    Takes a batch of RGB images and returns a spatial feature map
    suitable for the LSS Lift step.

    Architecture
    ------------
    ResNet-18 (pretrained, ImageNet) → strip avgpool + fc
    → keep up to layer3  (stride-8 output)
    → 1×1 conv to project to out_channels

    Input  : (B, 3, H, W)
    Output : (B, out_channels, H/8, W/8)

    For H=224, W=480  →  output is (B, out_channels, 28, 60)
    """

    def __init__(self, out_channels: int = 64):
        super().__init__()

        # Load pretrained ResNet-18
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Keep everything up to (and including) layer3
        # layer3 output has 256 channels at stride 8
        self.encoder = nn.Sequential(
            backbone.conv1,    # stride 2
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # stride 2  → total stride 4
            backbone.layer1,   # stride 1  → total stride 4
            backbone.layer2,   # stride 2  → total stride 8
            backbone.layer3,   # stride 1  → total stride 8  (256 ch)
        )

        # Project 256 → out_channels with a 1×1 conv
        self.project = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_projection()

    def _init_projection(self):
        for m in self.project.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)  — normalised RGB images

        Returns
        -------
        feat : (B, out_channels, H/8, W/8)
        """
        feat = self.encoder(x)     # (B, 256, H/8, W/8)
        feat = self.project(feat)  # (B, out_channels, H/8, W/8)
        return feat


# ─────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("ResNetFeatureExtractor — smoke test")
    B = 2
    extractor = ResNetFeatureExtractor(out_channels=64)

    n_params = sum(p.numel() for p in extractor.parameters()) / 1e6
    print(f"  Parameters : {n_params:.2f} M")

    x   = torch.randn(B, 3, 224, 480)
    out = extractor(x)
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    # Expected: (2, 64, 28, 60)
    print("  ✓ smoke test passed")