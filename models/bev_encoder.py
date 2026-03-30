import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GRID_H, GRID_W


# ─────────────────────────────────────────────
#  Building blocks
# ─────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU. The standard residual unit brick."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    """
    Pre-activation residual block (He et al. 2016 v2).
    Keeps spatial resolution.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        self.proj = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv1(x)
        x = F.relu(self.bn2(x), inplace=True)
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Downsample by 2× then apply N residual blocks."""
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 2):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_blocks):
            layers.append(ResBlock(out_ch, out_ch))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpBlock(nn.Module):
    """
    Simple ConvBNReLU for the decoder arm.
    The actual spatial resizing is handled in BEVEncoder.forward via F.interpolate.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # REMOVED: nn.Upsample(scale_factor=2) to prevent "Double-Size" error
        self.conv = ConvBNReLU(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─────────────────────────────────────────────
#  BEV Encoder
# ─────────────────────────────────────────────

class BEVEncoder(nn.Module):
    """U-Net style encoder for the 250×250 BEV feature map."""

    def __init__(
        self,
        in_channels:   int = 64,
        base_channels: int = 64,
        out_channels:  int = 128,
    ):
        super().__init__()
        C = base_channels

        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, C, kernel=3),
            ResBlock(C, C),
        )

        # Encoder
        self.down1 = DownBlock(C,     C * 2, n_blocks=2) # 250 -> 125
        self.down2 = DownBlock(C * 2, C * 4, n_blocks=2) # 125 -> 63

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(C * 4, C * 4),
            ResBlock(C * 4, C * 4),
        )

        # Decoder
        self.up1   = UpBlock(C * 4 + C * 2, C * 2)
        self.skip1 = ResBlock(C * 2, C * 2)

        self.up2   = UpBlock(C * 2 + C, C)
        self.skip2 = ResBlock(C, C)

        # Output projection
        self.out_proj = ConvBNReLU(C, out_channels, kernel=1, padding=0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s0 = self.stem(x)              # (B, C, 250, 250)
        s1 = self.down1(s0)            # (B, 2C, 125, 125)
        s2 = self.down2(s1)            # (B, 4C, 63, 63)
        b  = self.bottleneck(s2)       # (B, 4C, 63, 63)

        # Decoder 1: Upsample 63 -> 125
        # Explicit size ensures we match s1 exactly, avoiding off-by-one errors
        b_up = F.interpolate(b, size=s1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.up1(torch.cat([b_up, s1], dim=1))
        d1 = self.skip1(d1)

        # Decoder 2: Upsample 125 -> 250
        # Explicit size ensures we match s0 exactly, avoiding Double-Size errors
        d1_up = F.interpolate(d1, size=s0.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.up2(torch.cat([d1_up, s0], dim=1))
        d2 = self.skip2(d2)

        return self.out_proj(d2) # (B, out_channels, 250, 250)


# ─────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("BEV Encoder — smoke test")
    B, C_in = 2, 64
    enc = BEVEncoder(in_channels=C_in, base_channels=64, out_channels=128)

    n_params = sum(p.numel() for p in enc.parameters()) / 1e6
    print(f"  Parameters : {n_params:.2f} M")

    x   = torch.randn(B, C_in, GRID_H, GRID_W)
    out = enc(x)
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    assert out.shape == (B, 128, GRID_H, GRID_W), "Shape mismatch!"
    print("  ✓ shape correct")