"""U-Net architecture (Ronneberger et al., MICCAI 2015).

Symmetric encoder-decoder with skip connections. 23 convolutional layers total.
Encoder: 64 -> 128 -> 256 -> 512 -> 1024 channels.
Decoder mirrors encoder with transposed convolutions and concatenation.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int, use_padding: bool = True):
        super().__init__()
        padding = 1 if use_padding else 0
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Downsampling block: MaxPool2d(2) followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, use_padding: bool = True):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, use_padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class DecoderBlock(nn.Module):
    """Upsampling block: ConvTranspose2d, center-crop + concat skip, DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, use_padding: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, use_padding)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Center-crop skip connection to match upsampled size (for valid convolutions)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        skip = skip[
            :,
            :,
            diff_h // 2 : diff_h // 2 + x.size(2),
            diff_w // 2 : diff_w // 2 + x.size(3),
        ]
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net segmentation network.

    Args:
        in_channels: Number of input channels (3 for RGB, 1 for grayscale).
        num_classes: Number of output segmentation classes.
        base_features: Number of features in the first encoder block (doubled each level).
        use_padding: If True, use padded convolutions (output size = input size).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_features: int = 64,
        use_padding: bool = True,
    ):
        super().__init__()
        f = base_features

        # Encoder (contracting path)
        self.enc1 = DoubleConv(in_channels, f, use_padding)          # -> f
        self.enc2 = EncoderBlock(f, f * 2, use_padding)              # -> 2f
        self.enc3 = EncoderBlock(f * 2, f * 4, use_padding)          # -> 4f
        self.enc4 = EncoderBlock(f * 4, f * 8, use_padding)          # -> 8f

        # Bottleneck
        self.bottleneck = EncoderBlock(f * 8, f * 16, use_padding)   # -> 16f

        # Decoder (expansive path)
        self.dec4 = DecoderBlock(f * 16, f * 8, use_padding)         # -> 8f
        self.dec3 = DecoderBlock(f * 8, f * 4, use_padding)          # -> 4f
        self.dec2 = DecoderBlock(f * 4, f * 2, use_padding)          # -> 2f
        self.dec1 = DecoderBlock(f * 2, f, use_padding)              # -> f

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(f, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization (kaiming_normal_)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.final_conv(d1)
