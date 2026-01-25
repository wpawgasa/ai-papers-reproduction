"""
ResNet models for CIFAR-10.

Implements the residual network with skip connections following the paper's
CIFAR-10 architecture (depth = 6n + 2).
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """CIFAR ResNet basic block: (3x3 -> BN -> ReLU) x2 + skip."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.proj = None
        if stride != 1 or in_ch != out_ch:
            # Option B-style: projection when dimensions change
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = F.relu(out + identity, inplace=True)
        return out


class CIFARStem(nn.Module):
    """Initial convolution layer for CIFAR-10."""

    def __init__(self, out_ch: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(3, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResNetCIFAR(nn.Module):
    """
    CIFAR ResNet of depth: 6n + 2 (paper's CIFAR design).

    Example: depth=20 => n=3 blocks per stage; 56 => n=9.

    Args:
        depth: Network depth (must be 6n+2, e.g., 20, 32, 44, 56, 110)
        num_classes: Number of output classes (default: 10 for CIFAR-10)
    """

    def __init__(self, depth: int, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "For CIFAR ResNet, depth should be 6n+2 (e.g., 20, 32, 44, 56, 110)."
        n = (depth - 2) // 6

        self.stem = CIFARStem(16)
        self.layer1 = self._make_stage(16, 16, n, stride=1)
        self.layer2 = self._make_stage(16, 32, n, stride=2)
        self.layer3 = self._make_stage(32, 64, n, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        # Kaiming init is a common default; close in spirit to He initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
