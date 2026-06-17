"""Deep Q-Network architecture (Mnih et al., 2013, Section 4.1).

State-in, vector-of-Q-values-out: a single forward pass produces one Q-value
per action, enabling O(1)-action evaluation per state (summary Section 2.2.6).
"""

import torch
import torch.nn as nn

from ..config import ModelConfig


class DQN(nn.Module):
    """The 2013 DQN convolutional network.

    Input:  (B, in_channels, 84, 84) uint8 or float frames in [0, 255].
    Output: (B, num_actions) action-value estimates.

    Architecture (summary Section 2.4):
        Conv 16@8x8 stride 4, ReLU   -> 20x20x16
        Conv 32@4x4 stride 2, ReLU   ->  9x9x32  (flatten 2592)
        FC 256, ReLU
        FC num_actions (linear)
    ~1.7M trainable parameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.features = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.conv1_filters,
                kernel_size=config.conv1_kernel,
                stride=config.conv1_stride,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.conv1_filters,
                config.conv2_filters,
                kernel_size=config.conv2_kernel,
                stride=config.conv2_stride,
            ),
            nn.ReLU(inplace=True),
        )

        conv_out = self._conv_output_size()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, config.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.fc_hidden, config.num_actions),
        )

        self._init_weights()

    def _conv_output_size(self) -> int:
        """Infer the flattened conv feature size from a dummy forward pass."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.in_channels, 84, 84)
            return int(self.features(dummy).flatten(1).shape[1])

    def _init_weights(self) -> None:
        """He initialization (kaiming_normal) for conv and linear layers."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize pixel values to [0, 1]. Accepts uint8 or float input.
        x = x.float() / 255.0
        x = self.features(x)
        return self.head(x)
