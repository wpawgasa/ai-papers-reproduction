"""
Multi-layer perceptron model for ParityMNIST experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Simple MLP classifier for MNIST-like tasks.

    Takes flattened image + parity bit as input, outputs logits over 10 classes.
    """

    def __init__(self, hidden1=256, hidden2=256, out_dim=10):
        """
        Initialize MLP.

        Args:
            hidden1: Size of first hidden layer
            hidden2: Size of second hidden layer
            out_dim: Output dimension (number of classes)
        """
        super().__init__()
        # Input: 28*28 pixels + 1 parity bit = 785
        self.fc1 = nn.Linear(28 * 28 + 1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_dim)

    def forward(self, x, parity_bit):
        """
        Forward pass.

        Args:
            x: Image tensor of shape [B, 1, 28, 28]
            parity_bit: Parity indicator of shape [B], values in {0, 1}

        Returns:
            Logits of shape [B, out_dim]
        """
        b = x.shape[0]
        # Flatten image
        x = x.view(b, -1)
        # Append parity bit as feature
        p = parity_bit.float().view(b, 1)
        h = torch.cat([x, p], dim=1)

        # Two hidden layers with ReLU
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        # Output logits
        return self.fc3(h)
