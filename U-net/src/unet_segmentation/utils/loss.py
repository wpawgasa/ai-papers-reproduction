"""Weighted cross-entropy loss for U-Net (Ronneberger et al., 2015).

Implements Equation (1) from the paper: pixel-wise weighted cross-entropy.
Weight map (Equation 2) balances class frequencies and emphasizes borders
between touching instances.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def compute_class_weights(mask: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Compute inverse-frequency class weights for balancing.

    Args:
        mask: Ground truth segmentation mask (H, W) with integer class labels.
        num_classes: Number of segmentation classes.

    Returns:
        Array of shape (num_classes,) with per-class weights.
    """
    counts = np.bincount(mask.ravel(), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights /= weights.sum()
    return weights.astype(np.float32)


def compute_weight_map(
    segmentation: np.ndarray,
    w0: float = 10.0,
    sigma: float = 5.0,
) -> np.ndarray:
    """Compute pixel-wise weight map per Equation (2) of the paper.

    w(x) = w_c(x) + w0 * exp(-(d1(x) + d2(x))^2 / (2 * sigma^2))

    where d1 is the distance to the nearest cell border and d2 is the distance
    to the second nearest cell border. This emphasizes borders between touching
    instances of the same class.

    Note: This requires instance-level segmentation masks. For binary masks
    without instance labels (e.g., Oxford Pets), only class frequency
    balancing is applied.

    Args:
        segmentation: Instance-level mask (H, W) where each instance has a unique ID,
                      and 0 is background.
        w0: Border weight magnitude.
        sigma: Border weight width parameter.

    Returns:
        Weight map of shape (H, W).
    """
    # Class frequency balancing
    binary_mask = (segmentation > 0).astype(np.float32)
    total = binary_mask.size
    fg_fraction = binary_mask.sum() / total
    bg_fraction = 1.0 - fg_fraction

    w_c = np.where(binary_mask > 0, 1.0 / max(fg_fraction, 1e-6), 1.0 / max(bg_fraction, 1e-6))
    w_c = w_c / w_c.mean()

    # Instance border emphasis (only if instance labels exist)
    instance_ids = np.unique(segmentation)
    instance_ids = instance_ids[instance_ids > 0]

    if len(instance_ids) < 2:
        return w_c

    # Compute distance to each instance border
    distances = []
    for inst_id in instance_ids:
        inst_mask = segmentation == inst_id
        dist = distance_transform_edt(~inst_mask)
        distances.append(dist)

    distances = np.stack(distances, axis=0)
    distances.sort(axis=0)

    d1 = distances[0]
    d2 = distances[1]

    w_border = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma**2))
    return w_c + w_border


class WeightedCrossEntropyLoss(nn.Module):
    """Pixel-wise weighted cross-entropy loss (Equation 1).

    For use with pre-computed weight maps. Falls back to standard
    cross-entropy with class frequency balancing when no weight maps
    are provided.
    """

    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pixel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits: Model output of shape (B, C, H, W).
            targets: Ground truth labels of shape (B, H, W) with integer class indices.
            pixel_weights: Optional per-pixel weight map of shape (B, H, W).

        Returns:
            Scalar loss value.
        """
        if pixel_weights is not None:
            loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
            loss = (loss * pixel_weights).mean()
        else:
            loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        return loss
