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
        Weight map of shape (H, W) with dtype float32.
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
        return w_c.astype(np.float32)

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
    return (w_c + w_border).astype(np.float32)


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
            pixel_weights = pixel_weights.to(dtype=loss.dtype)
            weighted_loss = loss * pixel_weights
            denom = pixel_weights.sum().clamp_min(1e-8)
            loss = weighted_loss.sum() / denom
        else:
            loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        return loss


def estimate_class_weights(
    dataloader: torch.utils.data.DataLoader,
    num_classes: int = 2,
    max_batches: int = 50,
) -> torch.Tensor:
    """Estimate inverse-frequency class weights from a dataloader.

    Scans up to max_batches to accumulate pixel counts per class,
    then returns weights proportional to 1/frequency.

    Args:
        dataloader: Training DataLoader yielding (images, masks) tuples.
        num_classes: Number of segmentation classes.
        max_batches: Maximum batches to scan (caps cost on large datasets).

    Returns:
        Tensor of shape (num_classes,) with normalized class weights.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for i, (_, masks) in enumerate(dataloader):
        for cls in range(num_classes):
            counts[cls] += (masks == cls).sum().item()
        if i + 1 >= max_batches:
            break
    counts = counts.clamp(min=1.0)
    weights = 1.0 / counts
    weights /= weights.sum()
    return weights.float()


def build_criterion(
    loss_config=None,
    dataloader: torch.utils.data.DataLoader | None = None,
    num_classes: int = 2,
    device: torch.device | None = None,
) -> nn.Module:
    """Build a loss function from LossConfig.

    When loss_config is provided and ``loss_config.use_class_weights`` is True
    and a training dataloader is available, estimates inverse-frequency class
    weights for balancing. Otherwise falls back to standard (unweighted)
    cross-entropy.

    Args:
        loss_config: LossConfig instance (or None for plain CE).
        dataloader: Training DataLoader for class weight estimation.
        num_classes: Number of segmentation classes.
        device: Target device for the loss module.

    Returns:
        An nn.Module loss function compatible with (logits, targets) calls.
    """
    if loss_config is None:
        return nn.CrossEntropyLoss()

    class_weights = None
    if loss_config.use_class_weights and dataloader is not None:
        class_weights = estimate_class_weights(dataloader, num_classes)
        if device is not None:
            class_weights = class_weights.to(device)

    return WeightedCrossEntropyLoss(class_weights=class_weights)
