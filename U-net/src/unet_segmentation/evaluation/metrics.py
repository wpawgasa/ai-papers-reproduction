"""Evaluation metrics for segmentation: IOU, Dice, pixel accuracy."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SegmentationMetrics:
    """Container for segmentation evaluation results."""

    loss: float
    mean_iou: float
    pixel_accuracy: float
    dice: float


def compute_iou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> tuple[torch.Tensor, float]:
    """Compute per-class and mean Intersection over Union.

    Args:
        predictions: Predicted class labels (B, H, W).
        targets: Ground truth class labels (B, H, W).
        num_classes: Number of segmentation classes.

    Returns:
        Tuple of (per_class_iou tensor, mean_iou float).
    """
    ious = []
    for cls in range(num_classes):
        pred_mask = predictions == cls
        target_mask = targets == cls
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union == 0:
            ious.append(union.new_tensor(1.0))  # Empty class — perfect score
        else:
            ious.append(intersection / union)
    per_class = torch.stack(ious)
    return per_class, per_class.mean().item()


def compute_dice(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Dice coefficient (F1 score for segmentation).

    Dice = 2 * |P ∩ T| / (|P| + |T|)

    Args:
        predictions: Predicted class labels (B, H, W).
        targets: Ground truth class labels (B, H, W).

    Returns:
        Dice coefficient as a float.
    """
    pred_fg = (predictions == 1).float()
    target_fg = (targets == 1).float()
    intersection = (pred_fg * target_fg).sum()
    total = pred_fg.sum() + target_fg.sum()
    if total == 0:
        return 1.0
    return (2.0 * intersection / total).item()


def compute_pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute overall pixel-wise classification accuracy.

    Args:
        predictions: Predicted class labels (B, H, W).
        targets: Ground truth class labels (B, H, W).

    Returns:
        Pixel accuracy as a float.
    """
    correct = (predictions == targets).sum().float()
    total = targets.numel()
    return (correct / total).item()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int = 2,
) -> SegmentationMetrics:
    """Run full evaluation on a dataset.

    Args:
        model: Trained U-Net model.
        dataloader: Test/validation DataLoader.
        criterion: Loss function.
        device: Compute device.
        num_classes: Number of segmentation classes.

    Returns:
        SegmentationMetrics with aggregated results.
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    n_batches = 0

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        _, batch_iou = compute_iou(preds, masks, num_classes)
        total_iou += batch_iou
        total_dice += compute_dice(preds, masks)
        total_pixel_acc += compute_pixel_accuracy(preds, masks)
        n_batches += 1

    n = max(n_batches, 1)
    return SegmentationMetrics(
        loss=total_loss / n,
        mean_iou=total_iou / n,
        pixel_accuracy=total_pixel_acc / n,
        dice=total_dice / n,
    )
