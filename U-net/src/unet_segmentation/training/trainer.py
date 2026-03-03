"""Training loop for U-Net segmentation."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..evaluation.metrics import (
    compute_iou,
    compute_pixel_accuracy,
    evaluate_model,
)
from ..utils.loss import build_criterion


@dataclass
class TrainEpochMetrics:
    """Metrics from a single training epoch."""

    loss: float
    mean_iou: float
    pixel_accuracy: float


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 2,
) -> TrainEpochMetrics:
    """Train for a single epoch.

    Args:
        model: U-Net model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.
        num_classes: Number of segmentation classes.

    Returns:
        TrainEpochMetrics with loss, IOU, and pixel accuracy.
    """
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_pixel_acc = 0.0
    n_batches = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_loss += loss.item()
            _, batch_iou = compute_iou(preds, masks, num_classes)
            total_iou += batch_iou
            total_pixel_acc += compute_pixel_accuracy(preds, masks)
        n_batches += 1

    n = max(n_batches, 1)
    return TrainEpochMetrics(
        loss=total_loss / n,
        mean_iou=total_iou / n,
        pixel_accuracy=total_pixel_acc / n,
    )


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config,
    device: torch.device,
    verbose: bool = True,
    loss_config=None,
) -> dict:
    """Complete training loop with evaluation and history tracking.

    Args:
        model: U-Net model.
        train_loader: Training DataLoader.
        test_loader: Test/validation DataLoader.
        config: TrainConfig with hyperparameters.
        device: Compute device.
        verbose: Print progress every 5 epochs.
        loss_config: LossConfig for weighted cross-entropy. If None, uses plain CE.

    Returns:
        Dictionary with training history (lists of metrics per epoch).
    """
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_schedule_factor,
        patience=config.lr_schedule_patience,
    )

    criterion = build_criterion(loss_config, train_loader, device=device)
    # Prefer model config (ExperimentConfig.model.num_classes) over train config fallback
    if hasattr(config, "model") and hasattr(config.model, "num_classes"):
        num_classes = config.model.num_classes
    else:
        num_classes = getattr(config, "num_classes", 2)

    history = {
        "train_loss": [],
        "train_iou": [],
        "train_pixel_acc": [],
        "test_loss": [],
        "test_iou": [],
        "test_pixel_acc": [],
        "test_dice": [],
        "lr": [],
    }

    for epoch in range(1, config.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=num_classes,
        )

        # Evaluate
        test_metrics = evaluate_model(
            model,
            test_loader,
            criterion,
            device,
            num_classes=num_classes,
        )

        # Learning rate scheduling
        scheduler.step(test_metrics.loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_metrics.loss)
        history["train_iou"].append(train_metrics.mean_iou)
        history["train_pixel_acc"].append(train_metrics.pixel_accuracy)
        history["test_loss"].append(test_metrics.loss)
        history["test_iou"].append(test_metrics.mean_iou)
        history["test_pixel_acc"].append(test_metrics.pixel_accuracy)
        history["test_dice"].append(test_metrics.dice)
        history["lr"].append(current_lr)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == config.epochs):
            print(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"Train Loss: {train_metrics.loss:.4f} | "
                f"Train IOU: {train_metrics.mean_iou:.4f} | "
                f"Test IOU: {test_metrics.mean_iou:.4f} | "
                f"Test Dice: {test_metrics.dice:.4f} | "
                f"LR: {current_lr:.6f}"
            )

    return history
