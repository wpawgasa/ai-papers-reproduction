"""Training loop implementation."""

import torch
import torch.nn.functional as F
from ..evaluation import evaluate, Metrics


def train_one_epoch(model, loader, optimizer, device) -> Metrics:
    """
    Train model for one epoch.

    Args:
        model: The neural network to train
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Metrics object with training loss and accuracy
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)

    return Metrics(loss=total_loss / n, acc=correct / n)


def train_model(
    model,
    train_loader,
    test_loader,
    config,
    device,
    verbose: bool = True,
):
    """
    Complete training loop with evaluation and learning rate scheduling.

    Args:
        model: Neural network to train
        train_loader: Training data loader
        test_loader: Test data loader
        config: TrainConfig object
        device: Device to train on
        verbose: Whether to print progress

    Returns:
        Dictionary containing training history
    """
    model = model.to(device)

    # Setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )

    # Learning rate scheduler
    milestones = [int(config.epochs * m) for m in config.lr_milestones]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=config.lr_gamma
    )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
    }

    best_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)

        # Evaluate
        test_metrics = evaluate(model, test_loader, device)

        # Update scheduler
        scheduler.step()

        # Record history
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_metrics.loss)
        history["train_acc"].append(train_metrics.acc)
        history["test_loss"].append(test_metrics.loss)
        history["test_acc"].append(test_metrics.acc)
        history["lr"].append(current_lr)

        # Track best
        if test_metrics.acc > best_acc:
            best_acc = test_metrics.acc

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d}/{config.epochs} | LR: {current_lr:.4f} | "
                f"Train Loss: {train_metrics.loss:.3f} Acc: {train_metrics.acc*100:.2f}% | "
                f"Test Loss: {test_metrics.loss:.3f} Acc: {test_metrics.acc*100:.2f}% | "
                f"Best: {best_acc*100:.2f}%"
            )

    if verbose:
        print(f"\nTraining complete. Best test accuracy: {best_acc*100:.2f}%")

    return history
