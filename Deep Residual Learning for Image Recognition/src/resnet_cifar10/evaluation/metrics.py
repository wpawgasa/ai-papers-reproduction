"""Evaluation metrics and utilities."""

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class Metrics:
    """Container for evaluation metrics."""

    loss: float
    acc: float


@torch.no_grad()
def evaluate(model, loader, device) -> Metrics:
    """
    Evaluate model on a dataset.

    Args:
        model: The neural network to evaluate
        loader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Metrics object with loss and accuracy
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)

    return Metrics(loss=total_loss / n, acc=correct / n)
