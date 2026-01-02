"""
Evaluation metrics for measuring task performance and forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def fashion_accuracy(model: nn.Module, loader: DataLoader, device: str = "cuda") -> float:
    """
    Compute classification accuracy on FashionMNIST.

    Args:
        model: The MLP model
        loader: DataLoader for FashionMNIST test data
        device: Device to evaluate on

    Returns:
        Accuracy as a float in [0, 1]
    """
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Parity bit not used for FashionMNIST; keep as 0
        parity_bit = torch.zeros(len(x), device=device)

        logits = model(x, parity_bit)
        pred = logits.argmax(dim=-1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / max(1, total)


@torch.no_grad()
def parity_success(model: nn.Module, loader: DataLoader, device: str = "cuda") -> float:
    """
    Compute parity success rate on ParityMNIST.

    Success means the predicted digit has the correct parity
    (even/odd) relative to the input.

    Args:
        model: The MLP model
        loader: DataLoader for ParityMNIST test data
        device: Device to evaluate on

    Returns:
        Success rate as a float in [0, 1]
    """
    model.eval()
    succ = 0
    total = 0

    for x, parity, _digit in loader:
        x = x.to(device)
        parity = parity.to(device)

        logits = model(x, parity)
        pred = logits.argmax(dim=-1)

        # Success if predicted digit has correct parity
        succ += ((pred % 2) == parity).sum().item()
        total += parity.numel()

    return succ / max(1, total)


@torch.no_grad()
def kl_base_to_ft(
    base: nn.Module,
    ft: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    max_batches: int = 50
) -> float:
    """
    Compute average KL(base || fine-tuned) over new-task inputs.

    This measures how much the fine-tuned model's distribution has
    shifted from the base model on the new task.

    Args:
        base: Base (pretrained) model
        ft: Fine-tuned model
        loader: DataLoader for ParityMNIST test data
        device: Device to evaluate on
        max_batches: Maximum number of batches to evaluate (for speed)

    Returns:
        Average KL divergence
    """
    base.eval()
    ft.eval()
    kls = []

    for i, (x, parity, _d) in enumerate(loader):
        if i >= max_batches:
            break

        x, parity = x.to(device), parity.to(device)

        # Compute probability distributions
        p0 = F.softmax(base(x, parity), dim=-1)
        p1 = F.softmax(ft(x, parity), dim=-1)

        # KL(p0 || p1) = sum_y p0(y) * log(p0(y) / p1(y))
        kl = (p0 * (p0.clamp_min(1e-12).log() - p1.clamp_min(1e-12).log())).sum(dim=-1)
        kls.append(kl.mean().item())

    return float(sum(kls) / max(1, len(kls)))
