"""
Pretraining utilities for joint ParityMNIST + FashionMNIST training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.parity_mnist import correct_labels_for_parity


def pretrain_joint(
    model: nn.Module,
    parity_loader: DataLoader,
    fashion_loader: DataLoader,
    config,
    device: str = "cuda"
):
    """
    Joint pretraining on ParityMNIST and FashionMNIST.

    During pretraining:
    - FashionMNIST: standard cross-entropy on true labels
    - ParityMNIST: random correct digit label for each sample
      (uniform over correct parity set)

    This matches the paper's idea that pretraining puts probability mass
    on multiple correct labels.

    Args:
        model: The MLP model to train
        parity_loader: DataLoader for ParityMNIST training data
        fashion_loader: DataLoader for FashionMNIST training data
        config: TrainConfig object
        device: Device to train on
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)

    parity_iter = iter(parity_loader)
    fashion_iter = iter(fashion_loader)

    pbar = tqdm(range(config.pretrain_steps), desc="Pretrain")

    for step in pbar:
        # Get ParityMNIST batch
        try:
            x_p, parity, _d = next(parity_iter)
        except StopIteration:
            parity_iter = iter(parity_loader)
            x_p, parity, _d = next(parity_iter)

        # Get FashionMNIST batch
        try:
            x_f, y_f = next(fashion_iter)
        except StopIteration:
            fashion_iter = iter(fashion_loader)
            x_f, y_f = next(fashion_iter)

        x_p, parity = x_p.to(device), parity.to(device)
        x_f, y_f = x_f.to(device), y_f.to(device)

        # ParityMNIST: sample labels uniformly among correct digits
        targets_p = torch.empty(len(x_p), dtype=torch.long, device=device)
        for i in range(len(x_p)):
            cl = correct_labels_for_parity(int(parity[i].item())).to(device)
            targets_p[i] = cl[torch.randint(0, len(cl), (1,), device=device)]

        logits_p = model(x_p, parity)
        loss_p = F.cross_entropy(logits_p, targets_p)

        # FashionMNIST task (parity bit fixed to 0)
        logits_f = model(x_f, torch.zeros(len(x_f), device=device))
        loss_f = F.cross_entropy(logits_f, y_f)

        # Combined loss
        loss = loss_p + loss_f

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Update progress bar
        if step % 50 == 0:
            pbar.set_postfix({
                'loss_p': f'{loss_p.item():.3f}',
                'loss_f': f'{loss_f.item():.3f}'
            })
