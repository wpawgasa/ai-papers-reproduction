"""
Fine-tuning methods: SFT (fixed, oracle) and on-policy REINFORCE.
"""

import random
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.parity_mnist import correct_labels_for_parity


def finetune_sft_fixed(
    model: nn.Module,
    loader: DataLoader,
    config,
    device: str = "cuda",
    mapping: Literal["01", "random"] = "01"
):
    """
    Offline SFT on ParityMNIST using an arbitrary single-label mapping.

    Args:
        model: The MLP model to fine-tune
        loader: DataLoader for ParityMNIST training data
        config: TrainConfig object
        device: Device to train on
        mapping: Label mapping strategy
            - "01": even->0, odd->1 (intentionally collapses correct set)
            - "random": choose one fixed random even and odd label
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Determine label mapping
    if mapping == "01":
        even_label, odd_label = 0, 1
    else:
        even_label = random.choice([0, 2, 4, 6, 8])
        odd_label = random.choice([1, 3, 5, 7, 9])

    it = iter(loader)
    pbar = tqdm(range(config.finetune_steps), desc=f"Finetune SFT({mapping})")

    for step in pbar:
        try:
            x, parity, _d = next(it)
        except StopIteration:
            it = iter(loader)
            x, parity, _d = next(it)

        x, parity = x.to(device), parity.to(device)

        # Fixed label mapping
        y = torch.where(
            parity == 0,
            torch.tensor(even_label, device=device),
            torch.tensor(odd_label, device=device)
        )

        logits = model(x, parity)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})


@torch.no_grad()
def oracle_targets_from_base(
    base: nn.Module,
    x: torch.Tensor,
    parity: torch.Tensor
) -> torch.Tensor:
    """
    Sample from q*(y|x) ∝ p_base(y|x) restricted to correct labels.

    This is the minimum-KL correct distribution (I-projection) in the discrete case.

    Args:
        base: Base model to compute probabilities
        x: Input images
        parity: Parity labels

    Returns:
        Sampled target labels
    """
    base.eval()
    probs = F.softmax(base(x, parity), dim=-1)  # [B, 10]
    out = torch.empty(len(x), dtype=torch.long, device=x.device)

    for i in range(len(x)):
        # Get correct labels for this parity
        cl = correct_labels_for_parity(int(parity[i].item())).to(x.device)
        # Restrict probabilities to correct labels
        p = probs[i, cl]
        # Normalize
        p = p / p.sum()
        # Sample from normalized distribution
        out[i] = cl[torch.multinomial(p, 1)]

    return out


def finetune_sft_oracle(
    model: nn.Module,
    base_model: nn.Module,
    loader: DataLoader,
    config,
    device: str = "cuda"
):
    """
    Offline SFT using oracle supervision distribution q*.

    The oracle distribution minimizes KL to base subject to correctness.

    Args:
        model: The MLP model to fine-tune
        base_model: The base (pretrained) model for computing oracle targets
        loader: DataLoader for ParityMNIST training data
        config: TrainConfig object
        device: Device to train on
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    it = iter(loader)

    pbar = tqdm(range(config.finetune_steps), desc="Finetune SFT(oracle)")

    for step in pbar:
        try:
            x, parity, _d = next(it)
        except StopIteration:
            it = iter(loader)
            x, parity, _d = next(it)

        x, parity = x.to(device), parity.to(device)

        # Sample targets from oracle distribution
        y = oracle_targets_from_base(base_model, x, parity)

        logits = model(x, parity)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})


def finetune_reinforce(
    model: nn.Module,
    loader: DataLoader,
    config,
    device: str = "cuda"
):
    """
    On-policy REINFORCE with binary reward.

    Reward is 1 if sampled digit has correct parity, else 0.
    This approximates the paper's "1-0 Reinforce" idea.

    Args:
        model: The MLP model to fine-tune
        loader: DataLoader for ParityMNIST training data
        config: TrainConfig object
        device: Device to train on
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)

    baseline = 0.0
    beta = 0.95  # baseline smoothing factor

    it = iter(loader)
    pbar = tqdm(range(config.finetune_steps), desc="Finetune REINFORCE(on-policy)")

    for step in pbar:
        try:
            x, parity, _d = next(it)
        except StopIteration:
            it = iter(loader)
            x, parity, _d = next(it)

        x, parity = x.to(device), parity.to(device)

        # Get model's current distribution
        logits = model(x, parity)
        dist = torch.distributions.Categorical(logits=logits)

        # Sample action (digit label)
        a = dist.sample()
        logp = dist.log_prob(a)

        # Compute reward: 1 if correct parity, 0 otherwise
        reward = ((a % 2) == parity).float()

        # Update baseline
        r_mean = reward.mean().item()
        baseline = beta * baseline + (1 - beta) * r_mean

        # Compute advantage
        adv = reward - baseline

        # REINFORCE loss: -E[advantage * log_prob]
        loss = -(adv.detach() * logp).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'reward': f'{r_mean:.3f}'
            })
