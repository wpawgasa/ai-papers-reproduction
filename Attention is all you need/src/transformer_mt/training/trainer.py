"""
Training loop for the Transformer (Section 5).

Key components:
- Adam optimizer: beta1=0.9, beta2=0.98, eps=1e-9 (Section 5.3)
- Warmup LR schedule: Equation (5) (Section 5.3)
- Label smoothing: eps_ls = 0.1 (Section 5.4)
- Checkpoint saving at regular intervals (Section 6.1)
- Periodic validation with BLEU evaluation
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class TransformerLRScheduler:
    """
    Noam learning rate schedule (Section 5.3, Equation 5).

    lrate = d_model^{-0.5} * min(step_num^{-0.5}, step_num * warmup_steps^{-1.5})

    Linear warmup for first warmup_steps, then inverse sqrt decay.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _compute_lr(self) -> float:
        return self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5),
        )


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing (Section 5.4).

    eps_ls = 0.1: distributes 0.1 of probability mass uniformly across
    all tokens, keeping 0.9 on the target token.

    "This hurts perplexity, as the model learns to be more unsure,
    but improves accuracy and BLEU score."
    """

    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch * seq_len, vocab_size)
            target: (batch * seq_len,)
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        smooth_target = torch.full_like(
            log_probs, self.smoothing / (self.vocab_size - 2)
        )
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_target[:, self.padding_idx] = 0

        mask = target != self.padding_idx
        smooth_target[~mask] = 0

        return self.criterion(log_probs, smooth_target)


@dataclass
class TrainHistory:
    """Stores metrics collected during training."""

    steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    val_steps: List[int] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_bleu: List[float] = field(default_factory=list)


def _cycle_dataloader(dataloader: DataLoader) -> Iterator:
    """Yield batches from a DataLoader, cycling through epochs indefinitely."""
    epoch = 0
    while True:
        sampler = dataloader.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in dataloader:
            yield batch
        epoch += 1


def _to_device(
    batch: Tuple[torch.Tensor, ...], device: torch.device
) -> Tuple[torch.Tensor, ...]:
    """Move a batch of tensors to the target device."""
    return tuple(t.to(device, non_blocking=True) for t in batch)


def train_step(
    model: nn.Module,
    src: torch.Tensor,
    tgt: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
    criterion: LabelSmoothingLoss,
    optimizer: optim.Optimizer,
    scheduler: TransformerLRScheduler,
    max_grad_norm: float = 0.0,
) -> float:
    """Single training step with optional gradient clipping."""
    model.train()
    optimizer.zero_grad()

    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    logits = model(src, tgt_input, src_mask, tgt_mask)

    loss = criterion(
        logits.contiguous().view(-1, logits.size(-1)),
        tgt_output.contiguous().view(-1),
    )

    loss.backward()

    if max_grad_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    scheduler.step()

    return loss.item()


@torch.no_grad()
def validate(
    model: nn.Module,
    val_dataloader: DataLoader,
    criterion: LabelSmoothingLoss,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_dataloader:
        src, tgt, src_mask, tgt_mask = _to_device(batch, device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(
            logits.contiguous().view(-1, logits.size(-1)),
            tgt_output.contiguous().view(-1),
        )
        total_loss += loss.item()
        n_batches += 1

        if max_batches is not None and n_batches >= max_batches:
            break

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def compute_bleu(
    model: nn.Module,
    val_dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_samples: int = 200,
) -> float:
    """
    Compute BLEU score on validation data using greedy decoding.

    Uses sacrebleu for standard BLEU computation matching the paper's
    evaluation methodology (Section 6.1).
    """
    import sacrebleu

    model.eval()
    hypotheses: List[str] = []
    references: List[str] = []
    n_samples = 0

    for batch in val_dataloader:
        src, tgt, src_mask, _ = _to_device(batch, device)

        for i in range(src.size(0)):
            if n_samples >= max_samples:
                break

            src_i = src[i : i + 1]
            src_mask_i = src_mask[i : i + 1]
            encoder_output = model.encode(src_i, src_mask_i)

            # Greedy decode
            bos = tokenizer.bos_idx
            eos = tokenizer.eos_idx
            dec_input = torch.full(
                (1, 1), bos, dtype=torch.long, device=device
            )

            for _ in range(tgt.size(1) + 50):
                tgt_mask = model.make_causal_mask(dec_input.size(1)).to(device)
                dec_out = model.decode(
                    dec_input, encoder_output, src_mask_i, tgt_mask
                )
                logits = model.output_projection(dec_out[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)
                dec_input = torch.cat([dec_input, next_token], dim=1)
                if next_token.item() == eos:
                    break

            pred_ids = dec_input.squeeze(0).tolist()
            ref_ids = tgt[i].tolist()

            hypotheses.append(tokenizer.decode(pred_ids))
            references.append(tokenizer.decode(ref_ids))
            n_samples += 1

        if n_samples >= max_samples:
            break

    if not hypotheses:
        return 0.0

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: TransformerLRScheduler,
    step: int,
    loss: float,
    checkpoint_dir: str,
) -> str:
    """Save a training checkpoint and return the file path."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_step_num": scheduler.step_num,
            "loss": loss,
        },
        path,
    )
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[TransformerLRScheduler] = None,
) -> int:
    """Load a checkpoint and return the step number."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.step_num = checkpoint["scheduler_step_num"]
    return checkpoint["step"]


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    vocab_size: int,
    d_model: int = 512,
    warmup_steps: int = 4000,
    total_steps: int = 100000,
    padding_idx: int = 0,
    label_smoothing: float = 0.1,
    log_interval: int = 100,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "./checkpoints",
    checkpoint_interval: int = 1000,
    val_dataloader: Optional[DataLoader] = None,
    val_interval: int = 5000,
    val_max_batches: Optional[int] = 50,
    tokenizer=None,
    bleu_samples: int = 200,
    max_grad_norm: float = 1.0,
    resume_from: Optional[str] = None,
) -> TrainHistory:
    """
    Full training loop (Section 5).

    Hardware: 8 NVIDIA P100 GPUs
    Base model: ~0.4s/step, 100K steps, ~12 hours
    Big model: ~1.0s/step, 300K steps, ~3.5 days

    Args:
        model: Transformer model.
        train_dataloader: Training DataLoader (will be cycled automatically).
        vocab_size: Vocabulary size for label smoothing.
        d_model: Model dimension for LR schedule.
        warmup_steps: LR warmup steps (Equation 5).
        total_steps: Total training steps.
        padding_idx: Padding token index.
        label_smoothing: Label smoothing epsilon (Section 5.4).
        log_interval: Steps between log messages.
        device: Target device. Defaults to CUDA if available.
        checkpoint_dir: Directory for saving checkpoints.
        checkpoint_interval: Steps between checkpoint saves.
        val_dataloader: Optional validation DataLoader.
        val_interval: Steps between validation runs.
        val_max_batches: Max batches per validation (None = full pass).
        tokenizer: BPETokenizer for BLEU computation (optional).
        bleu_samples: Number of samples for BLEU evaluation.
        max_grad_norm: Max gradient norm for clipping (0 = disabled).
        resume_from: Path to checkpoint to resume from.

    Returns:
        TrainHistory with logged metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = TransformerLRScheduler(optimizer, d_model, warmup_steps)
    criterion = LabelSmoothingLoss(vocab_size, padding_idx, label_smoothing).to(device)
    history = TrainHistory()

    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(resume_from, model, optimizer, scheduler)
        print(f"Resumed from checkpoint at step {start_step}")

    train_iter = _cycle_dataloader(train_dataloader)
    running_loss = 0.0
    log_steps = 0
    t0 = time.time()

    for step in range(start_step + 1, total_steps + 1):
        src, tgt, src_mask, tgt_mask = _to_device(next(train_iter), device)

        loss = train_step(
            model,
            src,
            tgt,
            src_mask,
            tgt_mask,
            criterion,
            optimizer,
            scheduler,
            max_grad_norm,
        )

        running_loss += loss
        log_steps += 1

        if step % log_interval == 0:
            avg_loss = running_loss / log_steps
            elapsed = time.time() - t0
            steps_per_sec = log_steps / elapsed

            history.steps.append(step)
            history.losses.append(avg_loss)
            history.learning_rates.append(scheduler._compute_lr())

            print(
                f"Step {step}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {scheduler._compute_lr():.6e} | "
                f"{steps_per_sec:.2f} steps/s"
            )

            running_loss = 0.0
            log_steps = 0
            t0 = time.time()

        if checkpoint_interval > 0 and step % checkpoint_interval == 0:
            path = save_checkpoint(
                model, optimizer, scheduler, step, loss, checkpoint_dir
            )
            print(f"  Checkpoint saved: {path}")

        if (
            val_dataloader is not None
            and val_interval > 0
            and step % val_interval == 0
        ):
            val_loss = validate(
                model, val_dataloader, criterion, device, val_max_batches
            )
            history.val_steps.append(step)
            history.val_losses.append(val_loss)

            msg = f"  Validation loss: {val_loss:.4f}"

            if tokenizer is not None:
                bleu = compute_bleu(
                    model, val_dataloader, tokenizer, device, bleu_samples
                )
                history.val_bleu.append(bleu)
                msg += f" | BLEU: {bleu:.2f}"

            print(msg)

    # Final checkpoint
    if checkpoint_interval > 0:
        path = save_checkpoint(
            model, optimizer, scheduler, total_steps, loss, checkpoint_dir
        )
        print(f"Final checkpoint saved: {path}")

    return history
