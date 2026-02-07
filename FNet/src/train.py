"""
Training loop for FNet GLUE fine-tuning.
Reference: Section 4.1, Appendix A.1

Paper fine-tuning protocol:
    - 3 trials (Base) / 6 trials (Large) per learning rate
    - Best result across all trials reported
    - No early stopping
    - AdamW optimizer, linear LR decay
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import numpy as np
from typing import Dict

from .model import FNetForSequenceClassification, FNET_CONFIGS
from .data import GLUEDataset, TASK_NUM_LABELS
from .evaluate import compute_metrics


def train_epoch(model, dataloader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Train", leave=False):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            labels=batch["label"].to(device),
        )
        loss = outputs["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_model(model, dataloader, device, task) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
        )
        logits = outputs["logits"]
        if task == "stsb":
            preds = logits.squeeze(-1).cpu()
        else:
            preds = torch.argmax(logits, dim=-1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].tolist())
    return compute_metrics(task, np.array(all_preds), np.array(all_labels))


def main():
    parser = argparse.ArgumentParser(description="FNet GLUE Fine-tuning")
    parser.add_argument("--task", type=str, default="sst2", choices=TASK_NUM_LABELS.keys())
    parser.add_argument("--config", type=str, default="base", choices=FNET_CONFIGS.keys())
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    val_split = "validation_matched" if args.task == "mnli" else "validation"
    train_ds = GLUEDataset(args.task, "train", tokenizer, args.max_length)
    val_ds = GLUEDataset(args.task, val_split, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = FNetForSequenceClassification(
        num_labels=TASK_NUM_LABELS[args.task], **FNET_CONFIGS[args.config]
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Task: {args.task} | Config: {args.config} | Params: {n_params/1e6:.1f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )

    best_metric = -float("inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        metrics = evaluate_model(model, val_loader, device, args.task)
        primary = metrics.get("accuracy", metrics.get("spearman", 0))
        if primary > best_metric:
            best_metric = primary
            torch.save(model.state_dict(), f"fnet_{args.task}_best.pt")
        print(f"Epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  {metrics}")

    print(f"\nBest: {best_metric:.4f}")


if __name__ == "__main__":
    main()
