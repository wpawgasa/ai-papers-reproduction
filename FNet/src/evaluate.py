"""
Evaluation metrics for GLUE tasks.
Reference: Table 2 — Accuracy, F1, Spearman correlation.
"""

import numpy as np
from typing import Dict


def compute_metrics(task: str, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute task-appropriate metric(s)."""
    if task == "stsb":
        from scipy.stats import pearsonr, spearmanr
        return {
            "pearson": float(pearsonr(preds, labels)[0]),
            "spearman": float(spearmanr(preds, labels)[0]),
        }
    elif task in ("mrpc", "qqp"):
        acc = float((preds == labels).mean())
        # Simple F1 for binary classification
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {"accuracy": acc, "f1": f1, "acc_f1_mean": (acc + f1) / 2}
    else:
        return {"accuracy": float((preds == labels).mean())}
