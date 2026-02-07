"""
GLUE dataset loading for FNet fine-tuning.
Reference: Section 4.1 — GLUE Benchmark (Wang et al., 2018)
"""

import torch
from torch.utils.data import Dataset
from typing import Optional


# Task → (sentence_key_1, sentence_key_2_or_None)
TASK_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp":  ("question1", "question2"),
    "rte":  ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

# Task → number of classification labels (STS-B is regression = 1)
TASK_NUM_LABELS = {
    "cola": 2, "mnli": 3, "mrpc": 2, "qnli": 2,
    "qqp": 2, "rte": 2, "sst2": 2, "stsb": 1,
}


class GLUEDataset(Dataset):
    """Wraps HuggingFace `datasets` GLUE splits for FNet."""

    def __init__(self, task: str, split: str, tokenizer, max_length: int = 512):
        from datasets import load_dataset
        self.task = task
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.key1, self.key2 = TASK_KEYS[task]
        self.is_regression = (task == "stsb")
        self.dataset = load_dataset("glue", task, split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text_pair = (item[self.key1], item[self.key2]) if self.key2 else (item[self.key1],)
        encoding = self.tokenizer(
            *text_pair,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label_dtype = torch.float if self.is_regression else torch.long
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros_like(encoding["input_ids"])
            ).squeeze(0),
            "label": torch.tensor(item["label"], dtype=label_dtype),
        }
