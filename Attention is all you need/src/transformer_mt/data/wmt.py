"""
Data pipeline for WMT14 EN-DE translation (Section 5.1).

- Dataset: WMT 2014 English-German (~4.5M sentence pairs)
- Tokenization: Byte-pair encoding (BPE) with ~37K shared vocabulary
- Batching: ~25K source + ~25K target tokens per batch
- Sentence pairs batched by approximate sequence length

Uses HuggingFace datasets for downloading and the tokenizers library for BPE.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .tokenizer import BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX, BPETokenizer


class TranslationDataset(Dataset):
    """
    Pre-tokenized translation dataset stored as paired token-ID lists.

    Constructed from raw HuggingFace dataset splits via ``from_hf_dataset``.
    """

    def __init__(
        self,
        src_ids: List[List[int]],
        tgt_ids: List[List[int]],
    ):
        assert len(src_ids) == len(tgt_ids)
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.src_ids[idx], self.tgt_ids[idx]

    @classmethod
    def from_hf_dataset(
        cls,
        hf_dataset,
        tokenizer: BPETokenizer,
        src_lang: str = "en",
        tgt_lang: str = "de",
        max_seq_len: int = 512,
        max_samples: Optional[int] = None,
    ) -> TranslationDataset:
        """
        Tokenize a HuggingFace dataset split into a TranslationDataset.

        Filters out pairs where either side exceeds max_seq_len (including
        BOS/EOS tokens).

        Args:
            hf_dataset: A HuggingFace dataset split with a "translation" column.
            tokenizer: Trained BPETokenizer instance.
            src_lang: Source language key (default "en").
            tgt_lang: Target language key (default "de").
            max_seq_len: Maximum sequence length after tokenization.
            max_samples: If set, only process this many samples.
        """
        src_ids: List[List[int]] = []
        tgt_ids: List[List[int]] = []

        dataset_iter = hf_dataset
        if max_samples is not None:
            dataset_iter = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

        for example in dataset_iter:
            translation = example["translation"]
            src_tokens = tokenizer.encode(translation[src_lang])
            tgt_tokens = tokenizer.encode(translation[tgt_lang])

            # Filter by length (Section 5.1 implies length filtering)
            if len(src_tokens) > max_seq_len or len(tgt_tokens) > max_seq_len:
                continue

            src_ids.append(src_tokens)
            tgt_ids.append(tgt_tokens)

        return cls(src_ids, tgt_ids)


# ---------------------------------------------------------------------------
# Mask creation
# ---------------------------------------------------------------------------

def create_masks(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pad_idx: int = PAD_IDX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create source padding mask and target causal+padding mask."""
    # Source padding mask: (batch, 1, 1, src_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # Target: combine padding mask with causal mask
    tgt_len = tgt.size(1)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
    tgt_causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device))
    tgt_causal_mask = tgt_causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, tgt_len)
    tgt_mask = tgt_pad_mask & (tgt_causal_mask.bool())

    return src_mask, tgt_mask


# ---------------------------------------------------------------------------
# Token-count batching (Section 5.1)
# ---------------------------------------------------------------------------

class TokenBatchSampler(Sampler):
    """
    Groups sentence pairs into batches targeting a fixed token budget.

    "Sentence pairs were batched together by approximate sequence length.
    Each training batch contained a set of sentence pairs containing
    approximately 25000 source tokens and 25000 target tokens."
    -- Section 5.1

    Sorts by source length for minimal padding, then fills batches until
    the token budget is reached.
    """

    def __init__(
        self,
        dataset: TranslationDataset,
        batch_tokens: int = 25000,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_tokens = batch_tokens
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Pre-compute lengths for sorting
        self.src_lengths = [len(s) for s in dataset.src_ids]

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        # Sort indices by source length (for minimal padding)
        indices = sorted(range(len(self.dataset)), key=lambda i: self.src_lengths[i])

        if self.shuffle:
            # Shuffle within windows of similar-length sentences
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            # Create pools of ~100 similar-length sentences, shuffle within each
            pool_size = 100
            shuffled = []
            for i in range(0, len(indices), pool_size):
                pool = indices[i : i + pool_size]
                perm = torch.randperm(len(pool), generator=g).tolist()
                shuffled.extend(pool[j] for j in perm)
            indices = shuffled

        # Build batches by token count
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_max_src = 0
        current_max_tgt = 0

        for idx in indices:
            src_len = len(self.dataset.src_ids[idx])
            tgt_len = len(self.dataset.tgt_ids[idx])

            new_max_src = max(current_max_src, src_len)
            new_max_tgt = max(current_max_tgt, tgt_len)
            new_batch_size = len(current_batch) + 1

            # Estimate total tokens with padding
            estimated_tokens = new_batch_size * (new_max_src + new_max_tgt)

            if estimated_tokens > self.batch_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [idx]
                current_max_src = src_len
                current_max_tgt = tgt_len
            else:
                current_batch.append(idx)
                current_max_src = new_max_src
                current_max_tgt = new_max_tgt

        if current_batch:
            batches.append(current_batch)

        # Shuffle batch order
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        yield from batches

    def __len__(self) -> int:
        # Approximate; exact count depends on length distribution
        total_tokens = sum(
            len(self.dataset.src_ids[i]) + len(self.dataset.tgt_ids[i])
            for i in range(len(self.dataset))
        )
        return max(1, total_tokens // self.batch_tokens)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[List[int], List[int]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a batch of (src_ids, tgt_ids) pairs into padded tensors + masks.

    Returns:
        src: (batch, src_len) padded source token IDs
        tgt: (batch, tgt_len) padded target token IDs
        src_mask: (batch, 1, 1, src_len)
        tgt_mask: (batch, 1, tgt_len-1, tgt_len-1)
    """
    src_list, tgt_list = zip(*batch)

    src_tensors = [torch.tensor(s, dtype=torch.long) for s in src_list]
    tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in tgt_list]

    src = torch.nn.utils.rnn.pad_sequence(
        src_tensors, batch_first=True, padding_value=PAD_IDX
    )
    tgt = torch.nn.utils.rnn.pad_sequence(
        tgt_tensors, batch_first=True, padding_value=PAD_IDX
    )

    # Masks are for tgt[:, :-1] (decoder input excludes last token)
    src_mask, tgt_mask = create_masks(src, tgt[:, :-1])

    return src, tgt, src_mask, tgt_mask


# ---------------------------------------------------------------------------
# Top-level factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_config,
    tokenizer: BPETokenizer,
    splits: Tuple[str, ...] = ("train", "validation"),
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> dict:
    """
    Build DataLoaders for WMT14 EN-DE from config.

    Downloads the dataset via HuggingFace datasets on first call.

    Args:
        data_config: DataConfig instance.
        tokenizer: Trained BPETokenizer.
        splits: Which splits to load.
        max_train_samples: Limit training data (for quick testing).
        max_val_samples: Limit validation data.

    Returns:
        Dict mapping split name to DataLoader.
    """
    from datasets import load_dataset

    raw = load_dataset("wmt14", "de-en", trust_remote_code=True)

    loaders = {}

    for split in splits:
        hf_split = raw.get(split)
        if hf_split is None:
            if split == "validation":
                hf_split = raw.get("validation")
            elif split == "test":
                hf_split = raw.get("test")
            if hf_split is None:
                continue

        max_samples = max_train_samples if split == "train" else max_val_samples

        dataset = TranslationDataset.from_hf_dataset(
            hf_split,
            tokenizer=tokenizer,
            max_seq_len=data_config.max_seq_len,
            max_samples=max_samples,
        )

        is_train = split == "train"
        sampler = TokenBatchSampler(
            dataset,
            batch_tokens=data_config.batch_tokens,
            shuffle=is_train,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
        )

    return loaders


def train_tokenizer(
    data_config,
    save_path: Optional[str | Path] = None,
    max_samples: int = 1_000_000,
) -> BPETokenizer:
    """
    Train a BPE tokenizer on WMT14 EN-DE training data.

    Uses a subset of training data for efficiency. Feeds both English and
    German sentences to learn a shared vocabulary.

    Args:
        data_config: DataConfig instance.
        save_path: Where to save the trained tokenizer. Defaults to
                   ``{data_config.data_dir}/tokenizer.json``.
        max_samples: Max sentence pairs to train on.

    Returns:
        Trained BPETokenizer.
    """
    from datasets import load_dataset

    if save_path is None:
        save_path = os.path.join(data_config.data_dir, "tokenizer.json")

    save_path = Path(save_path)

    # Return cached tokenizer if it exists
    if save_path.exists():
        print(f"Loading existing tokenizer from {save_path}")
        return BPETokenizer.load(save_path)

    print(f"Training BPE tokenizer (vocab_size={data_config.vocab_size})...")
    raw = load_dataset("wmt14", "de-en", split="train", trust_remote_code=True)

    def text_iterator():
        for i, example in enumerate(raw):
            if i >= max_samples:
                break
            translation = example["translation"]
            yield translation["en"]
            yield translation["de"]

    tokenizer = BPETokenizer.train(
        text_iterator=text_iterator(),
        vocab_size=data_config.vocab_size,
        save_path=save_path,
    )
    print(f"Tokenizer saved to {save_path} (vocab_size={tokenizer.vocab_size})")
    return tokenizer
