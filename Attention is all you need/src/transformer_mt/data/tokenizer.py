"""
BPE Tokenizer for WMT14 EN-DE (Section 5.1).

- Shared source-target BPE vocabulary (~37K merge operations)
- Special tokens: [PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3
- Trained using HuggingFace tokenizers library

"Sentences were encoded using byte-pair encoding, which has a shared
source-target vocabulary of about 37000 tokens." -- Section 5.1
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class BPETokenizer:
    """
    Shared BPE tokenizer for source and target languages.

    Wraps HuggingFace tokenizers for fast BPE encoding/decoding.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_idx(self) -> int:
        return PAD_IDX

    @property
    def bos_idx(self) -> int:
        return BOS_IDX

    @property
    def eos_idx(self) -> int:
        return EOS_IDX

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs, optionally wrapping with BOS/EOS."""
        ids = self._tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [BOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if skip_special_tokens:
            ids = [i for i in ids if i not in (PAD_IDX, BOS_IDX, EOS_IDX)]
        return self._tokenizer.decode(ids)

    def encode_batch(
        self, texts: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        """Encode a batch of texts."""
        encodings = self._tokenizer.encode_batch(texts)
        if add_special_tokens:
            return [[BOS_IDX] + enc.ids + [EOS_IDX] for enc in encodings]
        return [enc.ids for enc in encodings]

    def save(self, path: str | Path) -> None:
        """Save tokenizer to file."""
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a pretrained tokenizer from file."""
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)

    @classmethod
    def train(
        cls,
        text_iterator: Iterator[str],
        vocab_size: int = 37000,
        save_path: str | Path | None = None,
    ) -> BPETokenizer:
        """
        Train a new BPE tokenizer from an iterator of text strings.

        Args:
            text_iterator: Iterator yielding raw text strings (both src and tgt).
            vocab_size: Target vocabulary size (~37K per paper).
            save_path: If provided, save the trained tokenizer here.

        Returns:
            Trained BPETokenizer instance.
        """
        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2,
            show_progress=True,
        )

        tokenizer.train_from_iterator(text_iterator, trainer=trainer)

        instance = cls(tokenizer)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            instance.save(save_path)

        return instance
