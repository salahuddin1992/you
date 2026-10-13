"""
kian.language.data
==================
Turn a text corpus into training batches for the Transformer.

The corpus is tokenised once into a flat ``int32`` stream. A batch is formed by
sampling random windows of ``block_size + 1`` tokens; the first ``block_size``
are the input and the same window shifted by one is the target. This is the
standard next-token-prediction setup — the model is asked, at every position,
to predict the token that actually followed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .tokenizer import Tokenizer


class TextDataset:
    def __init__(self, token_ids: np.ndarray, block_size: int) -> None:
        self.ids = np.asarray(token_ids, dtype=np.int64)
        self.block_size = block_size
        if len(self.ids) <= block_size + 1:
            # Pad short corpora so at least one window exists.
            reps = int(np.ceil((block_size + 2) / max(1, len(self.ids))))
            self.ids = np.tile(self.ids, reps)

    @classmethod
    def from_text(cls, text: str, tokenizer: Tokenizer, block_size: int) -> "TextDataset":
        return cls(np.asarray(tokenizer.encode(text), dtype=np.int64), block_size)

    @classmethod
    def from_files(cls, paths: List[str | Path], tokenizer: Tokenizer, block_size: int) -> "TextDataset":
        chunks: List[str] = []
        for p in paths:
            p = Path(p)
            if p.is_file():
                chunks.append(p.read_text(encoding="utf-8", errors="replace"))
        text = "\n".join(chunks) if chunks else " "
        return cls.from_text(text, tokenizer, block_size)

    def __len__(self) -> int:
        # windows start at s in [0, len - block_size - 1] inclusive
        return max(1, len(self.ids) - self.block_size)

    def batch(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        # exclusive upper bound: the last valid start is len - block_size - 1,
        # so the final corpus token can appear as a target
        hi = len(self.ids) - self.block_size
        starts = rng.integers(0, max(1, hi), size=batch_size)
        x = np.stack([self.ids[s : s + self.block_size] for s in starts])
        y = np.stack([self.ids[s + 1 : s + 1 + self.block_size] for s in starts])
        return x, y

    def split(self, val_fraction: float = 0.1) -> Tuple["TextDataset", "TextDataset"]:
        n = len(self.ids)
        cut = int(n * (1.0 - val_fraction))
        return (
            TextDataset(self.ids[:cut], self.block_size),
            TextDataset(self.ids[cut:], self.block_size),
        )
