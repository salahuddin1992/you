"""
kian.language.tokenizer
=======================
A byte-level Byte-Pair-Encoding (BPE) tokenizer, trained from scratch.

Why byte-level
--------------
Kian must read and write *any* language — Arabic, English, code, emoji —
without a hand-built vocabulary per script. Every string is first encoded to
its raw UTF-8 bytes (0-255). BPE then learns, purely from data, which byte
sequences recur and deserve their own token. Arabic letters, English words and
whitespace all fall out of the same statistical process. There are no
``unknown`` tokens: in the worst case a character falls back to its individual
bytes, so the tokenizer round-trips arbitrary text losslessly.

This mirrors the approach used by GPT-2/GPT-4, implemented here in a few
hundred readable lines with no external tokenizer dependency.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Special tokens live *above* the 256 byte ids so they can never collide with
# real data. Order here defines their ids.
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<kian>", "<sep>"]
# Splits text around literal special tokens (keeping them) for encode().
_SPECIAL_SPLIT_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")


class Tokenizer:
    """Byte-level BPE with a trainable merge table."""

    def __init__(self) -> None:
        # merges: ordered list of (a, b) byte/token-id pairs that fuse into a
        # new id. Rank = position in this list (lower = merged earlier).
        self.merges: List[Tuple[int, int]] = []
        self.ranks: Dict[Tuple[int, int], int] = {}
        # vocab: id -> bytes, for decoding.
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: Dict[str, int] = {}
        self._install_special_tokens()

    # ------------------------------------------------------------------ #
    # Vocabulary bookkeeping
    # ------------------------------------------------------------------ #
    def _install_special_tokens(self) -> None:
        base = 256 + len(self.merges)
        self.special = {tok: base + i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok, tid in self.special.items():
            self.vocab[tid] = tok.encode("utf-8")

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(SPECIAL_TOKENS)

    def _reindex_specials(self) -> None:
        """Rebuild the id->bytes vocab canonically, then place specials on top.

        The byte ids (0-255) and the learned merge ids are the ground truth;
        special-token ids always sit immediately above them. Rebuilding from
        ``self.merges`` rather than mutating the existing dict avoids a subtle
        collision: the placeholder special ids assigned at ``__init__`` (when
        no merges exist yet) occupy the 256+ range that merges later claim.
        """
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, pair in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self._install_special_tokens()

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, text: str, vocab_size: int = 1024, verbose: bool = False) -> "Tokenizer":
        """Learn merges from ``text`` until the target vocab size is reached."""
        assert vocab_size >= 256 + len(SPECIAL_TOKENS), "vocab_size too small"
        num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)

        ids = list(text.encode("utf-8"))
        self.merges = []
        self.ranks = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for m in range(num_merges):
            pairs = self._count_pairs(ids)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:  # nothing recurs any more
                break
            new_id = 256 + m
            ids = self._merge(ids, best, new_id)
            self.merges.append(best)
            self.ranks[best] = m
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            if verbose and (m % 50 == 0 or m == num_merges - 1):
                print(f"  merge {m + 1}/{num_merges}: {best} -> {new_id} "
                      f"(count {pairs[best]}, seq len {len(ids)})")

        self._reindex_specials()
        return self

    @staticmethod
    def _count_pairs(ids: List[int]) -> Counter:
        pairs: Counter = Counter()
        for a, b in zip(ids, ids[1:]):
            pairs[(a, b)] += 1
        return pairs

    @staticmethod
    def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out: List[int] = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    def _bpe(self, ids: List[int]) -> List[int]:
        """Greedily apply the lowest-rank merge available, repeatedly."""
        if len(ids) < 2:
            return ids
        while True:
            pairs = set(zip(ids, ids[1:]))
            candidate = min(
                (p for p in pairs if p in self.ranks),
                key=lambda p: self.ranks[p],
                default=None,
            )
            if candidate is None:
                break
            ids = self._merge(ids, candidate, 256 + self.ranks[candidate])
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to ids. Literal special-token strings (``<user>``,
        ``<kian>``, ``<eos>`` …) in the text map to their special ids, so a
        corpus written in the chat format trains the same ids the chat path
        conditions on; everything else goes through byte-level BPE."""
        ids: List[int] = []
        for piece in _SPECIAL_SPLIT_RE.split(text):
            if not piece:
                continue
            if piece in self.special:
                ids.append(self.special[piece])
            else:
                ids.extend(self._bpe(list(piece.encode("utf-8"))))
        if add_bos:
            ids = [self.special["<bos>"]] + ids
        if add_eos:
            ids = ids + [self.special["<eos>"]]
        return ids

    def encode_special(self, token: str) -> int:
        return self.special[token]

    def decode(self, ids: List[int]) -> str:
        parts = bytearray()
        specials_by_id = {v: k for k, v in self.special.items()}
        for i in ids:
            if i in specials_by_id:
                # Render special tokens literally so transcripts stay readable.
                parts.extend(specials_by_id[i].encode("utf-8"))
            else:
                parts.extend(self.vocab.get(i, b""))
        return parts.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "merges": [list(m) for m in self.merges],
            "special_tokens": SPECIAL_TOKENS,
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.ranks = {pair: i for i, pair in enumerate(tok.merges)}
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for i, pair in enumerate(tok.merges):
            tok.vocab[256 + i] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        tok._reindex_specials()
        return tok

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"
