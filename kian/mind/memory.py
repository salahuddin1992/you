"""
kian.mind.memory
===============
Persistent memory — what makes Kian continuous across runs rather than a
stateless function call.

Two stores:

* **Episodic** — a time-ordered log of experiences (things said, seen, learned).
* **Semantic** — distilled facts the being has chosen to keep.

Recall is by meaning, not keyword: every memory carries a vector produced by the
language model's own embeddings, and :meth:`recall` returns the memories whose
vectors are closest (cosine) to a query. As the model trains, its embeddings
sharpen, so memory recall improves alongside the rest of the being.

Everything is persisted to disk (``memory.jsonl`` + ``vectors.npz``) so the
being wakes up remembering yesterday.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

Embedder = Callable[[str], np.ndarray]


class MemoryItem:
    __slots__ = ("text", "kind", "ts", "importance", "vector")

    def __init__(self, text: str, kind: str, ts: float, importance: float,
                 vector: Optional[np.ndarray] = None) -> None:
        self.text = text
        self.kind = kind            # "episodic" | "semantic" | "perception"
        self.ts = ts
        self.importance = importance
        self.vector = vector

    def to_dict(self) -> dict:
        return {"text": self.text, "kind": self.kind, "ts": self.ts,
                "importance": self.importance}


class Memory:
    def __init__(self, embedder: Optional[Embedder] = None, dim: int = 0) -> None:
        self.items: List[MemoryItem] = []
        self.embedder = embedder
        self.dim = dim

    # ------------------------------------------------------------------ #
    def set_embedder(self, embedder: Embedder, dim: int) -> None:
        self.embedder = embedder
        self.dim = dim
        # (Re)embed anything stored before the model existed.
        for it in self.items:
            if it.vector is None or it.vector.shape[0] != dim:
                it.vector = self._embed(it.text)

    def _embed(self, text: str) -> np.ndarray:
        if self.embedder is None:
            # Fall back to a hashing bag-of-bytes vector so recall still works
            # before a model is attached.
            v = np.zeros(max(1, self.dim or 64), dtype=np.float32)
            for b in text.encode("utf-8"):
                v[b % v.shape[0]] += 1.0
        else:
            v = np.asarray(self.embedder(text), dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # ------------------------------------------------------------------ #
    def remember(self, text: str, kind: str = "episodic", importance: float = 1.0) -> MemoryItem:
        text = text.strip()
        if not text:
            return MemoryItem("", kind, time.time(), 0.0)
        item = MemoryItem(text, kind, time.time(), float(importance), self._embed(text))
        self.items.append(item)
        return item

    def recall(self, query: str, k: int = 5, kinds: Optional[List[str]] = None) -> List[MemoryItem]:
        pool = [it for it in self.items if it.vector is not None and
                (kinds is None or it.kind in kinds)]
        if not pool:
            return []
        q = self._embed(query)
        mat = np.stack([it.vector for it in pool])           # (N, dim)
        sims = mat @ q                                        # cosine (unit vectors)
        # Blend similarity with a mild recency + importance prior.
        now = time.time()
        recency = np.array([1.0 / (1.0 + (now - it.ts) / 86400.0) for it in pool])
        imp = np.array([it.importance for it in pool])
        score = sims + 0.1 * recency + 0.05 * imp
        order = np.argsort(-score)[:k]
        return [pool[i] for i in order]

    def recent(self, k: int = 10) -> List[MemoryItem]:
        return self.items[-k:]

    # ------------------------------------------------------------------ #
    def consolidate(self, max_items: int = 5000) -> int:
        """Nightly cleanup: drop the least useful memories if over capacity.

        Usefulness = importance weighted by recency. Semantic facts are kept
        preferentially over raw episodic chatter.
        """
        if len(self.items) <= max_items:
            return 0
        now = time.time()

        def keepscore(it: MemoryItem) -> float:
            rec = 1.0 / (1.0 + (now - it.ts) / 86400.0)
            bonus = 1.5 if it.kind == "semantic" else 1.0
            return it.importance * rec * bonus

        self.items.sort(key=keepscore, reverse=True)
        dropped = len(self.items) - max_items
        self.items = self.items[:max_items]
        # restore chronological order for episodic coherence
        self.items.sort(key=lambda it: it.ts)
        return dropped

    def __len__(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------------ #
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "memory.jsonl", "w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
        vecs = [it.vector for it in self.items if it.vector is not None]
        if vecs:
            np.savez_compressed(d / "vectors.npz", vectors=np.stack(vecs),
                                index=np.array([i for i, it in enumerate(self.items)
                                                if it.vector is not None]))

    @classmethod
    def load(cls, directory: str | Path, embedder: Optional[Embedder] = None, dim: int = 0) -> "Memory":
        d = Path(directory)
        mem = cls(embedder, dim)
        jpath = d / "memory.jsonl"
        if not jpath.exists():
            return mem
        rows = [json.loads(l) for l in jpath.read_text(encoding="utf-8").splitlines() if l.strip()]
        vectors, vindex = None, None
        vpath = d / "vectors.npz"
        if vpath.exists():
            npz = np.load(vpath)
            vectors, vindex = npz["vectors"], npz["index"]
        vmap = {int(i): vectors[j] for j, i in enumerate(vindex)} if vectors is not None else {}
        for i, r in enumerate(rows):
            mem.items.append(MemoryItem(r["text"], r.get("kind", "episodic"),
                                        r.get("ts", time.time()), r.get("importance", 1.0),
                                        vmap.get(i)))
        return mem
