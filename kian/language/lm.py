"""
kian.language.lm
================
``LanguageModel`` — the high-level language faculty that binds a
:class:`~kian.language.tokenizer.Tokenizer` to a
:class:`~kian.language.model.GPT`, and exposes text-in / text-out generation
plus a single-directory checkpoint format.

A checkpoint directory contains:
* ``tokenizer.json`` — the learned byte-level merges
* ``config.json``    — the model hyper-parameters
* ``weights.npz``    — every parameter array, in ``parameters()`` order
* ``meta.json``      — training provenance (steps seen, developmental stage)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .model import GPT, GPTConfig
from .tokenizer import Tokenizer


class LanguageModel:
    def __init__(self, tokenizer: Tokenizer, model: GPT, meta: Optional[dict] = None) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.meta = meta or {"steps": 0, "stage": "newborn", "tokens_seen": 0}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def new(cls, tokenizer: Tokenizer, **cfg_overrides) -> "LanguageModel":
        cfg = GPTConfig(vocab_size=tokenizer.vocab_size, **cfg_overrides)
        return cls(tokenizer, GPT(cfg))

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 120,
        temperature: float = 0.9,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.92,
        min_p: Optional[float] = 0.02,
        repetition_penalty: float = 1.3,
        seed: Optional[int] = None,
    ) -> str:
        rng = np.random.default_rng(seed)
        ids = self.tokenizer.encode(prompt) if prompt else [self.tokenizer.encode_special("<bos>")]
        ids = np.array([ids], dtype=np.int64)
        out = self.model.generate(
            ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty, rng=rng
        )
        new_ids = out[0, len(ids[0]) :].tolist()
        return self.tokenizer.decode(new_ids)

    def reply(self, user_text: str, **gen) -> str:
        """Chat-style turn using the <user>/<kian> role tokens."""
        u = self.tokenizer.encode_special("<user>")
        k = self.tokenizer.encode_special("<kian>")
        ids = [u] + self.tokenizer.encode(user_text) + [k]
        ids = np.array([ids], dtype=np.int64)
        rng = np.random.default_rng()
        out = self.model.generate(ids, max_new_tokens=gen.get("max_new_tokens", 120),
                                  temperature=gen.get("temperature", 0.9),
                                  top_k=gen.get("top_k", 40), top_p=gen.get("top_p", 0.92),
                                  min_p=gen.get("min_p", 0.02),
                                  repetition_penalty=gen.get("repetition_penalty", 1.3), rng=rng)
        text = self.tokenizer.decode(out[0, len(ids[0]):].tolist())
        # cut at the next role boundary if the model emits one
        for boundary in ("<user>", "<eos>", "<kian>"):
            if boundary in text:
                text = text.split(boundary)[0]
        return text.strip()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(d / "tokenizer.json")
        (d / "config.json").write_text(json.dumps(asdict(self.model.cfg)), encoding="utf-8")
        params = {f"p{i}": p.data for i, p in enumerate(self.model.parameters())}
        np.savez(d / "weights.npz", **params)
        (d / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "LanguageModel":
        d = Path(directory)
        tokenizer = Tokenizer.load(d / "tokenizer.json")
        cfg = GPTConfig(**json.loads((d / "config.json").read_text(encoding="utf-8")))
        model = GPT(cfg)
        npz = np.load(d / "weights.npz")
        state = [npz[f"p{i}"] for i in range(len(npz.files))]
        model.load_state_dict(state)
        meta = {}
        meta_path = d / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(tokenizer, model, meta)

    def __repr__(self) -> str:
        return (f"LanguageModel(vocab={self.tokenizer.vocab_size}, "
                f"params={self.model.num_parameters():,}, stage={self.meta.get('stage')})")
