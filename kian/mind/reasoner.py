"""
kian.mind.reasoner
=================
The ``Mind`` — Kian's thinking loop.

Each interaction runs the cognitive cycle:

    perceive  ->  recall  ->  think  ->  reflect

1. **perceive**  take in the input and log it as a perception.
2. **recall**    retrieve memories semantically related to the input.
3. **think**     build a working-memory context (recalled memories + recent
                 dialogue + the input) and let the language model continue it.
4. **reflect**   store the exchange so it can be recalled later.

The mind also exposes a sentence :meth:`embed` function that it hands to memory,
so recall is grounded in the language model's own representation of meaning.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..language.lm import LanguageModel
from .memory import Memory


class Mind:
    def __init__(self, lm: LanguageModel, memory: Optional[Memory] = None) -> None:
        self.lm = lm
        self.memory = memory or Memory()
        self.memory.set_embedder(self.embed, dim=lm.model.cfg.n_embd)
        self.working: List[str] = []   # short-term dialogue buffer

    # ------------------------------------------------------------------ #
    def embed(self, text: str) -> np.ndarray:
        """Mean token-embedding sentence vector (detached, numpy)."""
        ids = self.lm.tokenizer.encode(text)
        if not ids:
            return np.zeros(self.lm.model.cfg.n_embd, dtype=np.float32)
        W = self.lm.model.tok_emb.weight.data           # (vocab, n_embd)
        ids = [i for i in ids if 0 <= i < W.shape[0]]
        if not ids:
            return np.zeros(self.lm.model.cfg.n_embd, dtype=np.float32)
        return W[ids].mean(axis=0)

    # ------------------------------------------------------------------ #
    def perceive(self, text: str, kind: str = "perception") -> None:
        self.memory.remember(text, kind=kind, importance=1.0)

    def recall(self, query: str, k: int = 4) -> List[str]:
        return [it.text for it in self.memory.recall(query, k=k)]

    def think(self, prompt: str, max_new_tokens: int = 120, temperature: float = 0.8) -> str:
        memories = self.recall(prompt, k=3)
        context = ""
        if memories:
            context += "\n".join(memories[::-1]) + "\n"
        context += "\n".join(self.working[-4:])
        tok = self.lm.tokenizer
        u, k = tok.encode_special("<user>"), tok.encode_special("<kian>")
        ids = tok.encode(context) + [u] + tok.encode(prompt) + [k]
        ids = np.array([ids[-self.lm.model.cfg.block_size :]], dtype=np.int64)
        rng = np.random.default_rng()
        out = self.lm.model.generate(ids, max_new_tokens=max_new_tokens,
                                     temperature=temperature, top_k=40, rng=rng)
        text = tok.decode(out[0, len(ids[0]):].tolist())
        for b in ("<user>", "<kian>", "<eos>"):
            if b in text:
                text = text.split(b)[0]
        return text.strip()

    def respond(self, text: str, **gen) -> str:
        """One full cognitive cycle, returning Kian's reply."""
        self.perceive(text, kind="episodic")
        reply = self.think(text, **gen)
        self.working.append(f"<user> {text}")
        self.working.append(f"<kian> {reply}")
        self.working = self.working[-8:]
        self.memory.remember(f"I was asked: {text} — I answered: {reply}",
                             kind="episodic", importance=1.2)
        return reply

    def learn_fact(self, fact: str, importance: float = 2.0) -> None:
        self.memory.remember(fact, kind="semantic", importance=importance)

    def dream(self) -> int:
        """Consolidate memory (called while 'sleeping'). Returns items dropped."""
        return self.memory.consolidate()
