"""
kian.language.model
===================
A GPT-style, decoder-only Transformer — the language brain of Kian.

It is assembled entirely from :mod:`kian.core`: the attention scores, the
soft-max, the residual stream and the layer norms all run through the hand-written
autodiff engine, so the whole model is trainable by back-propagation with no
external deep-learning framework.

Architecture (GPT-2 family)
---------------------------
* token embedding + learned positional embedding
* ``n_layer`` pre-norm Transformer blocks, each =
    x + Attn(LN(x))   then   x + MLP(LN(x))
* causal (autoregressive) multi-head self-attention
* final LayerNorm + linear projection to vocabulary logits

The model predicts the next token at every position; trained on text, it learns
the statistics of language — letters, words, grammar — from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.nn import Dropout, Embedding, LayerNorm, Linear, Module, Parameter
from ..core.tensor import Tensor


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 64          # maximum context length (tokens)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    tie_weights: bool = True       # share token-embedding and output matrices

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# ---------------------------------------------------------------------- #
# Causal multi-head self-attention
# ---------------------------------------------------------------------- #
class CausalSelfAttention(Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.n_embd = cfg.n_embd
        # One fused projection produces query, key and value together.
        self.qkv = Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.proj = Linear(cfg.n_embd, cfg.n_embd, bias=True)
        self.attn_drop = Dropout(cfg.dropout)
        self.resid_drop = Dropout(cfg.dropout)
        # Python float (not np.float64) so it never promotes float32 attention
        # scores to float64 in either the autograd or the NumPy inference path.
        self.scale = float(1.0 / np.sqrt(self.head_dim))
        # Pre-computed additive causal mask: 0 on/below the diagonal, -inf above.
        mask = np.triu(np.full((cfg.block_size, cfg.block_size), -1e9, np.float32), k=1)
        self._mask = mask  # (block, block)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)                              # (B, T, 3C)
        q = qkv[:, :, : C]
        k = qkv[:, :, C : 2 * C]
        v = qkv[:, :, 2 * C :]

        # (B, T, C) -> (B, n_head, T, head_dim)
        def split_heads(t: Tensor) -> Tensor:
            t = t.reshape(B, T, self.n_head, self.head_dim)
            return t.transpose((0, 2, 1, 3))

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Attention scores (B, n_head, T, T)
        att = (q @ k.transpose((0, 1, 3, 2))) * self.scale
        att = att + Tensor(self._mask[:T, :T])         # causal masking
        att = att.softmax(axis=-1)
        att = self.attn_drop(att)

        out = att @ v                                  # (B, n_head, T, head_dim)
        out = out.transpose((0, 2, 1, 3)).reshape(B, T, C)   # merge heads
        return self.resid_drop(self.proj(out))


class MLP(Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.proj(self.fc(x).gelu()))


class Block(Module):
    """One pre-norm Transformer block with residual connections."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------- #
# The full model
# ---------------------------------------------------------------------- #
class GPT(Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = Embedding(cfg.block_size, cfg.n_embd)
        self.drop = Dropout(cfg.dropout)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = LayerNorm(cfg.n_embd)
        self.head = Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            # Weight tying: the output projection reuses the token-embedding
            # matrix (transposed at use). Fewer parameters, better generalisation.
            self.head.weight = self.tok_emb.weight  # shared (vocab, n_embd)
            self._tied = True
        else:
            self._tied = False
        self._init_scaled_residuals()

    def _init_scaled_residuals(self) -> None:
        # GPT-2 trick: shrink the projection that feeds each residual add by
        # 1/sqrt(2*n_layer) so the residual stream variance stays controlled.
        scale = 1.0 / np.sqrt(2 * self.cfg.n_layer)
        for blk in self.blocks:
            blk.attn.proj.weight.data *= scale
            blk.mlp.proj.weight.data *= scale

    # -- forward ----------------------------------------------------- #
    def forward(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx)
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size"
        tok = self.tok_emb(idx)                        # (B, T, C)
        pos = self.pos_emb(np.arange(T))               # (T, C)
        x = self.drop(tok + pos.reshape(1, T, self.cfg.n_embd))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        if self._tied:
            # logits = x @ Wemb^T, with Wemb shape (vocab, n_embd).
            return x @ self.tok_emb.weight.transpose()
        return self.head(x)

    def loss(self, idx: np.ndarray, targets: np.ndarray) -> Tensor:
        logits = self.forward(idx)                     # (B, T, vocab)
        B, T, V = logits.shape
        flat = logits.reshape(B * T, V)
        return flat.cross_entropy(np.asarray(targets).reshape(B * T))

    # -- generation -------------------------------------------------- #
    def generate(
        self,
        idx: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        repetition_window: int = 64,
        greedy: bool = False,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Autoregressively sample ``max_new_tokens`` continuations.

        Uses the pure-NumPy KV-cached inference path (no autograd graph), so
        each new token costs O(context) rather than O(context²). ``idx`` is an
        integer array ``(B, T0)``; returns ``(B, T0+N)``. Sampling controls:
        ``temperature``, ``top_k``, ``top_p`` (nucleus), ``min_p``, and a
        ``repetition_penalty`` applied over the last ``repetition_window``
        tokens. ``greedy`` takes the argmax.
        """
        from .infer import KVCache, forward_infer
        from .sampling import sample_logits

        rng = rng or np.random.default_rng()
        idx = np.asarray(idx).copy()
        B = idx.shape[0]
        bs = self.cfg.block_size

        def new_cache() -> "KVCache":
            return KVCache(len(self.blocks))

        # Prefill with the last block_size tokens of the prompt.
        cache = new_cache()
        window = idx[:, -bs:]
        logits = forward_infer(self, window, cache, start=0)
        last = logits[:, -1, :]
        cached_len = window.shape[1]

        for _ in range(max_new_tokens):
            nxt = np.array([
                sample_logits(last[b], rng, temperature=temperature, top_k=top_k,
                              top_p=top_p, min_p=min_p,
                              recent=idx[b, -repetition_window:].tolist(),
                              repetition_penalty=repetition_penalty, greedy=greedy)
                for b in range(B)
            ])
            idx = np.concatenate([idx, nxt[:, None]], axis=1)
            if cached_len >= bs:
                # Context window full: positions would shift on truncation, so
                # rebuild the cache from the last block_size tokens (exact, and
                # still far cheaper than per-step autograd).
                cache = new_cache()
                window = idx[:, -bs:]
                logits = forward_infer(self, window, cache, start=0)
                last = logits[:, -1, :]
                cached_len = bs
            else:
                logits = forward_infer(self, nxt[:, None], cache, start=cached_len)
                last = logits[:, -1, :]
                cached_len += 1
        return idx

    def num_parameters(self) -> int:
        n = super().num_parameters()
        # Weight tying shares one matrix counted once by the parameter walker.
        return n
