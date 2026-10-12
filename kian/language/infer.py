"""
kian.language.infer
==================
A pure-NumPy, no-autograd inference path for the Transformer, with a
key/value cache for incremental decoding.

Why
---
Training needs gradients, so the forward pass in :mod:`kian.language.model`
builds an autodiff graph. Generation needs none of that — it just wants the
next-token logits. Re-running the graph-building forward over the whole growing
context each step is the slow part of chatting and of the daily-evolution
journaling. This module recomputes the identical arithmetic in plain NumPy and
caches each block's keys and values, so each new token costs O(context) instead
of O(context²) and allocates no graph.

Correctness: every operation mirrors the training forward exactly (same GELU
tanh form, same LayerNorm eps, same weight-tied head), and this is asserted
against the autodiff path in the tests — cached and uncached greedy decoding
produce identical tokens.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


# --- numpy mirrors of the layer math ---------------------------------- #
def _ln(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    return xc / np.sqrt(var + eps) * gamma + beta


def _linear(x: np.ndarray, W: np.ndarray, b: Optional[np.ndarray]) -> np.ndarray:
    y = x @ W
    return y + b if b is not None else y


def _gelu(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class KVCache:
    """Per-block storage of concatenated keys and values across steps."""

    def __init__(self, n_layers: int) -> None:
        self.k: List[Optional[np.ndarray]] = [None] * n_layers
        self.v: List[Optional[np.ndarray]] = [None] * n_layers
        self.length = 0


def _attend(blk, x: np.ndarray, cache: KVCache, layer: int, start: int) -> np.ndarray:
    """Causal self-attention for the new positions ``x`` (B, Tn, C), appending
    to and reading from the cache for this layer."""
    a = blk.attn
    B, Tn, C = x.shape
    nh, hd = a.n_head, a.head_dim
    qkv = _linear(x, a.qkv.weight.data, a.qkv.bias.data)   # (B, Tn, 3C)
    q, k, v = qkv[:, :, :C], qkv[:, :, C:2 * C], qkv[:, :, 2 * C:]

    def heads(t):
        return t.reshape(B, Tn, nh, hd).transpose(0, 2, 1, 3)   # (B, nh, Tn, hd)

    q, k, v = heads(q), heads(k), heads(v)
    if cache.k[layer] is None:
        K, V = k, v
    else:
        K = np.concatenate([cache.k[layer], k], axis=2)
        V = np.concatenate([cache.v[layer], v], axis=2)
    cache.k[layer], cache.v[layer] = K, V
    Tcur = K.shape[2]

    scores = (q @ K.transpose(0, 1, 3, 2)) * a.scale          # (B, nh, Tn, Tcur)
    # causal mask: new token i (absolute pos start+i) sees keys 0..start+i
    qpos = start + np.arange(Tn)[:, None]
    kpos = np.arange(Tcur)[None, :]
    mask = kpos > qpos
    scores = np.where(mask[None, None], -1e9, scores)
    att = _softmax(scores)
    out = att @ V                                             # (B, nh, Tn, hd)
    out = out.transpose(0, 2, 1, 3).reshape(B, Tn, C)
    return _linear(out, a.proj.weight.data, a.proj.bias.data)


def forward_infer(model, ids: np.ndarray, cache: KVCache, start: int) -> np.ndarray:
    """Return logits (B, Tn, vocab) for the token ids at absolute positions
    ``start .. start+Tn-1``, updating ``cache`` in place."""
    ids = np.asarray(ids)
    B, Tn = ids.shape
    C = model.cfg.n_embd
    x = model.tok_emb.weight.data[ids]                        # (B, Tn, C)
    pos = model.pos_emb.weight.data[start:start + Tn]          # (Tn, C)
    x = x + pos.reshape(1, Tn, C)
    for i, blk in enumerate(model.blocks):
        x = x + _attend(blk, _ln(x, blk.ln1.gamma.data, blk.ln1.beta.data), cache, i, start)
        h = _ln(x, blk.ln2.gamma.data, blk.ln2.beta.data)
        h = _linear(_gelu(_linear(h, blk.mlp.fc.weight.data, blk.mlp.fc.bias.data)),
                    blk.mlp.proj.weight.data, blk.mlp.proj.bias.data)
        x = x + h
    x = _ln(x, model.ln_f.gamma.data, model.ln_f.beta.data)
    if model._tied:
        logits = x @ model.tok_emb.weight.data.T
    else:
        logits = _linear(x, model.head.weight.data, None)
    cache.length = start + Tn
    return logits
