"""
kian.language.sampling
=====================
Token sampling strategies for generation, operating on raw logits (NumPy).

Combines, in order: repetition penalty, temperature, top-k, nucleus (top-p) and
min-p filtering, then samples (or takes the argmax when ``greedy``). These are
what make generated text read well — repetition penalty stops loops, top-p/min-p
cut the improbable tail without the hard cliff of top-k alone.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def apply_repetition_penalty(logits: np.ndarray, recent: Sequence[int], penalty: float) -> np.ndarray:
    """Divide the logit of already-seen tokens by ``penalty`` (>1 discourages).
    Follows the CTRL formulation: scale positive logits down, negative ones
    further down (multiply)."""
    if not penalty or penalty == 1.0 or len(recent) == 0:
        return logits
    for t in set(int(x) for x in recent):
        if 0 <= t < logits.shape[-1]:
            logits[t] = logits[t] / penalty if logits[t] > 0 else logits[t] * penalty
    return logits


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def filter_top_k(logits: np.ndarray, k: Optional[int]) -> np.ndarray:
    if not k or k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = np.sort(logits)[-k]
    return np.where(logits < kth, -np.inf, logits)


def filter_top_p(logits: np.ndarray, p: Optional[float]) -> np.ndarray:
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability exceeds ``p``; mask the rest."""
    if not p or p >= 1.0 or p <= 0.0:
        return logits
    order = np.argsort(logits)[::-1]
    probs = _softmax(logits[order])
    cum = np.cumsum(probs)
    # Nucleus = the smallest prefix whose mass reaches p. A token is kept when
    # the mass *before* it is still below p, so the token that crosses the
    # threshold is included (e.g. [0.6, 0.3, 0.1] with p=0.8 keeps two).
    keep = (cum - probs) < p
    keep[0] = True  # always keep the top token
    masked = np.full_like(logits, -np.inf)
    masked[order[keep]] = logits[order[keep]]
    return masked


def filter_min_p(logits: np.ndarray, min_p: Optional[float]) -> np.ndarray:
    """Keep tokens whose probability is at least ``min_p`` times the top
    token's probability."""
    if not min_p or min_p <= 0.0:
        return logits
    probs = _softmax(logits)
    thresh = min_p * probs.max()
    return np.where(probs < thresh, -np.inf, logits)


def sample_logits(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    recent: Optional[Sequence[int]] = None,
    repetition_penalty: float = 1.0,
    greedy: bool = False,
) -> int:
    """Return one sampled token id from a 1-D logits vector."""
    logits = logits.astype(np.float64).copy()
    if recent is not None:
        logits = apply_repetition_penalty(logits, recent, repetition_penalty)
    if greedy or temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / max(1e-6, temperature)
    logits = filter_top_k(logits, top_k)
    logits = filter_top_p(logits, top_p)
    logits = filter_min_p(logits, min_p)
    probs = _softmax(logits)
    # guard against all-masked / NaN
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        return int(np.argmax(np.nan_to_num(logits, neginf=-1e30)))
    return int(rng.choice(len(probs), p=probs))
