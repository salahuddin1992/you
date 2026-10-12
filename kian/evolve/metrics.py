"""
kian.evolve.metrics
==================
Kian looking at its own growth. Two measures:

* :func:`perplexity` — how surprised the language brain is by a text. Lower is
  better; it is the standard yardstick of a language model and the honest
  number to watch as the being learns.
* :func:`growth_summary` — reads the being's ``evolution_log.jsonl`` and distils
  the learning trajectory: days lived, loss trend, stage path, tokens seen.

These let the daily loop, the CLI and the UI report real progress instead of a
vibe.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np


def perplexity(lm, text: str, max_tokens: int = 4000) -> float:
    """Mean per-token perplexity of ``text`` under ``lm`` (eval mode)."""
    ids = lm.tokenizer.encode(text)[:max_tokens]
    if len(ids) < 2:
        return float("inf")
    bs = lm.model.cfg.block_size
    was_training = lm.model.training
    lm.model.eval()
    losses: List[float] = []
    for i in range(0, len(ids) - 1, bs):
        chunk = ids[i:i + bs + 1]
        if len(chunk) < 2:
            break
        x = np.array([chunk[:-1]], dtype=np.int64)
        y = np.array([chunk[1:]], dtype=np.int64)
        losses.append(lm.model.loss(x, y).item())
    if was_training:
        lm.model.train()
    if not losses:
        return float("inf")
    return math.exp(min(20.0, sum(losses) / len(losses)))


def growth_summary(home: str | Path) -> dict:
    """Summarise the being's learning trajectory from its evolution log."""
    from .daily_loop import history
    log = history(home)
    if not log:
        return {"days": 0, "note": "لم يبدأ التطوّر بعد · no evolution cycles yet"}
    train = [r.get("train_loss") for r in log if isinstance(r.get("train_loss"), (int, float))]
    stages = []
    for r in log:
        s = r.get("stage_after")
        if s and (not stages or stages[-1] != s):
            stages.append(s)
    first, last = (train[0], train[-1]) if train else (None, None)
    improved = (first is not None and last is not None and last < first)
    return {
        "days": len(log),
        "first_loss": first,
        "last_loss": last,
        "improved": improved,
        "delta": round(first - last, 4) if improved else (round(last - first, 4) if first else None),
        "stage_path": stages,
        "current_stage": log[-1].get("stage_after"),
        "tokens_seen": log[-1].get("tokens_seen"),
        "total_grow_events": sum(1 for r in log if r.get("grew")),
    }


def summary_text(home: str | Path) -> str:
    g = growth_summary(home)
    if g["days"] == 0:
        return g["note"]
    trend = "تحسّن" if g.get("improved") else "ثابت/تذبذب"
    trend_en = "improving" if g.get("improved") else "flat/noisy"
    path = " → ".join(g["stage_path"]) if g["stage_path"] else g.get("current_stage", "")
    return (
        f"عاش {g['days']} يومًا، الخسارة {g['first_loss']:.2f} ← {g['last_loss']:.2f} ({trend})، "
        f"المراحل: {path}، تعلّم {g['tokens_seen']:,} رمزًا، نما {g['total_grow_events']} مرة. "
        f"| Lived {g['days']} days, loss {g['first_loss']:.2f} -> {g['last_loss']:.2f} "
        f"({trend_en}), stages: {path}, {g['tokens_seen']:,} tokens, grew "
        f"{g['total_grow_events']}x."
    )
