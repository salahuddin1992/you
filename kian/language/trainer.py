"""
kian.language.trainer
=====================
The training loop that teaches Kian to read and write.

It runs next-token prediction over a :class:`~kian.language.data.TextDataset`,
optimising with AdamW under a cosine learning-rate schedule with warm-up, and
periodically reports training/validation loss plus gradient-norm health. The
``Trainer`` is deliberately framework-free and resumable: it mutates the
:class:`~kian.language.lm.LanguageModel` in place and its ``meta`` provenance.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..core.optim import AdamW, CosineWarmup, clip_grad_norm_
from .data import TextDataset
from .lm import LanguageModel


@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 16
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 8
    seed: int = 1337
    accum_steps: int = 1   # gradient accumulation: micro-batches per optimizer update
    log: Optional[Callable[[str], None]] = print


class Trainer:
    def __init__(self, lm: LanguageModel, cfg: TrainConfig = TrainConfig()) -> None:
        self.lm = lm
        self.cfg = cfg
        self.opt = AdamW(lm.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = CosineWarmup(self.opt, cfg.lr, cfg.warmup, cfg.steps, cfg.min_lr)
        self.rng = np.random.default_rng(cfg.seed)
        self.history: list[dict] = []

    def _log(self, msg: str) -> None:
        if self.cfg.log:
            self.cfg.log(msg)

    def evaluate(self, val: TextDataset) -> float:
        self.lm.model.eval()
        total = 0.0
        for _ in range(self.cfg.eval_batches):
            x, y = val.batch(self.cfg.batch_size, self.rng)
            total += self.lm.model.loss(x, y).item()
        self.lm.model.train()
        return total / max(1, self.cfg.eval_batches)

    def train(self, train_ds: TextDataset, val_ds: Optional[TextDataset] = None) -> LanguageModel:
        c = self.cfg
        model = self.lm.model
        model.train()
        start = time.time()
        running = None
        accum = max(1, int(c.accum_steps))
        gnorm = 0.0
        for step in range(1, c.steps + 1):
            lr = self.sched.step()
            x, y = train_ds.batch(c.batch_size, self.rng)
            if (step - 1) % accum == 0:
                model.zero_grad()
            loss = model.loss(x, y)
            lv = loss.item()
            if not math.isfinite(lv):
                # Check BEFORE backward/step: a NaN loss means NaN gradients,
                # which clipping cannot remove and AdamW would write into every
                # parameter and its moment buffers. Nothing has been applied
                # yet, so the model (and any later save) stays intact.
                model.zero_grad()
                self._log(f"[step {step}] non-finite loss ({lv}); stopping before it touches the weights.")
                break
            # Scale so the accumulated gradient equals the mean over micro-batches.
            (loss * (1.0 / accum)).backward()
            if step % accum == 0:
                gnorm = clip_grad_norm_(model.parameters(), c.grad_clip)
                if not math.isfinite(gnorm):
                    model.zero_grad()
                    self._log(f"[step {step}] non-finite gradient norm; update skipped, stopping.")
                    break
                self.opt.step()
            running = lv if running is None else 0.9 * running + 0.1 * lv
            self.lm.meta["steps"] = self.lm.meta.get("steps", 0) + 1
            self.lm.meta["tokens_seen"] = self.lm.meta.get("tokens_seen", 0) + x.size

            if step % c.eval_every == 0 or step == 1:
                val_loss = self.evaluate(val_ds) if val_ds is not None else float("nan")
                ppl = math.exp(min(20.0, running))
                elapsed = time.time() - start
                rec = {"step": step, "train": running, "val": val_loss,
                       "ppl": ppl, "lr": lr, "grad_norm": gnorm}
                self.history.append(rec)
                vtxt = f"val {val_loss:.3f}  " if val_ds is not None else ""
                self._log(
                    f"[step {step:4d}/{c.steps}] loss {running:.3f}  {vtxt}"
                    f"ppl {ppl:6.1f}  lr {lr:.1e}  gnorm {gnorm:4.1f}  "
                    f"{step / max(1e-9, elapsed):.1f} it/s"
                )
        return self.lm
