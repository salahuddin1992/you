"""
kian.core.optim
===============
Optimisers and learning-rate schedules, written from scratch.

An optimiser owns a list of :class:`~kian.core.nn.Parameter` objects and, on
each :meth:`step`, moves their ``.data`` down the gradient stored in ``.grad``.
``AdamW`` is the default used to train the language model: it decouples weight
decay from the gradient update, which is the correct formulation for
Transformers.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .nn import Parameter


def clip_grad_norm_(params: Iterable[Parameter], max_norm: float) -> float:
    """Rescale gradients in place so their global L2 norm <= ``max_norm``.

    Returns the pre-clip norm, which is a useful training-health signal: a
    sudden spike usually precedes a loss divergence.
    """
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    total = np.sqrt(sum(float((p.grad ** 2).sum()) for p in params))
    if total > max_norm and total > 0:
        scale = max_norm / (total + 1e-6)
        for p in params:
            p.grad = p.grad * scale
    return total


class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        self.params: List[Parameter] = list(params)
        self.lr = lr
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0.0) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._vel = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self._vel[i] = self.momentum * self._vel[i] + g
            p.data = p.data - self.lr * self._vel[i]


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled: bool = True,
    ) -> None:
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled = decoupled  # True => AdamW
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        bias1 = 1.0 - self.b1 ** self.t
        bias2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            # Coupled (classic Adam) decay folds into the gradient; decoupled
            # (AdamW) applies it directly to the weights.
            if self.weight_decay and not self.decoupled:
                g = g + self.weight_decay * p.data
            self._m[i] = self.b1 * self._m[i] + (1 - self.b1) * g
            self._v[i] = self.b2 * self._v[i] + (1 - self.b2) * (g * g)
            m_hat = self._m[i] / bias1
            v_hat = self._v[i] / bias2
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay and self.decoupled:
                update = update + self.weight_decay * p.data
            p.data = p.data - self.lr * update


class AdamW(Adam):
    def __init__(self, params, lr: float = 3e-4, betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.01) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, decoupled=True)


class CosineWarmup:
    """Linear warm-up followed by cosine decay to a floor learning rate."""

    def __init__(self, optimizer: Optimizer, base_lr: float, warmup: int, total: int, min_lr: float = 0.0) -> None:
        self.opt = optimizer
        self.base_lr = base_lr
        self.warmup = max(1, warmup)
        self.total = max(self.warmup + 1, total)
        self.min_lr = min_lr
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        if self.step_num <= self.warmup:
            lr = self.base_lr * self.step_num / self.warmup
        else:
            progress = (self.step_num - self.warmup) / (self.total - self.warmup)
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + np.cos(np.pi * progress)
            )
        self.opt.lr = float(lr)
        return float(lr)
