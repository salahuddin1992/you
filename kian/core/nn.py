"""
kian.core.nn
============
Neural-network building blocks implemented on top of :mod:`kian.core.tensor`.

Every layer here is a thin, explicit composition of the autodiff primitives:
no hidden C kernels, no framework. ``Module`` gives parameter collection and
train/eval mode; the layers (``Linear``, ``Embedding``, ``LayerNorm``,
``Dropout``) are the pieces the Transformer and the vision network are
assembled from.
"""

from __future__ import annotations

from typing import Iterator, List

import numpy as np

from .tensor import Tensor


# ---------------------------------------------------------------------- #
# Parameter & Module
# ---------------------------------------------------------------------- #
class Parameter(Tensor):
    """A Tensor that is, by definition, trainable."""

    def __init__(self, data) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    """Base class. Holds sub-modules and parameters by attribute discovery."""

    def __init__(self) -> None:
        self.training: bool = True

    # -- parameter / module traversal -------------------------------- #
    def parameters(self) -> Iterator[Parameter]:
        seen: set[int] = set()
        for _name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    yield value
            elif isinstance(value, Module):
                for p in value.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        yield p
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        for p in item.parameters():
                            if id(p) not in seen:
                                seen.add(id(p))
                                yield p
                    elif isinstance(item, Parameter) and id(item) not in seen:
                        seen.add(id(item))
                        yield item

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        for i, p in enumerate(self.parameters()):
            yield f"param_{i}", p

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def train(self) -> "Module":
        self._set_mode(True)
        return self

    def eval(self) -> "Module":
        self._set_mode(False)
        return self

    def _set_mode(self, mode: bool) -> None:
        self.training = mode
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value._set_mode(mode)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item._set_mode(mode)

    # -- (de)serialisation ------------------------------------------- #
    def state_dict(self) -> List[np.ndarray]:
        return [p.data.copy() for p in self.parameters()]

    def load_state_dict(self, state: List[np.ndarray]) -> None:
        params = list(self.parameters())
        if len(params) != len(state):
            raise ValueError(
                f"parameter count mismatch: model has {len(params)}, "
                f"checkpoint has {len(state)}"
            )
        for p, s in zip(params, state):
            if p.shape != s.shape:
                raise ValueError(f"shape mismatch {p.shape} vs {s.shape}")
            p.data = s.astype(np.float32).copy()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# Layers
# ---------------------------------------------------------------------- #
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        # Kaiming/He-style scaling keeps activation variance stable through depth.
        scale = np.sqrt(2.0 / in_features)
        self.weight = Parameter(
            np.random.randn(in_features, out_features).astype(np.float32) * scale
        )
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """A lookup table mapping integer ids to dense vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        )

    def forward(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx)
        return self.weight[idx]


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = Parameter(np.ones(dim, dtype=np.float32))
        self.beta = Parameter(np.zeros(dim, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = (x - mu) / (var + self.eps).sqrt()
        return normed * self.gamma + self.beta


class Dropout(Module):
    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return x
        # Inverted dropout: scale at train time so inference needs no change.
        mask = (np.random.rand(*x.shape) >= self.p).astype(np.float32) / (1.0 - self.p)
        return x * Tensor(mask)


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
