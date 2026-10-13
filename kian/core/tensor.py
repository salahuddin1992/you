"""
kian.core.tensor
================
A from-scratch reverse-mode automatic differentiation engine built on NumPy.

This is the mathematical heart of Kian. Every neural network in the project —
the language model that learns to read and write, the vision network that
understands shapes and colours — is trained by the gradients this file
produces. There is no PyTorch, no TensorFlow, no autograd library underneath:
the chain rule is implemented here, by hand, one operation at a time.

Design
------
* ``Tensor`` wraps an ``np.ndarray`` and records the operation that produced it.
* Each operation defines a local ``_backward`` closure that knows how to push
  gradient from the output back onto its inputs (the vector-Jacobian product).
* ``Tensor.backward`` performs a topological sort of the computation graph and
  invokes every ``_backward`` in reverse order.
* Broadcasting is handled explicitly by :func:`_unbroadcast`, so gradients are
  always summed back onto the original (pre-broadcast) shape. Getting this
  wrong is the single most common autodiff bug; it is tested in ``tests/``.

The API intentionally mirrors a small slice of PyTorch so the rest of the code
base reads naturally, but every line is original and runs on CPU with nothing
but NumPy.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np

ArrayLike = Union["Tensor", np.ndarray, float, int, list]
Shape = Tuple[int, ...]


def _as_array(x: ArrayLike) -> np.ndarray:
    """Coerce to an ndarray, preserving an existing floating dtype.

    New data (python scalars, lists, integer arrays) becomes float32 — the
    default working precision for training. A float64 array passed in is kept
    at float64, which lets gradient-checking run in high precision.
    """
    if isinstance(x, Tensor):
        return x.data
    if isinstance(x, np.ndarray):
        # Preserve an existing float dtype (so float64 gradient-checks stay
        # exact); promote integer arrays to the float32 working precision.
        return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float32)
    if isinstance(x, np.floating):
        # NumPy float scalar (e.g. the result of ``ndarray.sum(axis=None)``):
        # keep its precision rather than silently collapsing it to float32.
        return np.asarray(x)
    return np.asarray(x, dtype=np.float32)


def _unbroadcast(grad: np.ndarray, shape: Shape) -> np.ndarray:
    """Sum ``grad`` back down to ``shape`` after NumPy broadcasting.

    When ``a + b`` broadcasts ``a`` of shape ``(1, N)`` against ``b`` of shape
    ``(M, N)``, the resulting gradient has shape ``(M, N)``. To attribute it to
    ``a`` we must sum over the broadcast axes and restore the original rank.
    """
    if grad.shape == shape:
        return grad
    # 1. Sum over the leading axes that were added by broadcasting.
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    # 2. Sum over axes that were size 1 in the original but expanded.
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def _is_basic_index(idx) -> bool:
    """True when ``idx`` uses only ints/slices/Ellipsis/None (no fancy arrays).

    Basic indexing selects each element at most once, so its gradient can be
    written with a plain assignment instead of the much slower ``np.add.at``
    (which must tolerate repeated indices). Fancy indexing — used by
    ``Embedding`` lookups, where ids repeat — still takes the accumulate path.
    """
    parts = idx if isinstance(idx, tuple) else (idx,)
    return all(p is None or p is Ellipsis or isinstance(p, (int, slice, np.integer))
               for p in parts)


class Tensor:
    """A node in the automatic-differentiation graph."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    # Tell NumPy not to handle ufuncs with a Tensor operand, so expressions
    # like ``np.float64(2.0) - tensor`` dispatch to Tensor.__rsub__ (and keep
    # the Tensor's dtype) instead of NumPy coercing the Tensor itself.
    __array_ufunc__ = None

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ) -> None:
        self.data: np.ndarray = _as_array(data)
        self.requires_grad: bool = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: Tuple["Tensor", ...] = tuple(_children)
        self._op: str = _op

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __len__(self) -> int:
        return self.shape[0]

    @staticmethod
    def zeros(shape: Shape, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad)

    @staticmethod
    def ones(shape: Shape, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.ones(shape, dtype=np.float32), requires_grad)

    # ------------------------------------------------------------------ #
    # Gradient bookkeeping
    # ------------------------------------------------------------------ #
    def _accumulate(self, g: np.ndarray) -> None:
        if not self.requires_grad:
            return
        g = _unbroadcast(g, self.shape)
        if g.dtype != self.data.dtype:
            g = g.astype(self.data.dtype)   # copy only when a cast is needed
        if self.grad is None:
            # Own the buffer: ``g`` may alias an upstream grad array.
            self.grad = g.copy()
        else:
            self.grad += g                  # in-place accumulate, no new array

    def _coerce(self, other) -> "Tensor":
        """Wrap an operand as a Tensor. Python/NumPy scalars and lists take THIS
        tensor's dtype, so an np.float64 scalar (e.g. ``1/np.sqrt(d)``) never
        promotes a float32 activation/gradient stream to float64."""
        if isinstance(other, Tensor):
            return other
        if isinstance(other, np.ndarray):
            return Tensor(other)
        return Tensor(np.asarray(other, dtype=self.data.dtype))

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Back-propagate from this (typically scalar) tensor."""
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: "Tensor") -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() on a non-scalar requires an explicit gradient"
                )
            grad = np.ones_like(self.data)
        self.grad = grad.astype(self.data.dtype)

        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------ #
    # Elementwise arithmetic
    # ------------------------------------------------------------------ #
    def __add__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        out = Tensor(
            self.data + other.data,
            self.requires_grad or other.requires_grad,
            (self, other),
            "+",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            self._accumulate(out.grad)
            other._accumulate(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        out = Tensor(
            self.data * other.data,
            self.requires_grad or other.requires_grad,
            (self, other),
            "*",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            self._accumulate(out.grad * other.data)
            other._accumulate(out.grad * self.data)

        out._backward = _backward
        return out

    def __pow__(self, p: Union[int, float]) -> "Tensor":
        assert isinstance(p, (int, float)), "only scalar powers are supported"
        out = Tensor(self.data ** p, self.requires_grad, (self,), f"**{p}")

        def _backward() -> None:
            if out.grad is None:
                return
            if p == 2:
                d = 2.0 * self.data
            elif p == -1:
                d = -1.0 / (self.data * self.data)
            elif p == 0.5:
                d = 0.5 / out.data            # out.data == sqrt(x)
            else:
                d = p * self.data ** (p - 1)
            self._accumulate(out.grad * d)

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = self._coerce(other)
        out = Tensor(
            self.data @ other.data,
            self.requires_grad or other.requires_grad,
            (self, other),
            "@",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # d/dA (A@B) = grad @ B^T ; d/dB (A@B) = A^T @ grad
            a, b, g = self.data, other.data, out.grad
            # Promote 1-D operands to 2-D so the standard formulas apply, then
            # squeeze the gradients back (vector @ matrix, matrix @ vector, dot).
            a2 = a[None, :] if a.ndim == 1 else a
            b2 = b[:, None] if b.ndim == 1 else b
            g2 = g
            if a.ndim == 1:
                g2 = g2[None, ...]
            if b.ndim == 1:
                g2 = g2[..., None]
            ga = g2 @ np.swapaxes(b2, -1, -2)
            gb = np.swapaxes(a2, -1, -2) @ g2
            if a.ndim == 1:
                ga = ga[0]
            if b.ndim == 1:
                gb = gb[..., 0]
            self._accumulate(ga)
            other._accumulate(gb)

        out._backward = _backward
        return out

    # Reverse / convenience operators ---------------------------------- #
    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-self._coerce(other))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) + (-self)

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) * (self ** -1.0)

    # ------------------------------------------------------------------ #
    # Shape operations
    # ------------------------------------------------------------------ #
    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,), "reshape")

        def _backward() -> None:
            if out.grad is None:
                return
            self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> "Tensor":
        out = Tensor(
            np.transpose(self.data, axes), self.requires_grad, (self,), "transpose"
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if axes is None:
                self._accumulate(np.transpose(out.grad))
            else:
                self._accumulate(np.transpose(out.grad, np.argsort(axes)))

        out._backward = _backward
        return out

    def swapaxes(self, a: int, b: int) -> "Tensor":
        out = Tensor(
            np.swapaxes(self.data, a, b), self.requires_grad, (self,), "swapaxes"
        )

        def _backward() -> None:
            if out.grad is None:
                return
            self._accumulate(np.swapaxes(out.grad, a, b))

        out._backward = _backward
        return out

    def __getitem__(self, idx) -> "Tensor":
        out = Tensor(self.data[idx], self.requires_grad, (self,), "getitem")

        def _backward() -> None:
            if out.grad is None:
                return
            g = np.zeros_like(self.data)
            if _is_basic_index(idx):
                g[idx] = out.grad             # each element selected once
            else:
                np.add.at(g, idx, out.grad)   # repeated ids must accumulate
            self._accumulate(g)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Reductions
    # ------------------------------------------------------------------ #
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            self.requires_grad,
            (self,),
            "sum",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis if isinstance(axis, int) else tuple(axis))
            self._accumulate(np.broadcast_to(g, self.shape))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = int(np.prod([self.data.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def var(self, axis=None, keepdims: bool = False) -> "Tensor":
        mu = self.mean(axis=axis, keepdims=True)
        centered = self - mu
        return (centered * centered).mean(axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        m = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(m, self.requires_grad, (self,), "max")

        def _backward() -> None:
            if out.grad is None:
                return
            mask = (self.data == (m if keepdims else np.expand_dims(m, axis))).astype(
                np.float32
            ) if axis is not None and not keepdims else (self.data == m).astype(np.float32)
            g = out.grad if keepdims or axis is None else np.expand_dims(out.grad, axis)
            self._accumulate(mask * g)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Activations & math
    # ------------------------------------------------------------------ #
    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = Tensor(e, self.requires_grad, (self,), "exp")

        def _backward() -> None:
            if out.grad is None:
                return
            self._accumulate(out.grad * e)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data + 1e-12), self.requires_grad, (self,), "log")

        def _backward() -> None:
            if out.grad is None:
                return
            self._accumulate(out.grad / (self.data + 1e-12))

        out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        return self ** 0.5

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,), "relu")

        def _backward() -> None:
            if out.grad is None:
                return
            self._accumulate(out.grad * (self.data > 0).astype(np.float32))

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _backward() -> None:
            if out.grad is None:
                return
            self._accumulate(out.grad * (1.0 - t * t))

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, self.requires_grad, (self,), "sigmoid")

        def _backward() -> None:
            if out.grad is None:
                return
            self._accumulate(out.grad * s * (1.0 - s))

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        """Gaussian Error Linear Unit (tanh approximation, as used by GPT-2).

        Powers are written as explicit products (``x*x*x`` rather than
        ``x**3``): NumPy's generic ``power`` is several times slower than
        multiplication on large float arrays, and GELU runs on the widest
        activation in the network, so this alone is a large share of a step.
        """
        x = self.data
        c = x.dtype.type(np.sqrt(2.0 / np.pi))
        k = x.dtype.type(0.044715)
        x2 = x * x
        inner = c * (x + k * x2 * x)
        t = np.tanh(inner)
        half_1pt = 0.5 * (1.0 + t)
        val = x * half_1pt
        out = Tensor(val, self.requires_grad, (self,), "gelu")

        def _backward() -> None:
            if out.grad is None:
                return
            sech2 = 1.0 - t * t
            dinner = c * (1.0 + 3.0 * k * x2)
            dx = half_1pt + 0.5 * x * sech2 * dinner
            self._accumulate(out.grad * dx)

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        """Numerically stable softmax with an exact Jacobian-vector backward."""
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(p, self.requires_grad, (self,), "softmax")

        def _backward() -> None:
            if out.grad is None:
                return
            # dL/dx = p * (g - sum(g * p))
            g = out.grad
            dot = (g * p).sum(axis=axis, keepdims=True)
            self._accumulate(p * (g - dot))

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Losses
    # ------------------------------------------------------------------ #
    def cross_entropy(self, targets: np.ndarray) -> "Tensor":
        """Softmax cross-entropy over the last axis.

        ``self`` holds raw logits of shape ``(N, C)``; ``targets`` is an
        integer array of shape ``(N,)``. Returns the mean loss. The softmax and
        the log are fused for numerical stability and an exact gradient.
        """
        logits = self.data
        n = logits.shape[0]
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        probs = e / e.sum(axis=-1, keepdims=True)
        log_probs = z - np.log(e.sum(axis=-1, keepdims=True))
        loss_val = -log_probs[np.arange(n), targets].mean()
        out = Tensor(np.asarray(loss_val, dtype=logits.dtype), self.requires_grad, (self,), "ce")

        def _backward() -> None:
            grad = probs.copy()
            grad[np.arange(n), targets] -= 1.0
            grad /= n
            scale = out.grad if out.grad is not None else 1.0
            self._accumulate(grad * scale)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def numpy(self) -> np.ndarray:
        return self.data

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, op={self._op or 'leaf'}, "
            f"requires_grad={self.requires_grad})"
        )


def cat(tensors: list["Tensor"], axis: int = -1) -> "Tensor":
    """Concatenate tensors along ``axis`` with gradient routing."""
    data = np.concatenate([t.data for t in tensors], axis=axis)
    req = any(t.requires_grad for t in tensors)
    out = Tensor(data, req, tuple(tensors), "cat")
    sizes = [t.shape[axis] for t in tensors]

    def _backward() -> None:
        if out.grad is None:
            return
        offset = 0
        for t, s in zip(tensors, sizes):
            sl = [slice(None)] * out.grad.ndim
            sl[axis] = slice(offset, offset + s)
            t._accumulate(out.grad[tuple(sl)])
            offset += s

    out._backward = _backward
    return out


def stack(tensors: list["Tensor"], axis: int = 0) -> "Tensor":
    nd = tensors[0].ndim + 1
    ax = axis if axis >= 0 else axis + nd      # e.g. -1 on (3,4) -> 2, giving (3,4,N)
    expanded = [t.reshape(*t.shape[:ax], 1, *t.shape[ax:]) for t in tensors]
    return cat(expanded, axis=ax)


def layer_norm(x: "Tensor", gamma: "Tensor", beta: "Tensor", eps: float = 1e-5) -> "Tensor":
    """Fused LayerNorm over the last axis with an analytic backward pass.

    Composing LayerNorm from ``mean/var/sqrt/div/mul/add`` builds ~8 graph nodes
    and as many intermediate arrays per call; the Transformer calls it many
    times per step. This single op computes the same value and the exact
    gradient (the standard LayerNorm backward) with a fraction of the overhead.
    """
    xd = x.data
    mu = xd.mean(axis=-1, keepdims=True)
    xc = xd - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = xc * rstd
    out_data = xhat * gamma.data + beta.data
    req = x.requires_grad or gamma.requires_grad or beta.requires_grad
    out = Tensor(out_data, req, (x, gamma, beta), "layernorm")

    def _backward() -> None:
        if out.grad is None:
            return
        g = out.grad
        axes = tuple(range(g.ndim - 1))
        gamma._accumulate((g * xhat).sum(axis=axes))
        beta._accumulate(g.sum(axis=axes))
        gxhat = g * gamma.data
        m1 = gxhat.mean(axis=-1, keepdims=True)
        m2 = (gxhat * xhat).mean(axis=-1, keepdims=True)
        x._accumulate(rstd * (gxhat - m1 - xhat * m2))

    out._backward = _backward
    return out
