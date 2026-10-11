"""
kian.vision.conv
===============
A trainable convolutional network, built on the from-scratch autodiff engine.

The only new primitive needed is 2-D convolution. It is implemented once, as a
custom :class:`~kian.core.tensor.Tensor` operation with an exact backward pass
(``im2col`` forward, ``col2im`` for the input gradient, a matmul for the weight
gradient). Everything above it — ReLU, global average pooling, the linear
classifier — reuses the ops already verified by the gradient checker, so the
whole vision network trains by back-propagation like the language model does.

This is what lets Kian *understand* shapes and colours: given labelled images it
learns convolutional filters that fire on edges, corners and hues.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..core.nn import Linear, Module, Parameter
from ..core.tensor import Tensor


# ---------------------------------------------------------------------- #
# im2col / col2im index helpers
# ---------------------------------------------------------------------- #
def _indices(C: int, kh: int, kw: int, outH: int, outW: int, stride: int):
    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(outH), outW)
    j0 = np.tile(np.arange(kw), kh * C)
    j1 = stride * np.tile(np.arange(outW), outH)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)      # (C*kh*kw, L)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), kh * kw).reshape(-1, 1)
    return k, i, j


def conv2d(x: Tensor, weight: Tensor, bias: Tensor, stride: int = 1, pad: int = 0) -> Tensor:
    """2-D convolution. x:(N,C,H,W) weight:(Cout,C,kh,kw) bias:(Cout,)."""
    N, C, H, W = x.shape
    Cout, Cin, kh, kw = weight.shape
    assert Cin == C, "channel mismatch"
    outH = (H + 2 * pad - kh) // stride + 1
    outW = (W + 2 * pad - kw) // stride + 1
    L = outH * outW

    xp = np.pad(x.data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    k, i, j = _indices(C, kh, kw, outH, outW, stride)
    cols = xp[:, k, i, j]                            # (N, C*kh*kw, L)
    wcol = weight.data.reshape(Cout, -1)             # (Cout, C*kh*kw)
    out_data = np.matmul(wcol[None], cols)           # (N, Cout, L)
    out_data = out_data.reshape(N, Cout, outH, outW) + bias.data.reshape(1, Cout, 1, 1)

    req = x.requires_grad or weight.requires_grad or bias.requires_grad
    out = Tensor(out_data, req, (x, weight, bias), "conv2d")

    def _backward() -> None:
        if out.grad is None:
            return
        g = out.grad.reshape(N, Cout, L)             # (N, Cout, L)
        # bias gradient
        bias._accumulate(out.grad.sum(axis=(0, 2, 3)))
        # weight gradient: sum_n g[n] @ cols[n]^T
        gw = np.einsum("nol,ncl->oc", g, cols)       # (Cout, C*kh*kw)
        weight._accumulate(gw.reshape(weight.shape))
        # input gradient: scatter (col2im)
        gcols = np.matmul(wcol.T[None], g)           # (N, C*kh*kw, L)
        gxp = np.zeros_like(xp)
        np.add.at(gxp, (slice(None), k, i, j), gcols)
        if pad > 0:
            gxp = gxp[:, :, pad:pad + H, pad:pad + W]
        x._accumulate(gxp)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------- #
# Modules
# ---------------------------------------------------------------------- #
class Conv2d(Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, pad: int = 1) -> None:
        super().__init__()
        self.stride, self.pad = stride, pad
        scale = np.sqrt(2.0 / (in_ch * kernel * kernel))
        self.weight = Parameter(np.random.randn(out_ch, in_ch, kernel, kernel).astype(np.float32) * scale)
        self.bias = Parameter(np.zeros(out_ch, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, self.bias, self.stride, self.pad)


def global_avg_pool(x: Tensor) -> Tensor:
    """(N,C,H,W) -> (N,C) mean over spatial dims, using verified ops."""
    N, C, H, W = x.shape
    return x.reshape(N, C, H * W).mean(axis=2)


class TinyConvNet(Module):
    """conv-relu x2 -> global average pool -> linear classifier."""

    def __init__(self, in_ch: int = 3, num_classes: int = 3, width: int = 8) -> None:
        super().__init__()
        self.c1 = Conv2d(in_ch, width, 3, stride=1, pad=1)
        self.c2 = Conv2d(width, width * 2, 3, stride=2, pad=1)
        self.c3 = Conv2d(width * 2, width * 2, 3, stride=2, pad=1)
        self.head = Linear(width * 2, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c1(x).relu()
        x = self.c2(x).relu()
        x = self.c3(x).relu()
        x = global_avg_pool(x)
        return self.head(x)

    def loss(self, x: np.ndarray, y: np.ndarray) -> Tensor:
        logits = self.forward(Tensor(x))
        return logits.cross_entropy(np.asarray(y))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(Tensor(x)).data.argmax(-1)
