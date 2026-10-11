"""Numerical gradient checks for the from-scratch autodiff engine.

Each test compares analytic gradients from ``Tensor.backward`` against central
finite differences computed in float64. This is the correctness proof the whole
project rests on.
"""
import numpy as np
import pytest

from kian.core.tensor import Tensor, cat


def _grads(fn, x0, eps=1e-6):
    x = Tensor(x0.copy(), requires_grad=True)
    fn(x).backward()
    ga = x.grad.copy()
    gn = np.zeros_like(x0)
    it = np.nditer(x0, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        gn[i] = (fn(Tensor(xp)).data - fn(Tensor(xm)).data) / (2 * eps)
        it.iternext()
    return ga, gn


def _rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_matmul_gelu():
    W, v = np.random.randn(4, 3), np.random.randn(2, 3)
    a, n = _grads(lambda x: ((x @ Tensor(W)).gelu() * Tensor(v)).sum(), np.random.randn(2, 4))
    assert _rel(a, n) < 1e-5


def test_cross_entropy():
    a, n = _grads(lambda x: x.cross_entropy(np.array([2, 0])), np.random.randn(2, 5))
    assert _rel(a, n) < 1e-5


def test_layernorm_like():
    g = np.random.randn(3, 4)
    a, n = _grads(lambda x: (((x - x.mean(-1, True)) / (x.var(-1, True) + 1e-5).sqrt()) * Tensor(g)).sum(),
                  np.random.randn(3, 4))
    assert _rel(a, n) < 1e-5


def test_softmax_entropy():
    a, n = _grads(lambda x: (x.softmax(-1) * x.softmax(-1).log()).sum(), np.random.randn(3, 5))
    assert _rel(a, n) < 1e-4


def test_batched_matmul():
    B = np.random.randn(3, 2)
    a, n = _grads(lambda x: (x @ Tensor(B)).sum(), np.random.randn(5, 4, 3))
    assert _rel(a, n) < 1e-4


def test_gather():
    idx = np.array([0, 2, 2, 1]); M = np.random.randn(4, 3)
    a, n = _grads(lambda x: (x[idx] * Tensor(M)).sum(), np.random.randn(4, 3))
    assert _rel(a, n) < 1e-5


def test_reshape_transpose():
    a, n = _grads(lambda x: x.reshape(2, 6).transpose().sum(), np.random.randn(3, 4))
    assert _rel(a, n) < 1e-6


def test_concat_backward():
    B = np.random.randn(2, 3)
    a, n = _grads(lambda x: cat([x, Tensor(B)], axis=0).sum(), np.random.randn(2, 3))
    assert _rel(a, n) < 1e-6
