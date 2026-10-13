"""Tests for v0.3 generation: the KV-cached NumPy inference path must match the
autograd forward exactly, and the sampler must behave."""
import numpy as np
import pytest

from kian.language.model import GPT, GPTConfig
from kian.language.infer import KVCache, forward_infer
from kian.language.sampling import sample_logits, filter_top_p, filter_top_k


@pytest.fixture
def model():
    np.random.seed(0)
    cfg = GPTConfig(vocab_size=120, block_size=24, n_layer=3, n_head=4, n_embd=48, dropout=0.0)
    m = GPT(cfg)
    m.eval()
    return m


def test_infer_matches_autograd(model):
    ctx = np.random.randint(0, 120, (2, 15))
    ref = model.forward(ctx).data[:, -1, :]
    cache = KVCache(len(model.blocks))
    inf = forward_infer(model, ctx, cache, start=0)[:, -1, :]
    assert np.abs(ref - inf).max() < 1e-5


def test_cached_greedy_equals_reference(model):
    cfg = model.cfg

    def reference(idx, n):
        idx = idx.copy()
        for _ in range(n):
            lg = model.forward(idx[:, -cfg.block_size:]).data[:, -1, :]
            idx = np.concatenate([idx, lg.argmax(-1)[:, None]], 1)
        return idx

    prompt = np.random.randint(0, 120, (2, 6))
    # 6 + 30 = 36 > block_size(24): exercises the cache-rebuild path too
    assert np.array_equal(reference(prompt, 30), model.generate(prompt, 30, greedy=True))


def test_sampler_valid_and_deterministic():
    rng = np.random.default_rng(0)
    logits = np.random.randn(100)
    for _ in range(20):
        t = sample_logits(logits, rng, temperature=0.8, top_k=40, top_p=0.9,
                          min_p=0.02, recent=[1, 1, 2], repetition_penalty=1.4)
        assert 0 <= t < 100
    # greedy ignores rng
    assert sample_logits(logits, rng, greedy=True) == int(logits.argmax())


def test_filters():
    logits = np.array([3.0, 2.0, 1.0, 0.0, -1.0])
    # top_k=2 keeps only the two largest finite
    tk = filter_top_k(logits.copy(), 2)
    assert np.isfinite(tk).sum() == 2
    # top_p keeps at least the top token
    tp = filter_top_p(logits.copy(), 0.5)
    assert np.isfinite(tp).sum() >= 1 and np.argmax(tp) == 0


def test_repetition_penalty_reduces_repeat(model):
    # With a strong penalty, greedy-ish decoding should not collapse to a single
    # repeated token as readily. We just check it runs and returns valid ids.
    prompt = np.array([[5, 6, 7]])
    out = model.generate(prompt, 20, temperature=0.9, repetition_penalty=1.8,
                         rng=np.random.default_rng(3))
    assert out.shape == (1, 23)
    assert out.min() >= 0 and out.max() < model.cfg.vocab_size
