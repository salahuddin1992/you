"""Behavioural tests: tokenizer, language model learning, CNN conv gradient,
vision IO, memory persistence, and the computer-control safety gate."""
import numpy as np
import pytest

from kian.core.optim import AdamW
from kian.core.tensor import Tensor


# --------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------- #
def test_tokenizer_roundtrip_multilingual(tmp_path):
    from kian.language.tokenizer import Tokenizer
    corpus = ("السلام عليكم ورحمة الله. الذكاء الصناعي. Hello world Kian. "
              "123 café 🌍 你好 ") * 30
    tok = Tokenizer().train(corpus, vocab_size=400)
    for s in ["السلام عليكم", "Kian كيان 🌍", "لا إله إلا الله", "你好世界", ""]:
        assert tok.decode(tok.encode(s)) == s
    # save/load keeps it consistent
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = __import__("kian.language.tokenizer", fromlist=["Tokenizer"]).Tokenizer.load(p)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.decode(tok2.encode("مرحبا world")) == "مرحبا world"


# --------------------------------------------------------------------- #
# Language model actually learns (overfits a fixed batch)
# --------------------------------------------------------------------- #
def test_transformer_overfits():
    from kian.language.model import GPT, GPTConfig
    np.random.seed(0)
    cfg = GPTConfig(vocab_size=24, block_size=12, n_layer=2, n_head=2, n_embd=48, dropout=0.0)
    model = GPT(cfg)
    data = np.random.randint(0, cfg.vocab_size, size=(4, 9))
    x, y = data[:, :-1], data[:, 1:]
    opt = AdamW(model.parameters(), lr=4e-3)
    first = model.loss(x, y).item()
    for _ in range(60):
        model.zero_grad(); loss = model.loss(x, y); loss.backward(); opt.step()
    assert loss.item() < first * 0.3
    assert (model.forward(x).data.argmax(-1) == y).mean() > 0.9


# --------------------------------------------------------------------- #
# Conv gradient + CNN learns
# --------------------------------------------------------------------- #
def test_conv2d_gradient():
    from kian.vision.conv import conv2d
    np.random.seed(0)
    x0 = np.random.randn(2, 3, 7, 7).astype(np.float64)
    w0 = np.random.randn(4, 3, 3, 3).astype(np.float64)
    b0 = np.random.randn(4).astype(np.float64)
    xt = Tensor(x0, True)
    conv2d(xt, Tensor(w0, True), Tensor(b0, True), stride=2, pad=1).sum().backward()
    eps = 1e-6
    for idx in [(0, 1, 3, 4), (1, 2, 0, 6)]:
        xp = x0.copy(); xp[idx] += eps
        xm = x0.copy(); xm[idx] -= eps
        num = (conv2d(Tensor(xp), Tensor(w0), Tensor(b0), 2, 1).data.sum()
               - conv2d(Tensor(xm), Tensor(w0), Tensor(b0), 2, 1).data.sum()) / (2 * eps)
        assert abs(num - xt.grad[idx]) / (abs(num) + abs(xt.grad[idx]) + 1e-9) < 1e-5


# --------------------------------------------------------------------- #
# Image IO (pure-python PNG) round-trips exactly
# --------------------------------------------------------------------- #
def test_png_roundtrip(tmp_path):
    from kian.util.imageio import save_png, load_image
    a = (np.random.rand(20, 30, 3) * 255).astype(np.uint8)
    save_png(a, tmp_path / "x.png")
    assert np.array_equal(load_image(tmp_path / "x.png"), a)


# --------------------------------------------------------------------- #
# Memory persistence
# --------------------------------------------------------------------- #
def test_memory_persist(tmp_path):
    from kian.mind.memory import Memory
    m = Memory(dim=32)
    m.remember("الشمس نجم", kind="semantic", importance=2.0)
    m.remember("the sea is blue")
    m.save(tmp_path)
    m2 = Memory.load(tmp_path, dim=32)
    assert len(m2) == 2
    assert any("الشمس" in it.text for it in m2.items)


# --------------------------------------------------------------------- #
# Safety gate
# --------------------------------------------------------------------- #
def test_safety_gate(tmp_path):
    from kian.senses.policy import SafetyPolicy
    from kian.senses.computer import Computer
    audit = tmp_path / "audit.log"

    # disabled by default
    assert Computer(audit_path=audit).open_url("https://x.com").ok is False
    # commands off by default even with approving gate
    c = Computer(SafetyPolicy(), confirm=lambda a: True, audit_path=audit, enabled=True)
    assert c.run_allowed_command(["notepad"]).ok is False
    # allow-listed echo runs; destructive/security ops refused even if allow-listed
    pol = SafetyPolicy(allow_commands=True, allowed_commands=["echo", "format", "icacls"])
    c2 = Computer(pol, confirm=lambda a: True, audit_path=audit, enabled=True)
    assert c2.run_allowed_command(["echo", "hi"]).ok is True
    assert c2.run_allowed_command(["format", "C:"]).ok is False
    assert c2.run_allowed_command(["icacls", "C:\\", "/grant", "Everyone:F"]).ok is False
    assert c2.run_allowed_command(
        ["powershell", "Set-MpPreference", "-DisableRealtimeMonitoring", "$true"]).ok is False
    assert c2.open_url("file:///etc/passwd").ok is False
