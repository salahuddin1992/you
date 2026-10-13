"""Regression tests for the 19 findings of the adversarial audit (v0.6)."""
import math
import time

import numpy as np
import pytest

from kian.core.tensor import Tensor, stack


# 1) Linear / Conv2d weights are float32 and state_dict round-trips exactly --- #
def test_params_are_float32_and_roundtrip_exact():
    from kian.core.nn import Linear
    from kian.vision.conv import Conv2d
    from kian.language.model import GPT, GPTConfig
    assert Linear(4, 4).weight.data.dtype == np.float32
    assert Conv2d(3, 4).weight.data.dtype == np.float32
    np.random.seed(0)
    m = GPT(GPTConfig(vocab_size=50, n_embd=16, n_head=2, n_layer=2, block_size=8, dropout=0.0))
    assert all(p.data.dtype == np.float32 for p in m.parameters())
    x = np.random.randint(0, 50, (2, 8))
    ref = m.forward(x).data.copy()
    assert ref.dtype == np.float32                      # (6) whole stream stays float32
    m.load_state_dict(m.state_dict())
    assert np.array_equal(m.forward(x).data, ref)


# 6) numpy float64 scalars never promote a float32 stream --------------------- #
def test_numpy_scalar_operand_keeps_dtype():
    x = Tensor(np.ones((2, 3), np.float32), requires_grad=True)
    assert (x * np.float64(0.35)).data.dtype == np.float32
    assert (np.float64(2.0) - x).data.dtype == np.float32
    assert (x / np.float64(4.0)).data.dtype == np.float32


# 7) matmul backward with 1-D operands ---------------------------------------- #
def test_matmul_backward_1d():
    x = Tensor(np.arange(4, dtype=np.float64), requires_grad=True)
    W = Tensor(np.arange(12, dtype=np.float64).reshape(4, 3), requires_grad=True)
    (x @ W).sum().backward()
    assert np.allclose(x.grad, W.data.sum(1)) and np.allclose(W.grad, np.outer(x.data, np.ones(3)))
    A = Tensor(np.arange(6, dtype=np.float64).reshape(2, 3), requires_grad=True)
    v = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    (A @ v).sum().backward()
    assert np.allclose(v.grad, A.data.sum(0)) and np.allclose(A.grad, np.outer(np.ones(2), v.data))
    a = Tensor(np.array([1.0, 2.0]), requires_grad=True); b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    (a @ b).backward()
    assert np.allclose(a.grad, b.data) and np.allclose(b.grad, a.data)


# 8) stack with negative axis -------------------------------------------------- #
def test_stack_negative_axis():
    a = Tensor(np.ones((3, 4), np.float32)); b = Tensor(np.zeros((3, 4), np.float32))
    s = stack([a, b], axis=-1)
    assert s.shape == (3, 4, 2)
    assert np.array_equal(s.data, np.stack([a.data, b.data], axis=-1))


# 2) non-finite loss never reaches the weights --------------------------------- #
def test_trainer_stops_before_nan_update():
    from kian.language.tokenizer import Tokenizer
    from kian.language.lm import LanguageModel
    from kian.language.data import TextDataset
    from kian.language.trainer import Trainer, TrainConfig
    tok = Tokenizer().train("abc def " * 50, vocab_size=300)
    lm = LanguageModel.new(tok, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    before = [p.data.copy() for p in lm.model.parameters()]
    # poison the embedding so the first loss is NaN
    lm.model.tok_emb.weight.data[:] = np.nan
    ds = TextDataset.from_text("abc def " * 30, tok, block_size=8)
    Trainer(lm, TrainConfig(steps=3, batch_size=2, log=None, eval_every=100)).train(ds)
    # every non-embedding parameter must be untouched (not NaN)
    others = [p for p in lm.model.parameters() if p is not lm.model.tok_emb.weight]
    assert all(np.isfinite(p.data).all() for p in others)


# 3 + 10) calculator: pow() bounded, results bounded, no escaping exceptions --- #
def test_pow_function_is_bounded_and_results_capped():
    from kian.mind import tools
    with pytest.raises(tools.CalcError):
        tools.calculate("pow(9, 999999)")
    with pytest.raises(tools.CalcError):
        tools.calculate("10**1000 * 10**1000 * 10**1000 * 10**1000 * 10**1000")
    with pytest.raises(tools.CalcError):
        tools.calculate("pow(0, -1)")
    assert tools.calculate("pow(2, 10)") == 1024


def test_planner_never_crashes_on_calc_edge_cases():
    from kian.mind import planner
    for s in ["factorial(2000)", "pow(10, 5000) + 1", "pow(0, -1)", "9" * 300 + "+1"]:
        planner.plan(s, None)  # must not raise (returns a Plan or None)


# 11) English sentences with arithmetic reach the calculator ------------------- #
def test_english_sentence_math():
    from kian.mind import planner, tools
    assert not tools.looks_like_math("what is 12 * 3")
    p = planner.plan("what is 12 * 3", None)
    assert p is not None and p.action == "calc" and p.data == 36


# 12) 'do you remember' is recall, not store ---------------------------------- #
def test_remember_question_is_recall():
    from kian.mind import planner
    stored = []
    class Mind:
        def recall(self, q, k=3): return ["x"]
    class B:
        mind = Mind()
        def remember(self, fact, importance=1.5): stored.append(fact)
    b = B()
    assert planner.plan("هل تتذكر اسمي؟", b).action == "recall"
    assert planner.plan("Do you remember that I like tea?", b).action == "recall"
    assert planner.plan("تذكر أن اسمي صلاح", b).action == "remember"
    assert stored == ["أن اسمي صلاح"]


# 9) chat special tokens are produced by the training tokenizer ---------------- #
def test_special_tokens_encoded_from_text():
    from kian.language.tokenizer import Tokenizer
    tok = Tokenizer().train("hello world " * 40, vocab_size=300)
    ids = tok.encode("<user>hello<kian>hi<eos>")
    assert tok.special["<user>"] in ids and tok.special["<kian>"] in ids and tok.special["<eos>"] in ids
    assert tok.decode(ids) == "<user>hello<kian>hi<eos>"


# 13) archived lessons keep being trained on ---------------------------------- #
def test_learned_dir_is_gathered(tmp_path):
    from kian.being import Being
    from kian.evolve.daily_loop import Evolver
    corpus = tmp_path / "corpus"; corpus.mkdir()
    (corpus / "seed.txt").write_text("seed text", encoding="utf-8")
    (corpus / "learned").mkdir()
    (corpus / "learned" / "old_lesson.txt").write_text("UNIQUE_LEARNED_TOKEN", encoding="utf-8")
    b = Being.birth(tmp_path / "b", vocab_size=320, n_layer=1, n_head=2, n_embd=32, block_size=32)
    ev = Evolver(b, corpus_glob=str(corpus / "*.txt"), inbox_dir=str(tmp_path / "inbox"), log=lambda m: None)
    text, _ = ev._gather_text()
    assert "UNIQUE_LEARNED_TOKEN" in text


# 14) charset-less UTF-8 bodies decode correctly ------------------------------ #
def test_decode_bytes_utf8_without_charset():
    from kian.senses.web import _decode_bytes
    body = "<p>كيان يتعلم كل يوم</p>".encode("utf-8")
    assert "كيان" in _decode_bytes(body, "text/html", "ISO-8859-1")
    assert "كيان" in _decode_bytes(body, "text/html; charset=utf-8", None)
    meta = ('<meta charset="utf-8"><p>كيان</p>').encode("utf-8")
    assert "كيان" in _decode_bytes(meta, "text/html", "ISO-8859-1")


# 4 + 15) blocklist variants and typed text ----------------------------------- #
@pytest.mark.parametrize("argv", [
    ["rm", "-fr", "/home/me"], ["rm", "-r", "-f", "x"], ["rm", "--recursive", "--force", "x"],
    ["cmd", "/c", "rd", "/s", "/q", "C:\\x"], ["cmd", "/c", "del", "C:\\x\\*"],
    ["powershell", "Remove-Item", "-Recurse", "-Force", "C:\\x"], ["sh", "-c", "rm   -rf  /"],
])
def test_blocklist_spelling_variants(argv):
    from kian.senses.policy import SafetyPolicy
    p = SafetyPolicy(allow_commands=True, allowed_commands=["rm", "cmd", "powershell", "sh"])
    assert p.command_allowed(argv)[0] is False


def test_blocklist_allows_benign():
    from kian.senses.policy import SafetyPolicy
    p = SafetyPolicy(allow_commands=True, allowed_commands=["echo", "ls"])
    assert p.command_allowed(["echo", "hello world"])[0] is True
    assert p.command_allowed(["ls", "-la", "/tmp"])[0] is True


def test_typed_text_is_blocklisted(tmp_path):
    from kian.senses.policy import SafetyPolicy
    from kian.senses.computer import Computer
    c = Computer(SafetyPolicy(require_confirmation=False), audit_path=tmp_path / "a.log", enabled=True)
    r = c.type_text("rm -rf ~\n")
    assert r.ok is False and "blocklist" in r.detail


# 5) interlaced PNG fails clearly instead of decoding garbage ------------------ #
def test_interlaced_png_rejected(tmp_path):
    import struct, zlib
    from kian.util.imageio import _decode_png
    ihdr = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 1)   # interlace method = 1
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00" * 13 * 4)) + chunk(b"IEND", b"")
    p = tmp_path / "i.png"; p.write_bytes(png)
    with pytest.raises(ValueError, match="interlaced"):
        _decode_png(p)


# 16) MST with unorderable nodes and correct dedup ----------------------------- #
def test_mst_unorderable_nodes():
    from kian.algorithms.graph import Graph, minimum_spanning_tree
    class P:
        def __init__(s, n): s.n = n
    a, b, c = P(1), P(2), P(3)
    mst, total = minimum_spanning_tree(Graph().add_edge(a, b, 1.0).add_edge(b, c, 2.0).add_edge(a, c, 3.0))
    assert total == 3.0 and len(mst) == 2


# 17) k-means++ with fewer distinct rows than k -------------------------------- #
def test_kmeans_few_distinct_rows():
    from kian.algorithms import ml
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]], float)
    centers, labels = ml.kmeans(X, 3)
    assert centers.shape == (3, 2) and len(labels) == 5


# 18) last corpus token can be a target ---------------------------------------- #
def test_dataset_last_token_reachable():
    from kian.language.data import TextDataset
    ds = TextDataset(np.arange(20), block_size=4)
    rng = np.random.default_rng(0); seen = set()
    for _ in range(600):
        seen |= set(ds.batch(8, rng)[1].ravel().tolist())
    assert 19 in seen
    assert len(ds) == 16


# 19) CLI birth hint is a valid command ---------------------------------------- #
def test_birth_hint_parses():
    from kian.cli import build_parser
    args = build_parser().parse_args(["--home", "/tmp/x", "evolve"])
    assert args.command == "evolve" and args.home == "/tmp/x"
