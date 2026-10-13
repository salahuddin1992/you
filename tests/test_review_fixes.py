"""Regression tests for the six Codex review findings on PR #2."""
import math
import time

import numpy as np
import pytest


# 1) calculator resource bounds ---------------------------------------- #
@pytest.mark.parametrize("expr", ["9**999999999", "2^100000", "factorial(100000)",
                                  "(10**1000)**100", "9" * 300 + "+1"])
def test_calc_refuses_resource_bombs_quickly(expr):
    from kian.mind import tools
    t0 = time.time()
    with pytest.raises(tools.CalcError):
        tools.calculate(expr)
    assert time.time() - t0 < 0.5


def test_calc_still_allows_reasonable_powers():
    from kian.mind import tools
    assert tools.calculate("2**64") == 2 ** 64
    assert tools.calculate("factorial(20)") == math.factorial(20)
    assert tools.calculate("(-1)**1001") == -1 and tools.calculate("0**5") == 0


# 2) nucleus sampling keeps the crossing token --------------------------- #
def test_top_p_keeps_crossing_token():
    from kian.language.sampling import filter_top_p
    logits = np.log(np.array([0.6, 0.3, 0.1]))
    kept = np.isfinite(filter_top_p(logits.copy(), 0.8))
    assert kept.tolist() == [True, True, False]
    # p=1 keeps everything; tiny p keeps only the top token
    assert np.isfinite(filter_top_p(logits.copy(), 0.999999)).sum() == 3
    assert np.isfinite(filter_top_p(logits.copy(), 0.01)).sum() == 1


# 3) perplexity is a true per-token mean ---------------------------------- #
def test_perplexity_weights_by_tokens(tmp_path):
    from kian.being import Being
    from kian.evolve.metrics import perplexity
    b = Being.birth(tmp_path / "b", vocab_size=320, n_layer=1, n_head=2, n_embd=32, block_size=8)
    text = "abc def ghi jkl mno pqr stu vwx yz"  # encodes to a non-multiple of block_size
    ids = b.lm.tokenizer.encode(text)
    assert (len(ids) - 1) % 8 != 0
    # manual token-weighted reference
    b.lm.model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(ids) - 1, 8):
        chunk = ids[i:i + 9]
        if len(chunk) < 2:
            break
        x = np.array([chunk[:-1]]); y = np.array([chunk[1:]])
        tot += b.lm.model.loss(x, y).item() * (len(chunk) - 1)
        n += len(chunk) - 1
    assert abs(perplexity(b.lm, text) - math.exp(tot / n)) < 1e-3


# 4) feed never overwrites a lesson fed in the same second ------------------ #
def test_feed_same_name_same_second_keeps_both(tmp_path):
    from kian.evolve.feed import feed_text
    inbox = tmp_path / "inbox"
    p1 = feed_text("first lesson", inbox=inbox)      # default name
    p2 = feed_text("second lesson", inbox=inbox)     # same default name, same second
    assert p1 != p2
    assert len(list(inbox.glob("*.txt"))) == 2


# 5) chat returns the drawing it just made ---------------------------------- #
def test_chat_returns_draw_image(tmp_path):
    from kian.being import Being
    from kian.app.server import KianApp
    Being.birth(tmp_path / "b", vocab_size=320, n_layer=1, n_head=2, n_embd=32, block_size=32)
    app = KianApp(str(tmp_path / "b"))
    r = app.chat("ارسم شجرة")
    assert r["action"] == "draw"
    assert r.get("image", "").startswith("data:image/png;base64,")


# 6) Dijkstra / A* never compare nodes ------------------------------------- #
def test_search_mixed_unorderable_nodes():
    from kian.algorithms.search import dijkstra, astar
    # equal-cost successors of mixed types: an int and a str
    graph = {"s": [(1, 1.0), ("a", 1.0)], 1: [("goal", 1.0)], "a": [("goal", 1.0)], "goal": []}
    nbrs = lambda n: graph.get(n, [])
    path, cost = dijkstra("s", nbrs, "goal")
    assert path is not None and cost == 2.0
    path2, cost2 = astar("s", nbrs, "goal", heuristic=lambda n: 0.0)
    assert path2 is not None and cost2 == 2.0
