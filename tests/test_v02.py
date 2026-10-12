"""Tests for the v0.2 additions: the algorithms library, the safe calculator,
mood dynamics, and the intent planner."""
import random

import numpy as np
import pytest

from kian.algorithms import sort, search, graph, dp, numbers, ml
from kian.mind import tools
from kian.mind.mood import Mood


# --------------------------------------------------------------------- #
# Algorithms
# --------------------------------------------------------------------- #
def test_sorts():
    r = [random.randint(0, 99) for _ in range(150)]
    exp = sorted(r)
    assert sort.quicksort(r) == exp
    assert sort.merge_sort(r) == exp
    assert sort.heap_sort(r) == exp
    assert sort.insertion_sort(r) == exp


def test_search_and_graph_paths():
    arr = sorted(random.randint(0, 999) for _ in range(200))
    assert search.binary_search(arr, arr[50]) != -1
    assert search.binary_search(arr, -1) == -1

    def nbrs(p):
        x, y = p
        return [(a, b) for a, b in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] if 0 <= a < 5 and 0 <= b < 5]
    assert len(search.bfs((0, 0), nbrs, (4, 4))) == 9

    def wn(p):
        return [(n, 1.0) for n in nbrs(p)]
    _, cost = search.dijkstra((0, 0), wn, (4, 4))
    assert cost == 8.0
    _, ca = search.astar((0, 0), wn, (4, 4), lambda p: abs(p[0]-4)+abs(p[1]-4))
    assert ca == 8.0


def test_graph_algorithms():
    g = graph.Graph(directed=True)
    for u, v in [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")]:
        g.add_edge(u, v)
    topo = graph.topological_sort(g)
    assert topo.index("a") < topo.index("c") < topo.index("d")
    cyc = graph.Graph(directed=True)
    for u, v in [("x", "y"), ("y", "z"), ("z", "x")]:
        cyc.add_edge(u, v)
    assert graph.has_cycle(cyc) and graph.topological_sort(cyc) is None
    gm = graph.Graph()
    for a, b, w in [("a", "b", 1), ("b", "c", 2), ("a", "c", 5)]:
        gm.add_edge(a, b, w)
    assert graph.minimum_spanning_tree(gm)[1] == 3.0


def test_dp():
    assert dp.edit_distance("kitten", "sitting") == 3
    assert dp.longest_common_subsequence("ABCBDAB", "BDCAB") == list("BCAB")
    assert dp.knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7)[0] == 9.0
    assert dp.longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert dp.coin_change([1, 2, 5], 11) == 3


def test_numbers():
    assert numbers.gcd(54, 24) == 6 and numbers.lcm(4, 6) == 12
    assert numbers.sieve(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert numbers.prime_factors(360) == {2: 3, 3: 2, 5: 1}
    assert numbers.is_prime(97) and not numbers.is_prime(91)
    assert numbers.mod_pow(2, 10, 1000) == 24
    g, x, y = numbers.extended_gcd(240, 46)
    assert 240 * x + 46 * y == g == numbers.gcd(240, 46)


def test_classic_ml():
    np.random.seed(0)
    A = np.random.randn(40, 2) + [5, 5]
    B = np.random.randn(40, 2) + [-5, -5]
    X = np.vstack([A, B]); y = np.array([0]*40 + [1]*40)
    _, lab = ml.kmeans(X, 2)
    assert len(set(lab[:40])) == 1 and len(set(lab[40:])) == 1
    assert list(ml.knn_classify(X, y, np.array([[5, 5], [-5, -5]]), k=5)) == [0, 1]
    Xr = np.linspace(0, 10, 40).reshape(-1, 1)
    yr = 3 * Xr.ravel() + 2 + np.random.randn(40) * 0.05
    w = ml.LinearRegression().fit(Xr, yr).w
    assert abs(w[1] - 3) < 0.1
    assert (ml.LogisticRegression(lr=0.5, epochs=800).fit(X, y).predict(X) == y).mean() > 0.95


# --------------------------------------------------------------------- #
# Calculator
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("expr,val", [
    ("2+3*4", 14), ("(10-4)/2", 3), ("2^10", 1024), ("sqrt(144)+1", 13),
    ("٣ + ٤", 7), ("١٠ × ٥", 50), ("factorial(5)", 120), ("100 % 7", 2),
])
def test_calc_values(expr, val):
    assert tools.calculate(expr) == val


@pytest.mark.parametrize("bad", [
    "__import__('os')", "open('x')", "1 if True else 2", "[x for x in range(3)]", "a.b", "lambda: 1",
])
def test_calc_rejects_code(bad):
    with pytest.raises(tools.CalcError):
        tools.calculate(bad)


def test_math_detection():
    assert tools.looks_like_math("2+2") and not tools.looks_like_math("مرحبا")
    assert tools.extract_expression("كم يساوي 12 * (3 + 4)؟") == "12 * (3 + 4)"
    assert tools.extract_expression("رتب 5 2 9") is None  # no operator


# --------------------------------------------------------------------- #
# Mood
# --------------------------------------------------------------------- #
def test_mood():
    m = Mood()
    for _ in range(3):
        m.event("grew")
    assert m.valence > 0.5 and m.label()[1] in ("excited", "happy")
    bad = Mood()
    for _ in range(3):
        bad.event("negative")
    assert bad.valence < -0.3
    assert 0.6 <= m.temperature() <= 1.1
    assert Mood.from_dict(m.to_dict()).label() == m.label()


# --------------------------------------------------------------------- #
# Planner routing (uses a tiny being)
# --------------------------------------------------------------------- #
def test_planner_routing(tmp_path):
    from kian.being import Being
    b = Being.birth(tmp_path / "b", vocab_size=320, n_layer=1, n_head=2, n_embd=32, block_size=32)
    cases = {
        "كم يساوي 12 * (3 + 4)؟": "calc",
        "رتب هذه الأرقام 5 2 9 1 7": "sort",
        "أعطني الأعداد الأولية حتى 20": "primes",
        "تذكر أن اسم صاحبي صلاح الدين": "remember",
        "ماذا تتذكر عن صاحبي؟": "recall",
        "ارسم غابة خضراء": "draw",
    }
    for q, want in cases.items():
        assert b.act(q, max_new_tokens=4)["action"] == want
    # calc correctness through the being
    assert b.act("٢٥ + ٧٥", max_new_tokens=4)["answer"].endswith("100")
