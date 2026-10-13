"""
kian.algorithms
==============
A library of classic algorithms, implemented from scratch, that Kian can use as
tools for reasoning and problem-solving. This is the "has all the algorithms"
faculty: sorting, searching, graphs, dynamic programming, number theory, and
classic (non-neural) machine learning — the complement to the neural brain.

Every function is pure, documented, and covered by tests.
"""
from . import sort, search, graph, dp, numbers, ml  # noqa: F401

__all__ = ["sort", "search", "graph", "dp", "numbers", "ml"]
