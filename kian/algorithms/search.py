"""Search algorithms: binary search on sorted sequences, and uninformed /
informed graph search (BFS, DFS, Dijkstra, A*). Graph callables take a
``neighbors`` function so they work on any graph representation."""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

Node = Hashable


def binary_search(seq: Sequence, target, key=lambda x: x) -> int:
    """Return an index of ``target`` in sorted ``seq``, or -1. O(log n)."""
    lo, hi = 0, len(seq) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        v = key(seq[mid])
        if v == target:
            return mid
        if v < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def bisect_left(seq: Sequence, target, key=lambda x: x) -> int:
    """Leftmost insertion point keeping ``seq`` sorted. O(log n)."""
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if key(seq[mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def bfs(start: Node, neighbors: Callable[[Node], List[Node]], goal: Node) -> Optional[List[Node]]:
    """Shortest path by edge count on an unweighted graph."""
    q = deque([start])
    came: Dict[Node, Optional[Node]] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            return _reconstruct(came, goal)
        for nxt in neighbors(cur):
            if nxt not in came:
                came[nxt] = cur
                q.append(nxt)
    return None


def dfs(start: Node, neighbors: Callable[[Node], List[Node]], goal: Node) -> Optional[List[Node]]:
    """A path (not necessarily shortest) via iterative depth-first search."""
    stack = [start]
    came: Dict[Node, Optional[Node]] = {start: None}
    while stack:
        cur = stack.pop()
        if cur == goal:
            return _reconstruct(came, goal)
        for nxt in neighbors(cur):
            if nxt not in came:
                came[nxt] = cur
                stack.append(nxt)
    return None


def dijkstra(start: Node, weighted_neighbors: Callable[[Node], List[Tuple[Node, float]]],
             goal: Node) -> Tuple[Optional[List[Node]], float]:
    """Shortest path on a non-negative weighted graph. Returns (path, cost)."""
    dist: Dict[Node, float] = {start: 0.0}
    came: Dict[Node, Optional[Node]] = {start: None}
    tie = itertools.count()  # heap entries are (cost, tie, node): nodes are never compared
    pq: List[Tuple[float, int, Node]] = [(0.0, next(tie), start)]
    while pq:
        d, _, cur = heapq.heappop(pq)
        if cur == goal:
            return _reconstruct(came, goal), d
        if d > dist.get(cur, float("inf")):
            continue
        for nxt, w in weighted_neighbors(cur):
            nd = d + w
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                came[nxt] = cur
                heapq.heappush(pq, (nd, next(tie), nxt))
    return None, float("inf")


def astar(start: Node, weighted_neighbors: Callable[[Node], List[Tuple[Node, float]]],
          goal: Node, heuristic: Callable[[Node], float]) -> Tuple[Optional[List[Node]], float]:
    """A* search with an admissible ``heuristic`` toward ``goal``."""
    g: Dict[Node, float] = {start: 0.0}
    came: Dict[Node, Optional[Node]] = {start: None}
    tie = itertools.count()
    pq: List[Tuple[float, int, Node]] = [(heuristic(start), next(tie), start)]
    while pq:
        _, _, cur = heapq.heappop(pq)
        if cur == goal:
            return _reconstruct(came, goal), g[cur]
        for nxt, w in weighted_neighbors(cur):
            ng = g[cur] + w
            if ng < g.get(nxt, float("inf")):
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(pq, (ng + heuristic(nxt), next(tie), nxt))
    return None, float("inf")


def _reconstruct(came: Dict[Node, Optional[Node]], goal: Node) -> List[Node]:
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    return path[::-1]
