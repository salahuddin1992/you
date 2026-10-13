"""A small graph data structure and classic graph algorithms:
topological sort, connected components, cycle detection, and minimum spanning
tree (Kruskal with union-find)."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Hashable, List, Optional, Set, Tuple

Node = Hashable


class Graph:
    def __init__(self, directed: bool = False) -> None:
        self.directed = directed
        self.adj: Dict[Node, List[Tuple[Node, float]]] = defaultdict(list)
        self.nodes: Set[Node] = set()

    def add_edge(self, u: Node, v: Node, weight: float = 1.0) -> "Graph":
        self.adj[u].append((v, weight))
        self.nodes.update((u, v))
        if not self.directed:
            self.adj[v].append((u, weight))
        else:
            self.adj.setdefault(v, self.adj[v])
        return self

    def neighbors(self, u: Node) -> List[Node]:
        return [v for v, _ in self.adj[u]]

    def weighted_neighbors(self, u: Node) -> List[Tuple[Node, float]]:
        return list(self.adj[u])


def topological_sort(g: Graph) -> Optional[List[Node]]:
    """Kahn's algorithm. Returns an ordering, or None if the graph has a cycle."""
    if not g.directed:
        raise ValueError("topological sort requires a directed graph")
    indeg: Dict[Node, int] = {n: 0 for n in g.nodes}
    for u in g.nodes:
        for v, _ in g.adj[u]:
            indeg[v] += 1
    q = deque(sorted((n for n in g.nodes if indeg[n] == 0), key=repr))
    order: List[Node] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v, _ in g.adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == len(g.nodes) else None


def connected_components(g: Graph) -> List[List[Node]]:
    """Undirected connected components via BFS."""
    seen: Set[Node] = set()
    comps: List[List[Node]] = []
    for start in g.nodes:
        if start in seen:
            continue
        comp, q = [], deque([start])
        seen.add(start)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in g.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(sorted(comp, key=repr))
    return comps


def has_cycle(g: Graph) -> bool:
    if g.directed:
        return topological_sort(g) is None
    seen: Set[Node] = set()
    for start in g.nodes:
        if start in seen:
            continue
        stack = [(start, None)]
        while stack:
            u, parent = stack.pop()
            if u in seen:
                return True
            seen.add(u)
            for v in g.neighbors(u):
                if v != parent:
                    stack.append((v, u))
    return False


class _DSU:
    def __init__(self):
        self.p: Dict[Node, Node] = {}

    def find(self, x: Node) -> Node:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: Node, b: Node) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def minimum_spanning_tree(g: Graph) -> Tuple[List[Tuple[Node, Node, float]], float]:
    """Kruskal's MST for an undirected weighted graph. Returns (edges, cost)."""
    # Key each undirected edge by (weight, sorted reprs) so both adjacency
    # directions collapse to one entry and sorting never compares raw nodes
    # (which may be unorderable or of mixed types).
    edges: Dict[Tuple[float, str, str], Tuple[Node, Node]] = {}
    for u in g.nodes:
        for v, w in g.adj[u]:
            key = (w, *sorted((repr(u), repr(v))))
            edges.setdefault(key, (u, v))
    dsu = _DSU()
    mst: List[Tuple[Node, Node, float]] = []
    total = 0.0
    for key in sorted(edges):
        w = key[0]
        u, v = edges[key]
        if dsu.union(u, v):
            mst.append((u, v, w))
            total += w
    return mst, total
