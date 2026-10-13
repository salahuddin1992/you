"""Sorting algorithms, from scratch. Each returns a new sorted list and does
not mutate its input, so they compose cleanly as tools."""

from __future__ import annotations

from typing import Callable, List, Sequence, TypeVar

T = TypeVar("T")
Key = Callable[[T], object]


def _id(x):
    return x


def quicksort(seq: Sequence[T], key: Key = _id) -> List[T]:
    """Average O(n log n). Median-of-three pivot to avoid worst case on sorted
    input; recursion depth bounded by sorting the smaller side first."""
    a = list(seq)

    def part(lo: int, hi: int) -> None:
        while lo < hi:
            mid = (lo + hi) // 2
            # median-of-three pivot
            trio = sorted((lo, mid, hi), key=lambda i: key(a[i]))
            p = trio[1]
            a[p], a[hi] = a[hi], a[p]
            pivot = key(a[hi])
            i = lo
            for j in range(lo, hi):
                if key(a[j]) < pivot:
                    a[i], a[j] = a[j], a[i]
                    i += 1
            a[i], a[hi] = a[hi], a[i]
            # recurse into the smaller partition, loop on the larger
            if i - lo < hi - i:
                part(lo, i - 1)
                lo = i + 1
            else:
                part(i + 1, hi)
                hi = i - 1

    part(0, len(a) - 1)
    return a


def merge_sort(seq: Sequence[T], key: Key = _id) -> List[T]:
    """Stable O(n log n), O(n) extra space."""
    a = list(seq)
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left, right = merge_sort(a[:mid], key), merge_sort(a[mid:], key)
    out: List[T] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if key(right[j]) < key(left[i]):
            out.append(right[j]); j += 1
        else:                                  # <= keeps it stable
            out.append(left[i]); i += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out


def heap_sort(seq: Sequence[T], key: Key = _id) -> List[T]:
    """In-place-style O(n log n) via a binary max-heap."""
    a = list(seq)
    n = len(a)

    def sift(root: int, size: int) -> None:
        while True:
            largest, l, r = root, 2 * root + 1, 2 * root + 2
            if l < size and key(a[l]) > key(a[largest]):
                largest = l
            if r < size and key(a[r]) > key(a[largest]):
                largest = r
            if largest == root:
                return
            a[root], a[largest] = a[largest], a[root]
            root = largest

    for i in range(n // 2 - 1, -1, -1):
        sift(i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift(0, end)
    return a


def insertion_sort(seq: Sequence[T], key: Key = _id) -> List[T]:
    """Stable O(n^2); fast for small or nearly-sorted input."""
    a = list(seq)
    for i in range(1, len(a)):
        cur, j = a[i], i - 1
        while j >= 0 and key(a[j]) > key(cur):
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = cur
    return a
