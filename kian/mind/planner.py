"""
kian.mind.planner
================
A light intent router — the step where Kian *decides what to do* with a request
instead of only continuing text. It inspects the input and dispatches to the
faculty that fits: the calculator, memory (store/recall), the algorithms library
(sorting, primes), vision, drawing, or — by default — the language brain.

The planner returns a structured ``Plan`` (the chosen action, a human-readable
answer, and any data) so callers (CLI, UI) can show *how* Kian thought, not just
what it said. Routing is intentionally transparent and rule-based; the language
model remains the fallback for everything open-ended.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from . import tools

# bilingual trigger words
_DRAW = ("ارسم", "رسمة", "ارسمي", "draw", "paint", "sketch")
_SEE = ("انظر", "شاهد", "الشاشة", "ماذا ترى", "see screen", "look at screen", "screenshot")
_REMEMBER = ("تذكر", "احفظ", "تذكّر", "remember that", "remember:", "note that")
_RECALL = ("ماذا تتذكر", "بماذا تتذكر", "اذكر", "استرجع", "what do you remember", "recall")
_SORT = ("رتب", "ترتيب", "sort")
_PRIMES = ("أولية", "اولية", "primes", "prime number")


@dataclass
class Plan:
    action: str                 # calc | remember | recall | sort | primes | draw | see | talk
    answer: str                 # human-readable result/reply
    data: Any = None            # structured payload (numbers, path, …)
    thought: str = ""           # one-line trace of the decision
    meta: dict = field(default_factory=dict)


def _nums_in(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", tools.normalize(text))]


def plan(text: str, being=None) -> Optional[Plan]:
    """Decide the action for ``text``. Returns None to mean 'use the language
    model' (so the caller falls back to free generation)."""
    t = text.strip()
    low = t.lower()

    def has(words):
        return any(w in t or w in low for w in words)

    # 1) memory: recall (checked before store: "تتذكر" contains "تذكر")
    if has(_RECALL) and being is not None:
        q = re.split(r"ماذا تتذكر|بماذا تتذكر|اذكر|استرجع|what do you remember|recall",
                     t, maxsplit=1, flags=re.I)[-1].strip(" :،,؟?") or t
        hits = being.mind.recall(q, k=3)
        body = "؛ ".join(hits) if hits else "لا شيء بعد · nothing yet"
        return Plan("recall", f"أتذكّر: {body}", data=hits, thought="semantic memory recall")

    # 2) memory: store
    if has(_REMEMBER):
        fact = re.split(r"تذكّر|تذكر|احفظ|remember that|remember:|note that",
                        t, maxsplit=1, flags=re.I)[-1].strip(" :،,")
        if fact and being is not None:
            being.remember(fact, importance=2.0)
            return Plan("remember", f"حفظتُها: {fact} · Noted: {fact}", data=fact,
                        thought="stored a fact in long-term memory")

    # 3) calculator — whole string, or an expression embedded in a sentence
    expr = t if tools.looks_like_math(t) else tools.extract_expression(t)
    if expr and not has(_SORT) and not has(_PRIMES):
        try:
            val = tools.calculate(expr)
            return Plan("calc", f"{expr.strip()} = {val}", data=val,
                        thought="parsed arithmetic and evaluated it safely")
        except tools.CalcError:
            pass  # not really math; fall through

    # 4) sort numbers
    if has(_SORT):
        nums = _nums_in(t)
        if len(nums) >= 2:
            from ..algorithms import sort as S
            s = S.merge_sort(nums)
            return Plan("sort", f"مرتبة · sorted: {s}", data=s, thought="merge_sort on the numbers")

    # 5) primes up to n
    if has(_PRIMES):
        nums = _nums_in(t)
        if nums:
            from ..algorithms import numbers as N
            primes = N.sieve(int(nums[0]))
            return Plan("primes", f"الأعداد الأولية حتى {int(nums[0])}: {primes}", data=primes,
                        thought="Sieve of Eratosthenes")

    # 6) draw
    if has(_DRAW):
        subject = t
        for w in _DRAW:
            subject = re.sub(w, "", subject, flags=re.I)
        subject = subject.strip(" :،,") or "فن تجريدي abstract art"
        if being is not None:
            import tempfile
            path = being.draw(subject, tempfile.mktemp(suffix=".png"))
            return Plan("draw", f"رسمتُ: {subject}", data=path, thought="generative drawing from the prompt")

    # 7) see the screen
    if has(_SEE) and being is not None:
        desc = being.see_screen()
        return Plan("see", desc, data=desc, thought="captured and analysed the screen")

    # default: let the language model answer
    return None
