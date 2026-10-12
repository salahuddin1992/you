"""
kian.mind.tools
==============
Safe reasoning tools the mind can invoke. Currently: an arithmetic calculator.

The calculator parses an expression into a Python AST and walks it, allowing
only numeric literals, arithmetic operators, parentheses, a fixed set of math
functions, and the constants pi/e. It never calls ``eval``/``exec`` and never
touches names, attributes, calls to arbitrary functions, comprehensions, or
anything else — so it cannot execute code. Arabic-Indic and Persian digits are
normalised first, so "٣ + ٤" works.
"""

from __future__ import annotations

import ast
import math
import operator as op
from typing import Dict

# Map Arabic-Indic (U+0660..) and Extended/Persian (U+06F0..) digits to ASCII.
_DIGIT_MAP = {ord("٠") + i: str(i) for i in range(10)}
_DIGIT_MAP.update({ord("۰") + i: str(i) for i in range(10)})
# common Arabic math symbols
_SYMBOL_MAP = {"×": "*", "÷": "/", "،": ",", "−": "-"}

_BIN = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
}
_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil, "round": round,
    "factorial": math.factorial, "gcd": math.gcd, "pow": pow,
    "min": min, "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


class CalcError(ValueError):
    pass


def normalize(expr: str) -> str:
    expr = expr.translate(_DIGIT_MAP)
    for a, b in _SYMBOL_MAP.items():
        expr = expr.replace(a, b)
    return expr.replace("^", "**").strip()


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalcError("only numbers are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise CalcError(f"unknown name '{node.id}'")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise CalcError("only whitelisted math functions are allowed")
        if node.keywords:
            raise CalcError("keyword arguments are not allowed")
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise CalcError("unsupported expression")


def calculate(expr: str) -> float:
    """Evaluate an arithmetic expression safely. Raises CalcError on anything
    outside the arithmetic whitelist."""
    cleaned = normalize(expr)
    if not cleaned:
        raise CalcError("empty expression")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"could not parse: {e.msg}") from None
    result = _eval(tree)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


# A permissive detector: does this text look like a math expression?
import re
_MATH_RE = re.compile(r"^[\s\d٠-٩۰-۹\.\+\-\*\/\%\^\(\)×÷،eπ,a-z]+$", re.IGNORECASE)
_HAS_OP = re.compile(r"[\+\-\*\/\^×÷%]|sqrt|sin|cos|tan|log|exp|factorial")


def looks_like_math(text: str) -> bool:
    t = normalize(text)
    return bool(t) and bool(_MATH_RE.match(t)) and bool(_HAS_OP.search(t)) and any(
        c.isdigit() for c in t)


_EXPR_RUN = re.compile(r"[0-9\.\s\+\-\*\/%\(\)]+")


def extract_expression(text: str) -> str | None:
    """Pull the longest embedded arithmetic run out of a sentence, e.g.
    'كم يساوي 12 * (3 + 4)؟' -> '12 * (3 + 4)'. Requires an operator between
    digits so plain number lists ('رتب 5 2 9') are not treated as math."""
    t = normalize(text)
    best = None
    for m in _EXPR_RUN.finditer(t):
        s = m.group().strip()
        if len(s) >= 3 and _HAS_OP.search(s) and any(c.isdigit() for c in s):
            if best is None or len(s) > len(best):
                best = s
    return best
