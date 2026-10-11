"""
kian.draw.canvas
===============
A tiny raster drawing engine (pure NumPy) and a generative drawer.

Kian can *draw*: :class:`Canvas` provides anti-aliased primitives (fills,
gradients, circles, rectangles, triangles, lines) rendered into an RGB array
and saved as PNG via :mod:`kian.util.imageio` — no graphics library required.

:func:`draw_from_text` turns a text prompt into a deterministic generative
composition: the prompt seeds the palette and the arrangement, and if a trained
being is supplied, its language-model embedding of the prompt nudges the colours
so the same words drawn by a more-developed Kian look different. :func:`draw_face`
renders the being's "form" — an orb whose colour and liveliness reflect its
developmental stage.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Tuple

import numpy as np

from ..util.imageio import save_png


def _seed_from_text(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


class Canvas:
    def __init__(self, width: int = 512, height: int = 512, bg=(18, 20, 28)) -> None:
        self.w, self.h = width, height
        self.img = np.zeros((height, width, 3), dtype=np.float32)
        self.img[:] = np.array(bg, np.float32)
        yy, xx = np.mgrid[0:height, 0:width]
        self._yy = yy.astype(np.float32)
        self._xx = xx.astype(np.float32)

    # -- fills ------------------------------------------------------- #
    def gradient(self, top, bottom) -> "Canvas":
        t = (self._yy / max(1, self.h - 1))[..., None]
        self.img = (1 - t) * np.array(top, np.float32) + t * np.array(bottom, np.float32)
        return self

    def radial(self, inner, outer, cx=None, cy=None) -> "Canvas":
        cx = self.w / 2 if cx is None else cx
        cy = self.h / 2 if cy is None else cy
        r = np.sqrt((self._xx - cx) ** 2 + (self._yy - cy) ** 2)
        r = (r / r.max())[..., None]
        self.img = (1 - r) * np.array(inner, np.float32) + r * np.array(outer, np.float32)
        return self

    # -- primitives (anti-aliased via coverage) ---------------------- #
    def _blend(self, mask: np.ndarray, color, alpha: float = 1.0) -> None:
        m = (mask * alpha)[..., None]
        self.img = self.img * (1 - m) + np.array(color, np.float32) * m

    def circle(self, cx, cy, radius, color, alpha=1.0, feather=1.5) -> "Canvas":
        d = np.sqrt((self._xx - cx) ** 2 + (self._yy - cy) ** 2)
        mask = np.clip((radius - d) / feather, 0, 1)
        self._blend(mask, color, alpha)
        return self

    def ring(self, cx, cy, radius, thickness, color, alpha=1.0) -> "Canvas":
        d = np.sqrt((self._xx - cx) ** 2 + (self._yy - cy) ** 2)
        mask = np.clip(1 - np.abs(d - radius) / (thickness / 2), 0, 1)
        self._blend(mask, color, alpha)
        return self

    def rect(self, x0, y0, x1, y1, color, alpha=1.0) -> "Canvas":
        mask = ((self._xx >= x0) & (self._xx < x1) & (self._yy >= y0) & (self._yy < y1)).astype(np.float32)
        self._blend(mask, color, alpha)
        return self

    def triangle(self, p0, p1, p2, color, alpha=1.0) -> "Canvas":
        def edge(a, b):
            return (self._xx - a[0]) * (b[1] - a[1]) - (self._yy - a[1]) * (b[0] - a[0])
        w0, w1, w2 = edge(p1, p2), edge(p2, p0), edge(p0, p1)
        inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        self._blend(inside.astype(np.float32), color, alpha)
        return self

    def line(self, x0, y0, x1, y1, color, width=2.0, alpha=1.0) -> "Canvas":
        dx, dy = x1 - x0, y1 - y0
        length2 = dx * dx + dy * dy + 1e-6
        t = np.clip(((self._xx - x0) * dx + (self._yy - y0) * dy) / length2, 0, 1)
        px, py = x0 + t * dx, y0 + t * dy
        dist = np.sqrt((self._xx - px) ** 2 + (self._yy - py) ** 2)
        mask = np.clip((width - dist) / 1.5, 0, 1)
        self._blend(mask, color, alpha)
        return self

    def save(self, path: str) -> str:
        return save_png(np.clip(self.img, 0, 255).astype(np.uint8), path)

    def array(self) -> np.ndarray:
        return np.clip(self.img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------- #
# Palettes
# ---------------------------------------------------------------------- #
def _hsv_to_rgb(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return [int(r * 255), int(g * 255), int(b * 255)]


def palette_from(seed: int, n: int = 5, bias: float = 0.0):
    rng = np.random.default_rng(seed)
    base = (rng.random() + bias) % 1.0
    return [_hsv_to_rgb((base + k / n + rng.random() * 0.05) % 1.0,
                        0.55 + 0.4 * rng.random(), 0.6 + 0.35 * rng.random()) for k in range(n)]


# ---------------------------------------------------------------------- #
# Generative drawing
# ---------------------------------------------------------------------- #
def draw_from_text(prompt: str, out_path: str = "drawing.png", being=None,
                   size: int = 512) -> str:
    seed = _seed_from_text(prompt or "kian")
    rng = np.random.default_rng(seed)
    bias = 0.0
    if being is not None:
        try:
            emb = being.mind.embed(prompt or "kian")
            bias = float(abs(np.tanh(emb.mean())))   # being's "taste" shifts the hue
        except Exception:
            bias = 0.0
    pal = palette_from(seed, n=6, bias=bias)
    c = Canvas(size, size)
    c.gradient(pal[0], [max(0, x - 40) for x in pal[1]])
    n_shapes = 10 + int(rng.integers(0, 24))
    for _ in range(n_shapes):
        col = pal[int(rng.integers(0, len(pal)))]
        kind = rng.integers(0, 4)
        a = 0.25 + 0.55 * rng.random()
        if kind == 0:
            c.circle(rng.integers(0, size), rng.integers(0, size),
                     rng.integers(size // 20, size // 4), col, alpha=a)
        elif kind == 1:
            x0, y0 = rng.integers(0, size), rng.integers(0, size)
            c.rect(x0, y0, x0 + rng.integers(20, size // 3), y0 + rng.integers(20, size // 3), col, alpha=a)
        elif kind == 2:
            pts = rng.integers(0, size, size=(3, 2))
            c.triangle(pts[0], pts[1], pts[2], col, alpha=a)
        else:
            c.ring(rng.integers(0, size), rng.integers(0, size),
                   rng.integers(size // 12, size // 3), rng.integers(4, 20), col, alpha=a)
    return c.save(out_path)


STAGE_HUE = {"newborn": 0.55, "infant": 0.5, "child": 0.35,
             "adolescent": 0.15, "adult": 0.08, "sage": 0.75}


def draw_face(being, out_path: str = "face.png", size: int = 512, mood: float = 0.6) -> str:
    """Render Kian's 'form': an orb whose colour/energy reflects its stage."""
    stage = getattr(being, "stage", "newborn")
    hue = STAGE_HUE.get(stage, 0.55)
    glow = _hsv_to_rgb(hue, 0.6, 0.9)
    deep = _hsv_to_rgb(hue, 0.8, 0.25)
    c = Canvas(size, size, bg=deep)
    c.radial([int(x * 0.5) for x in glow], deep)
    cx = cy = size / 2
    # halo rings — more rings = more developed
    stages = ("newborn", "infant", "child", "adolescent", "adult", "sage")
    rings = 2 + (stages.index(stage) if stage in stages else 0)
    for r in range(rings):
        c.ring(cx, cy, size * (0.2 + 0.05 * r), 3, glow, alpha=0.5 - 0.05 * r)
    c.circle(cx, cy, size * 0.18, glow, alpha=0.9)
    c.circle(cx, cy, size * 0.18, [255, 255, 255], alpha=0.15)
    # two "eyes" to read as a face/form
    eye = [10, 12, 18]
    ox = size * 0.06
    c.circle(cx - ox, cy - size * 0.02, size * 0.035, eye, alpha=0.9)
    c.circle(cx + ox, cy - size * 0.02, size * 0.035, eye, alpha=0.9)
    # mood as a curved mouth (line segments)
    for t in np.linspace(-1, 1, 24):
        mx = cx + t * size * 0.08
        my = cy + size * 0.07 + (mood - 0.5) * (t * t) * size * 0.12
        c.circle(mx, my, 3, eye, alpha=0.8)
    return c.save(out_path)
