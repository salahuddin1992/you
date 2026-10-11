"""
kian.vision.image
================
Classical image understanding: colour, brightness, edges and shapes.

This is how Kian *sees* without a trained network — deterministic algorithms
over pixels that answer "what colours, how bright, how busy, how many regions".
The trainable convolutional network in :mod:`kian.vision.conv` builds on top of
these perceptions. Everything is NumPy; colours are named in Arabic and English.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..util.imageio import load_image

# Named reference colours (RGB) with Arabic + English labels.
NAMED_COLORS: List[Tuple[str, str, Tuple[int, int, int]]] = [
    ("أسود", "black", (0, 0, 0)),
    ("أبيض", "white", (255, 255, 255)),
    ("رمادي", "gray", (128, 128, 128)),
    ("أحمر", "red", (220, 30, 30)),
    ("أخضر", "green", (30, 180, 60)),
    ("أزرق", "blue", (40, 70, 220)),
    ("أصفر", "yellow", (240, 220, 40)),
    ("برتقالي", "orange", (240, 140, 20)),
    ("بنفسجي", "purple", (140, 50, 200)),
    ("وردي", "pink", (240, 120, 180)),
    ("بني", "brown", (120, 70, 30)),
    ("سماوي", "cyan", (40, 200, 220)),
]


def color_name(rgb) -> Tuple[str, str]:
    rgb = np.asarray(rgb, dtype=np.float32)[:3]
    best = min(NAMED_COLORS, key=lambda c: np.sum((np.array(c[2], np.float32) - rgb) ** 2))
    return best[0], best[1]


# ---------------------------------------------------------------------- #
# K-means for dominant colours (from scratch)
# ---------------------------------------------------------------------- #
def dominant_colors(img: np.ndarray, k: int = 4, iters: int = 10, seed: int = 0):
    pixels = img.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 20000:  # subsample large images for speed
        idx = np.random.default_rng(seed).choice(len(pixels), 20000, replace=False)
        pixels = pixels[idx]
    rng = np.random.default_rng(seed)
    centers = pixels[rng.choice(len(pixels), k, replace=False)].copy()
    labels = np.zeros(len(pixels), dtype=np.int64)
    for _ in range(iters):
        d = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # (N,k)
        labels = d.argmin(1)
        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = pixels[m].mean(0)
    counts = np.bincount(labels, minlength=k).astype(np.float32)
    order = np.argsort(-counts)
    total = counts.sum()
    return [(centers[j].astype(int).tolist(), float(counts[j] / total)) for j in order]


# ---------------------------------------------------------------------- #
# Edges & shapes
# ---------------------------------------------------------------------- #
def to_gray(img: np.ndarray) -> np.ndarray:
    return (img[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32))


def sobel_edges(gray: np.ndarray) -> np.ndarray:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = kx.T
    g = np.pad(gray, 1, mode="edge")
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            gx += kx[i, j] * g[i:i + gray.shape[0], j:j + gray.shape[1]]
            gy += ky[i, j] * g[i:i + gray.shape[0], j:j + gray.shape[1]]
    return np.sqrt(gx * gx + gy * gy)


def count_regions(mask: np.ndarray, min_size: int = 30) -> int:
    """Count connected foreground regions with iterative flood fill (4-conn)."""
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    regions = 0
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] and not visited[sy, sx]:
                stack = [(sy, sx)]
                visited[sy, sx] = True
                size = 0
                while stack:
                    y, x = stack.pop()
                    size += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if size >= min_size:
                    regions += 1
    return regions


# ---------------------------------------------------------------------- #
# Full analysis + bilingual description
# ---------------------------------------------------------------------- #
def analyze(img: np.ndarray) -> dict:
    img = np.asarray(img)[:, :, :3].astype(np.uint8)
    h, w = img.shape[:2]
    gray = to_gray(img)
    brightness = float(gray.mean() / 255.0)
    edges = sobel_edges(gray)
    edge_density = float((edges > 60).mean())
    # Downsample for region counting to keep flood fill fast.
    scale = max(1, int(max(h, w) / 128))
    small = gray[::scale, ::scale]
    fg = small > (small.mean())
    regions = count_regions(fg) + count_regions(~fg)
    colors = dominant_colors(img, k=4)
    named = [{"rgb": c, "fraction": round(f, 3),
              "ar": color_name(c)[0], "en": color_name(c)[1]} for c, f in colors]
    return {
        "size": [w, h],
        "brightness": round(brightness, 3),
        "edge_density": round(edge_density, 3),
        "regions": int(regions),
        "dominant_colors": named,
    }


def describe(info: dict) -> str:
    bright = info["brightness"]
    blvl_ar = "مضيئة" if bright > 0.6 else "معتمة" if bright < 0.35 else "متوسطة الإضاءة"
    blvl_en = "bright" if bright > 0.6 else "dark" if bright < 0.35 else "medium-lit"
    busy_ar = "مزدحمة بالتفاصيل" if info["edge_density"] > 0.12 else "بسيطة وهادئة"
    busy_en = "detailed/busy" if info["edge_density"] > 0.12 else "simple/calm"
    top = info["dominant_colors"][0]
    cols_ar = "، ".join(c["ar"] for c in info["dominant_colors"][:3])
    cols_en = ", ".join(c["en"] for c in info["dominant_colors"][:3])
    return (
        f"صورة {info['size'][0]}×{info['size'][1]}، {blvl_ar}، {busy_ar}، "
        f"حوالي {info['regions']} منطقة، ألوانها الغالبة: {cols_ar} (الأبرز {top['ar']}). "
        f"| A {info['size'][0]}x{info['size'][1]} image, {blvl_en}, {busy_en}, "
        f"~{info['regions']} regions, dominant colours: {cols_en} (mostly {top['en']})."
    )


def analyze_image(path_or_array) -> str:
    img = load_image(path_or_array) if isinstance(path_or_array, str) else np.asarray(path_or_array)
    return describe(analyze(img))
