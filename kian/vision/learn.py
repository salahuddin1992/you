"""
kian.vision.learn
================
Teach Kian to recognise *your* images.

Point :func:`train_from_folder` at a directory laid out as::

    images/
      cat/    a.png b.jpg ...
      car/    ...
      tree/   ...

Each sub-folder name is a label. Images are loaded with the dependency-free
reader in :mod:`kian.util.imageio` (PNG always; JPEG etc. if Pillow is present),
resized to a small square, and used to train the from-scratch convolutional
network in :mod:`kian.vision.conv`. The resulting :class:`VisionClassifier`
is saved into the being's home (``vision/``) and, once present, is consulted
automatically by ``Being.see_image`` so Kian can say what it thinks it is
looking at — with a confidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..core.optim import AdamW
from ..util.imageio import load_image
from .conv import TinyConvNet

IMG_SIZE = 32
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".npy"}


def resize_nearest(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Nearest-neighbour resize of an (H, W, 3) image to (size, size, 3)."""
    h, w = img.shape[:2]
    ys = (np.arange(size) * h / size).astype(int).clip(0, h - 1)
    xs = (np.arange(size) * w / size).astype(int).clip(0, w - 1)
    return img[ys][:, xs]


def to_input(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """(H,W,3) uint8 -> (3,size,size) float32 in [0,1]."""
    small = resize_nearest(np.asarray(img)[:, :, :3], size)
    return (small.astype(np.float32) / 255.0).transpose(2, 0, 1)


def load_folder_dataset(root: str | Path, size: int = IMG_SIZE) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    root = Path(root)
    labels = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not labels:
        raise ValueError(f"no label sub-folders found under {root}")
    X: List[np.ndarray] = []
    y: List[int] = []
    for li, label in enumerate(labels):
        for f in sorted((root / label).iterdir()):
            if f.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                X.append(to_input(load_image(f), size))
                y.append(li)
            except Exception:
                continue  # skip unreadable files, keep going
    if not X:
        raise ValueError(f"no readable images under {root}")
    return np.stack(X), np.asarray(y, dtype=np.int64), labels


class VisionClassifier:
    def __init__(self, labels: List[str], width: int = 8, size: int = IMG_SIZE) -> None:
        self.labels = list(labels)
        self.width = width
        self.size = size
        self.net = TinyConvNet(3, len(self.labels), width=width)

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 40, batch_size: int = 8,
            lr: float = 5e-3, seed: int = 0, log=None) -> float:
        """Small batches on purpose: users' image folders are small, and the
        number of *gradient steps* (not epochs) is what the tiny convnet
        needs. 32 images as one batch gives ~25 steps and chance accuracy;
        batch 8 over 40 epochs gives ~160 steps and separates shapes."""
        rng = np.random.default_rng(seed)
        opt = AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        n = len(X)
        self.net.train()
        for ep in range(epochs):
            perm = rng.permutation(n)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                self.net.zero_grad()
                loss = self.net.loss(X[idx], y[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if log and (ep % 5 == 0 or ep == epochs - 1):
                log(f"  epoch {ep + 1}/{epochs} loss {total / n:.3f}")
        self.net.eval()
        return float((self.net.predict(X) == y).mean())

    def predict(self, img: np.ndarray) -> Tuple[str, float]:
        """Return (label, confidence) for an (H,W,3) image."""
        x = to_input(img, self.size)[None]
        self.net.eval()
        logits = self.net.forward(__import__("kian.core.tensor", fromlist=["Tensor"]).Tensor(x)).data[0]
        z = logits - logits.max()
        p = np.exp(z) / np.exp(z).sum()
        k = int(p.argmax())
        return self.labels[k], float(p[k])

    # ------------------------------------------------------------------ #
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "vision.npz", **{f"p{i}": p.data for i, p in enumerate(self.net.parameters())})
        (d / "labels.json").write_text(json.dumps(
            {"labels": self.labels, "width": self.width, "size": self.size}, ensure_ascii=False),
            encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> Optional["VisionClassifier"]:
        d = Path(directory)
        if not (d / "vision.npz").exists() or not (d / "labels.json").exists():
            return None
        meta = json.loads((d / "labels.json").read_text(encoding="utf-8"))
        clf = cls(meta["labels"], width=meta.get("width", 8), size=meta.get("size", IMG_SIZE))
        npz = np.load(d / "vision.npz")
        clf.net.load_state_dict([npz[f"p{i}"] for i in range(len(npz.files))])
        clf.net.eval()
        return clf


def train_from_folder(root: str | Path, out_dir: str | Path, epochs: int = 40,
                      width: int = 8, log=None) -> Tuple[VisionClassifier, float]:
    X, y, labels = load_folder_dataset(root)
    if log:
        log(f"  loaded {len(X)} images, {len(labels)} labels: {labels}")
    clf = VisionClassifier(labels, width=width)
    acc = clf.fit(X, y, epochs=epochs, log=log)
    clf.save(out_dir)
    return clf, acc
