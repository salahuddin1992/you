"""
kian.util.imageio
================
Dependency-free image input/output.

Kian must be able to *see* and *draw* on a bare machine, so this module can
read and write PNG images using nothing but the Python standard library
(``zlib`` + ``struct``). If Pillow happens to be installed it is used to widen
the set of readable formats (JPEG, etc.), but it is never required.

* :func:`save_png`  — write an ``(H, W, 3|4)`` uint8 array as a PNG.
* :func:`load_image` — read PNG (always) or any Pillow-supported format, to an
  ``(H, W, 3)`` uint8 RGB array.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

try:  # optional, only broadens *reading*
    from PIL import Image as _PILImage  # type: ignore
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


# ---------------------------------------------------------------------- #
# Writing
# ---------------------------------------------------------------------- #
def save_png(array: np.ndarray, path: str | Path) -> str:
    """Write ``array`` (H,W,3 or H,W,4, uint8) to ``path`` as PNG."""
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w, c = arr.shape
    if c == 3:
        color_type = 2
    elif c == 4:
        color_type = 6
    else:
        raise ValueError("array must have 3 or 4 channels")

    # Each scanline is prefixed with a filter-type byte (0 = none).
    raw = bytearray()
    stride = w * c
    flat = arr.reshape(h, stride)
    for y in range(h):
        raw.append(0)
        raw.extend(flat[y].tobytes())
    compressed = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", compressed)
           + chunk(b"IEND", b""))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(png)
    return str(p)


# ---------------------------------------------------------------------- #
# Reading
# ---------------------------------------------------------------------- #
def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_png(path: Path) -> np.ndarray:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos = 8
    width = height = bit_depth = color_type = 0
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
            interlace = chunk[12] if len(chunk) > 12 else 0
            if interlace:
                # Adam7 lays IDAT out as 7 sub-images; this plain scanline
                # decoder would read garbage. Fail clearly so load_image can
                # fall back to Pillow (if installed) instead of returning noise.
                raise ValueError("interlaced (Adam7) PNG is not supported by the built-in decoder; install Pillow")
        elif tag == b"IDAT":
            idat.extend(chunk)
        elif tag == b"IEND":
            break
    if bit_depth != 8 or color_type not in (2, 6, 0):
        raise ValueError(f"unsupported PNG (bit_depth={bit_depth}, color_type={color_type})")
    channels = {0: 1, 2: 3, 6: 4}[color_type]
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int16)
    i = 0
    for y in range(height):
        ftype = raw[i]; i += 1
        line = np.frombuffer(raw[i:i + stride], dtype=np.uint8).astype(np.int16)
        i += stride
        rec = np.zeros(stride, dtype=np.int16)
        if ftype == 0:
            rec = line.copy()
        elif ftype == 1:  # Sub
            for x in range(stride):
                left = rec[x - channels] if x >= channels else 0
                rec[x] = (line[x] + left) & 0xFF
        elif ftype == 2:  # Up
            rec = (line + prev) & 0xFF
        elif ftype == 3:  # Average
            for x in range(stride):
                left = rec[x - channels] if x >= channels else 0
                rec[x] = (line[x] + ((left + prev[x]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for x in range(stride):
                left = rec[x - channels] if x >= channels else 0
                ul = prev[x - channels] if x >= channels else 0
                rec[x] = (line[x] + _paeth(left, prev[x], ul)) & 0xFF
        else:
            raise ValueError(f"unknown PNG filter {ftype}")
        out[y] = rec
        prev = rec
    img = out.reshape(height, width, channels)
    if channels == 1:
        img = np.repeat(img, 3, axis=2)
    return img[:, :, :3].astype(np.uint8)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image to an (H, W, 3) uint8 RGB array."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
        return np.clip(arr, 0, 255).astype(np.uint8)[:, :, :3]
    if _HAS_PIL and suffix not in (".png",):
        return np.asarray(_PILImage.open(p).convert("RGB"), dtype=np.uint8)
    if suffix == ".png":
        try:
            return _decode_png(p)
        except Exception:
            if _HAS_PIL:
                return np.asarray(_PILImage.open(p).convert("RGB"), dtype=np.uint8)
            raise
    if _HAS_PIL:
        return np.asarray(_PILImage.open(p).convert("RGB"), dtype=np.uint8)
    raise ValueError(f"cannot read {suffix} without Pillow installed")
