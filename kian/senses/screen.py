"""
kian.senses.screen
=================
Let Kian *see* the computer screen.

:func:`grab_screen` captures a screenshot (via ``mss`` or ``pyautogui`` if
available) as an RGB array; :func:`describe_screen` runs the classical vision
analysis from :mod:`kian.vision.image` over it and returns a bilingual
description. This is strictly read-only perception — no input is sent, nothing
is changed. If no screenshot backend is installed the functions return a clear
message instead of raising.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def grab_screen() -> Optional[np.ndarray]:
    # Prefer mss (fast, cross-platform).
    try:
        import mss  # type: ignore
        with mss.mss() as sct:
            mon = sct.monitors[1]
            shot = np.asarray(sct.grab(mon))[:, :, :3][:, :, ::-1]  # BGRA->RGB
            return np.ascontiguousarray(shot)
    except Exception:
        pass
    try:
        import pyautogui  # type: ignore
        return np.asarray(pyautogui.screenshot().convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def describe_screen(being=None) -> str:
    from ..vision.image import analyze, describe
    img = grab_screen()
    if img is None:
        return ("لا يمكنني رؤية الشاشة الآن (ثبّت mss أو pyautogui). "
                "| I can't see the screen yet (install mss or pyautogui).")
    # Downsample large screens for speed.
    h, w = img.shape[:2]
    scale = max(1, int(max(h, w) / 480))
    small = img[::scale, ::scale]
    return describe(analyze(small))
