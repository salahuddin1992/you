"""
kian.evolve.feed
===============
Feed Kian new material to learn from. Anything dropped into the inbox is ingested
by the next daily-evolution cycle (and then archived into the corpus), so this is
the practical way to grow the being's fluency on *your* texts.

* :func:`feed_text` — write a raw string as a lesson.
* :func:`feed_file` — copy a local ``.txt``/``.md`` file into the inbox.
* :func:`feed_url`  — fetch a web page (read-only) and save its text as a lesson.

Web content is untrusted external text; it is stored for training only, never
executed.
"""

from __future__ import annotations

import re
import secrets
import time
from pathlib import Path
from urllib.parse import urlparse


def _safe_name(stem: str) -> str:
    """Unique, sortable filename: nanosecond timestamp + random suffix, so two
    lessons fed within the same second (or the same instant) never collide."""
    stem = re.sub(r"[^\w\-.]+", "_", stem)[:60].strip("_") or "lesson"
    return f"{time.time_ns()}_{secrets.token_hex(3)}_{stem}.txt"


def feed_text(text: str, inbox: str | Path = "data/inbox", name: str = "lesson") -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    d = Path(inbox); d.mkdir(parents=True, exist_ok=True)
    path = d / _safe_name(name)
    path.write_text(text, encoding="utf-8")
    return str(path)


def feed_file(src: str | Path, inbox: str | Path = "data/inbox") -> str:
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)
    text = src.read_text(encoding="utf-8", errors="replace")
    return feed_text(text, inbox, name=src.stem)


def feed_url(url: str, inbox: str | Path = "data/inbox", max_chars: int = 20000) -> str:
    from ..senses.web import read_page
    text = read_page(url, max_chars=max_chars)
    if text.startswith("[error]") or text.startswith("[refused]"):
        raise RuntimeError(text)
    host = urlparse(url).netloc or "web"
    return feed_text(f"# source: {url}\n{text}", inbox, name=host)


def feed(target: str, inbox: str | Path = "data/inbox") -> str:
    """Dispatch: a URL is fetched, anything else is treated as a file path."""
    if re.match(r"^https?://", target, re.I):
        return feed_url(target, inbox)
    return feed_file(target, inbox)
