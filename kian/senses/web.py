"""
kian.senses.web
==============
Let Kian *browse* — read the text of a web page.

:func:`read_page` fetches an ``http(s)`` URL (using ``requests`` if present,
else the standard-library ``urllib``), strips the HTML to readable text, and
returns a bounded snippet. It is read-only: it fetches and parses, it does not
submit forms, run scripts, or authenticate. Size and time are capped. Fetched
text that will be learned from or fed to the model should be treated as
untrusted external content by the caller.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urlparse

MAX_BYTES = 2_000_000
TIMEOUT = 15.0


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            t = data.strip()
            if t:
                self.parts.append(t)


def _fetch(url: str) -> str:
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Kian/0.1"}, stream=True)
        r.raise_for_status()
        content = r.raw.read(MAX_BYTES, decode_content=True)
        enc = r.encoding or "utf-8"
        return content.decode(enc, errors="replace")
    except ImportError:
        from urllib.request import Request, urlopen
        req = Request(url, headers={"User-Agent": "Kian/0.1"})
        with urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 (scheme checked below)
            return resp.read(MAX_BYTES).decode("utf-8", errors="replace")


def read_page(url: str, max_chars: int = 4000) -> str:
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        return f"[refused] only http/https URLs are allowed (got '{scheme}')."
    try:
        html = _fetch(url)
    except Exception as e:
        return f"[error] could not fetch {url}: {e}"
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    text = re.sub(r"\s+\n", "\n", "\n".join(parser.parts))
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text[:max_chars].strip()
