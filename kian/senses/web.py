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


def _decode_bytes(data: bytes, content_type: str, declared: str | None) -> str:
    """Decode a response body without mangling UTF-8 pages.

    Order: an explicit ``charset=`` in the Content-Type header; else an HTML
    ``<meta charset>``; else strict UTF-8; else the transport's declared guess
    (requests defaults text/* to ISO-8859-1, which turns Arabic into mojibake —
    hence it is the LAST resort, not the first).
    """
    m = re.search(r"charset=([\w\-]+)", (content_type or "").lower())
    if m:
        try:
            return data.decode(m.group(1), errors="replace")
        except LookupError:
            pass
    head = data[:4096].decode("ascii", errors="ignore").lower()
    m = re.search(r'charset=["\']?([\w\-]+)', head)
    if m:
        try:
            return data.decode(m.group(1), errors="replace")
        except LookupError:
            pass
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(declared or "latin-1", errors="replace")


def _fetch(url: str) -> str:
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Kian/0.1"}, stream=True)
        r.raise_for_status()
        content = r.raw.read(MAX_BYTES, decode_content=True)
        return _decode_bytes(content, r.headers.get("content-type", ""), r.encoding)
    except ImportError:
        from urllib.request import Request, urlopen
        req = Request(url, headers={"User-Agent": "Kian/0.1"})
        with urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 (scheme checked below)
            return _decode_bytes(resp.read(MAX_BYTES), resp.headers.get("Content-Type", ""), None)


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
