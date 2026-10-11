"""
kian.app.server
==============
The local web server that gives Kian a *face* — the "specific form" (شكل معين)
it is asked to have.

It serves a single-page UI (``templates/index.html``) that renders an animated
living orb/face in the browser and talks to a small JSON API backed by one
in-process :class:`~kian.being.Being`. The server uses the standard library
only (``http.server``), so ``python -m kian live`` works with nothing installed
beyond NumPy. A lock serialises access to the being because the NumPy brain is
single-threaded.

API
---
  GET  /                  the face UI
  GET  /api/vitals        vital signs
  POST /api/chat          {text}        -> {reply, vitals}
  POST /api/evolve        {steps}       -> {record, vitals}
  POST /api/draw          {prompt}      -> {image} (data URL)
  GET  /api/face          -> {image} (data URL, server-rendered orb)
  POST /api/see           {}            -> {description}
"""

from __future__ import annotations

import base64
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_TEMPLATE = Path(__file__).parent / "templates" / "index.html"


class KianApp:
    def __init__(self, home: str) -> None:
        from ..being import Being
        self.lock = threading.Lock()
        self.being = Being.load_or_birth(home)

    # -- actions (all under the lock) -------------------------------- #
    def vitals(self) -> dict:
        with self.lock:
            return self.being.vitals()

    def chat(self, text: str) -> dict:
        with self.lock:
            reply = self.being.respond(text, max_new_tokens=100, temperature=0.85)
            return {"reply": reply, "vitals": self.being.vitals()}

    def evolve(self, steps: int = 60) -> dict:
        from ..evolve.daily_loop import Evolver
        with self.lock:
            ev = Evolver(self.being, log=lambda m: None)
            rec = ev.evolve_once(steps=max(10, min(steps, 400)))
            return {"record": rec, "vitals": self.being.vitals()}

    def draw(self, prompt: str) -> dict:
        from ..draw.canvas import draw_from_text
        import tempfile
        with self.lock:
            path = draw_from_text(prompt, tempfile.mktemp(suffix=".png"), being=self.being)
        return {"image": _data_url(path)}

    def face(self) -> dict:
        from ..draw.canvas import draw_face
        import tempfile
        with self.lock:
            path = draw_face(self.being, tempfile.mktemp(suffix=".png"), size=320)
        return {"image": _data_url(path)}

    def see(self) -> dict:
        with self.lock:
            return {"description": self.being.see_screen()}


def _data_url(png_path: str) -> str:
    data = Path(png_path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _make_handler(app: KianApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except Exception:
                return {}

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = _TEMPLATE.read_text(encoding="utf-8")
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/vitals":
                self._json(app.vitals())
            elif self.path == "/api/face":
                self._json(app.face())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            data = self._read_json()
            try:
                if self.path == "/api/chat":
                    self._json(app.chat(str(data.get("text", "")).strip()))
                elif self.path == "/api/evolve":
                    self._json(app.evolve(int(data.get("steps", 60))))
                elif self.path == "/api/draw":
                    self._json(app.draw(str(data.get("prompt", ""))))
                elif self.path == "/api/see":
                    self._json(app.see())
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": str(e)}, 500)

    return Handler


def run_server(home: str, host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    app = KianApp(home)
    handler = _make_handler(app)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"🫧 Kian is alive at {url}  (stage: {app.being.stage})")
    print("   Press Ctrl-C to let it sleep.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n   saving and sleeping…")
        app.being.save()
        httpd.shutdown()
