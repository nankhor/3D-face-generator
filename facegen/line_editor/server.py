"""Local web page for drawing jaw lines and clicking anchors on the photos.

Runs on 127.0.0.1 only. Lines are saved to the project's work/lines.json:

    {"photo.jpg": {"jaw": [[[x, y], ...], ...],      # polylines, original photo pixels
                   "beard": [...], "hairline": [...],
                   "anchors": {"nose_tip": [x, y], "ear_left": [x, y], ...}}}
"""

from __future__ import annotations

import io
import json
import mimetypes
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from .. import landmarks as L

STATIC = Path(__file__).parent / "static"
OVERLAY_LANDMARKS = sorted(set(L.FACE_OVAL + L.EYE_R + L.EYE_L + L.LIPS_OUTER + L.BROW_R + L.BROW_L +
                               [L.NOSE_TIP, 168, 6, 197, 195, 5, 2, 98, 327]))


class EditorState:
    def __init__(self, project):
        self.project = project
        self._thumbs: dict[tuple[str, int], bytes] = {}
        self._sizes: dict[str, tuple[int, int]] = {}
        self.lock = threading.Lock()

    def size(self, name: str) -> tuple[int, int]:
        if name not in self._sizes:
            from ..photos import photo_size

            self._sizes[name] = photo_size(self.project.photos_dir / name)
        return self._sizes[name]

    def image(self, name: str, max_side: int) -> bytes:
        key = (name, max_side)
        if key not in self._thumbs:
            from PIL import Image

            from ..photos import load_rgb, thumbnail

            img = thumbnail(load_rgb(self.project.photos_dir / name), max_side)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, "JPEG", quality=88)
            self._thumbs[key] = buf.getvalue()
        return self._thumbs[key]

    def state(self) -> dict:
        lms = {p.photo: p.points is not None for p in self.project.landmarks()}
        try:
            cams = set(self.project.cameras().cameras)
        except Exception:
            cams = set()
        lines = self.project.lines()
        photos = []
        for name in self.project.photos():
            w, h = self.size(name)
            entry = lines.get(name, {})
            photos.append({"name": name, "width": w, "height": h, "face": lms.get(name),
                           "camera": name in cams, "jaw": len(entry.get("jaw", [])),
                           "anchors": sorted(entry.get("anchors", {}))})
        return {"project": self.project.name, "photos": photos, "anchors": list(L.ANCHORS)}

    def overlay(self, name: str) -> dict:
        out: dict = {"landmarks": [], "jaw": []}
        for p in self.project.landmarks():
            if p.photo == name and p.points is not None:
                out["landmarks"] = np.round(p.points[OVERLAY_LANDMARKS], 1).tolist()
        try:
            sol = self.project.cameras()
            if name in sol.cameras:
                from ..fitting import JawCurve

                head = self.project.head()
                X = self.project.fitted() if self.project.path("fit.npz").exists() else head.vertices
                jaw = JawCurve(head).positions(X, head.faces)
                px, _ = sol.cameras[name].project(jaw)
                out["jaw"] = np.round(px, 1).tolist()
        except Exception:
            pass
        return out


def _handler(state: EditorState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the MCP server's stderr quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            path = unquote(url.path)
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path.startswith("/static/"):
                    f = (STATIC / path[len("/static/"):]).resolve()
                    if STATIC.resolve() not in f.parents or not f.is_file():
                        return self._send(404, b"not found", "text/plain")
                    return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
                if path == "/api/state":
                    return self._json(state.state())
                if path == "/api/lines":
                    return self._json(state.project.lines())
                if path.startswith("/api/photo/"):
                    name = path[len("/api/photo/"):]
                    if name not in state.project.photos():
                        return self._send(404, b"no such photo", "text/plain")
                    q = parse_qs(url.query)
                    max_side = int(q.get("max", ["1600"])[0])
                    return self._send(200, state.image(name, max(200, min(max_side, 4000))), "image/jpeg")
                if path.startswith("/api/overlay/"):
                    return self._json(state.overlay(path[len("/api/overlay/"):]))
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": str(exc)}, 500)
            self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/lines":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", "0"))
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("expected an object")
                with state.lock:
                    state.project.save_lines(data)
                return self._json({"ok": True})
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": str(exc)}, 400)

    return Handler


class LineEditor:
    def __init__(self, project, port: int = 0):
        self.state = EditorState(project)
        self.server = ThreadingHTTPServer(("127.0.0.1", port or _free_port()), _handler(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="facegen-editor")

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self, open_browser: bool = True) -> str:
        self.thread.start()
        if open_browser:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass
        return self.url

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
