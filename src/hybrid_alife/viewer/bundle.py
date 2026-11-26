"""Serve the viewer locally, or inline it with a replay into one offline HTML file."""

from __future__ import annotations

import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).with_name("web")
ASSETS = {
    "index.html": "text/html; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "viewer.js": "text/javascript; charset=utf-8",
}


def standalone(replay: str | Path, out: str | Path) -> Path:
    """Write a self-contained HTML file that opens from disk (no server, no network)."""
    html, css, script = ((WEB / name).read_text(encoding="utf-8") for name in ASSETS)
    data = base64.b64encode(Path(replay).read_bytes()).decode("ascii")
    for marker, replacement in (
        ('<link rel="stylesheet" href="style.css">', f"<style>\n{css}</style>"),
        ('<script src="viewer.js"></script>',
         f'<script>window.REPLAY_B64="{data}";</script>\n<script>\n{script}</script>'),
    ):
        if html.count(marker) != 1:
            raise RuntimeError(f"index.html must contain exactly one {marker!r}")
        html = html.replace(marker, replacement)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def make_server(replay: str | Path, port: int) -> ThreadingHTTPServer:
    """Loopback-only server exposing exactly the three assets and one replay."""
    replay = Path(replay)
    if not replay.is_file():
        raise FileNotFoundError(f"replay not found: {replay} (run the export command first)")
    routes = {f"/{name}": (WEB / name, kind) for name, kind in ASSETS.items()}
    routes["/"] = routes["/index.html"]
    routes["/replay.hvr"] = (replay, "application/octet-stream")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            route = routes.get(self.path.split("?", 1)[0])
            if route is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = route[0].read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", route[1])
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(replay: str | Path, port: int) -> None:
    with make_server(replay, port) as server:
        print(f"viewer: http://127.0.0.1:{server.server_address[1]}/  (Ctrl-C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
