"""Static frontend server with no-cache headers.

The browser must never reuse a stale app.js/style.css after an update —
a cached old JS was the reason uploads looked "stuck in processing".

Usage: python serve.py [port]
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent


class NoCacheHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    print(f"Serving web/ on http://localhost:{port} (no-cache)")
    ThreadingHTTPServer(("0.0.0.0", port), NoCacheHandler).serve_forever()
