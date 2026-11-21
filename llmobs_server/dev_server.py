"""Zero-dependency server for the trace + experiment UI:  python -m llmobs_server.dev_server --db llmobs.db"""
from __future__ import annotations

import argparse
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from llmobs_eval.store import ExperimentStore

from .stdlib_app import Api
from .store import SpanStore


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _respond(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            status, ctype, payload = api.handle(method, self.path, body, dict(self.headers.items()))
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._respond("GET")

        def do_POST(self):
            self._respond("POST")

        def log_message(self, fmt, *args):
            if os.environ.get("LLMOBS_ACCESS_LOG"):
                super().log_message(fmt, *args)
    return Handler


def serve(db: str, host: str = "127.0.0.1", port: int = 8000, api_key: Optional[str] = None) -> ThreadingHTTPServer:
    api = Api(SpanStore(db), ExperimentStore(db), api_key)
    return ThreadingHTTPServer((host, port), make_handler(api))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="llmobs_server.dev_server")
    p.add_argument("--db", default=os.environ.get("LLMOBS_DB", "llmobs.db"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=os.environ.get("LLMOBS_API_KEY"))
    args = p.parse_args(argv)
    srv = serve(args.db, args.host, args.port, args.api_key)
    print(f"llmobs UI on http://{args.host}:{srv.server_address[1]}   (db: {args.db})   Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
