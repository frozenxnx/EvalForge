"""Access rules shared by the FastAPI app and the stdlib server."""
from __future__ import annotations

import hmac
from typing import Optional

MAX_BODY = 10 * 1024 * 1024                     # largest request body either server will read
# Reachable without the API key: the health probe, and the viewer page itself (static HTML, no data in it;
# the page then sends the key with every API call it makes).
OPEN_PATHS = frozenset({"/healthz", "/", "/index.html"})


MAX_INFLATED = 200 * 1024 * 1024               # a gzip body may not expand past this


class BodyError(ValueError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def decode_body(body: bytes, content_encoding: Optional[str]) -> bytes:
    """Undo `Content-Encoding: gzip` (experiment uploads are sent compressed), refusing a gzip bomb."""
    enc = (content_encoding or "identity").strip().lower()
    if enc == "identity":
        return body
    if enc != "gzip":
        raise BodyError(415, f"unsupported Content-Encoding {enc!r} (use gzip or none)")
    import zlib
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = d.decompress(body, MAX_INFLATED + 1)
    except zlib.error:
        raise BodyError(400, "body is not valid gzip") from None
    if len(out) > MAX_INFLATED or d.unconsumed_tail:
        raise BodyError(413, "decompressed body too large")
    return out


def key_ok(authorization: Optional[str], api_key: Optional[str]) -> bool:
    """True when no key is configured, or the Authorization header carries it. Constant-time comparison."""
    if not api_key:
        return True
    return hmac.compare_digest((authorization or "").encode("utf-8"), f"Bearer {api_key}".encode("utf-8"))
