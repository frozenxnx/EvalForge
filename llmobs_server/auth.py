"""Access rules shared by the FastAPI app and the stdlib server."""
from __future__ import annotations

import hmac
from typing import Optional

MAX_BODY = 10 * 1024 * 1024                     # largest request body either server will read
# Reachable without the API key: the health probe, and the viewer page itself (static HTML, no data in it;
# the page then sends the key with every API call it makes).
OPEN_PATHS = frozenset({"/healthz", "/", "/index.html"})


def key_ok(authorization: Optional[str], api_key: Optional[str]) -> bool:
    """True when no key is configured, or the Authorization header carries it. Constant-time comparison."""
    if not api_key:
        return True
    return hmac.compare_digest((authorization or "").encode("utf-8"), f"Bearer {api_key}".encode("utf-8"))
