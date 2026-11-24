"""Send an experiment from the local database to an llmobs server (POST /v1/experiments/import).

The CLI keeps writing to its own SQLite file while it works, so a run never depends on the network; the
finished experiment is then uploaded in one request. Uploading is idempotent: the server adds what it lacks,
keeps what it already has, and refuses (409) a conflicting history instead of overwriting it.
"""
from __future__ import annotations

import gzip
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .store import ExperimentStore


class PushError(RuntimeError):
    pass


def server_url(explicit: Optional[str]) -> Optional[str]:
    """--server wins, else $LLMOBS_SERVER; returned without a trailing slash, or None when neither is set."""
    url = (explicit or os.environ.get("LLMOBS_SERVER") or "").strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        raise PushError(f"server must be an http(s) URL, got {url!r}")
    return url or None


def push_experiment(store: ExperimentStore, exp_id: str, server: str, api_key: Optional[str] = None,
                    timeout: float = 60.0) -> Dict[str, Any]:
    body = gzip.compress(json.dumps(store.export_experiment(exp_id), default=str).encode("utf-8"))
    headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
    key = api_key if api_key is not None else os.environ.get("LLMOBS_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = server.rstrip("/") + "/v1/experiments/import"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail")
        except Exception:
            detail = None
        hint = {401: " (set LLMOBS_API_KEY to the server's key)",
                413: " (the experiment is larger than the server accepts in one request)"}.get(exc.code, "")
        raise PushError(f"{url}: HTTP {exc.code}: {detail or exc.reason}{hint}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PushError(f"cannot reach {url}: {getattr(exc, 'reason', exc)}") from None


def describe(result: Dict[str, Any]) -> str:
    names = {"labels": "labels new or changed"}
    parts = [f"{v} {names.get(k, k)}" for k, v in (result.get("added") or {}).items() if v]
    if result.get("hidden_metrics_changed"):
        parts.append("hidden metrics updated")
    if result.get("status_changed"):
        parts.append("status changed")
    what = ", ".join(parts) or "nothing new"
    verb = "created" if result.get("created") else "updated"
    return f"{verb} {result.get('experiment_id')} on the server ({what}; status {result.get('status')})"
