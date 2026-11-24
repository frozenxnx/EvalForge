"""The whole API on the standard library: no FastAPI, no uvicorn.

    python -m llmobs_server.dev_server --db llmobs.db

It serves the same routes as the FastAPI app (see app.py / experiments_api.py) and the web UI, and it is what the
test-suite talks to over real HTTP. `Api.handle` is a pure function of (method, path, query, body), so it is
testable without sockets too.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from llmobs_eval.compare import compare_payload, examples_payload
from llmobs_eval.judgecheck import calibration_payload, judge_check_payload
from llmobs_eval.patterns import failure_patterns
from llmobs_eval.store import ExperimentStore, ImportConflict
from llmobs_eval.summary import experiment_overview

from .auth import MAX_BODY, OPEN_PATHS, BodyError, decode_body, key_ok
from .store import SpanStore

STATIC = Path(__file__).parent / "static"
_REQUIRED = ("trace_id", "span_id", "name", "start_ns")


class HttpError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def _num(q: Dict[str, str], name: str, kind: Callable, default: Any, lo: Any = None, hi: Any = None) -> Any:
    if name not in q or q[name] == "":
        return default
    try:
        v = kind(q[name])
    except ValueError:
        raise HttpError(422, f"query parameter {name!r} must be a {kind.__name__}") from None
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise HttpError(422, f"query parameter {name!r} must be between {lo} and {hi}")
    return v


def _bool(q: Dict[str, str], name: str, default: bool) -> bool:
    if name not in q:
        return default
    return q[name].strip().lower() in ("1", "true", "yes", "on")


class Api:
    def __init__(self, spans: SpanStore, experiments: ExperimentStore, api_key: Optional[str] = None) -> None:
        self.spans, self.experiments, self.api_key = spans, experiments, api_key
        self.routes: List[Tuple[str, "re.Pattern[str]", Callable]] = [
            ("GET", re.compile(r"^/healthz$"), self._health),
            ("POST", re.compile(r"^/v1/spans$"), self._ingest),
            ("POST", re.compile(r"^/v1/experiments/import$"), self._import),
            ("GET", re.compile(r"^/v1/traces$"), self._traces),
            ("GET", re.compile(r"^/v1/traces/([^/]+)$"), self._trace),
            ("GET", re.compile(r"^/v1/experiments$"), self._experiments),
            ("GET", re.compile(r"^/v1/experiments/([^/]+)$"), self._experiment),
            ("GET", re.compile(r"^/v1/experiments/([^/]+)/compare$"), self._compare),
            ("GET", re.compile(r"^/v1/experiments/([^/]+)/examples$"), self._examples),
            ("GET", re.compile(r"^/v1/experiments/([^/]+)/judge-check$"), self._judge_check),
            ("GET", re.compile(r"^/v1/experiments/([^/]+)/calibration$"), self._calibration),
            ("GET", re.compile(r"^/v1/experiments/([^/]+)/failure-patterns$"), self._patterns),
        ]

    # -- dispatch -----------------------------------------------------------
    def handle(self, method: str, target: str, body: bytes = b"", headers: Optional[Dict[str, str]] = None
               ) -> Tuple[int, str, bytes]:
        """Returns (status, content-type, body)."""
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        url = urlparse(target)
        path = url.path.rstrip("/") or "/"
        q = {k: v[-1] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
        try:
            if method == "GET" and path in ("/", "/index.html"):
                return 200, "text/html; charset=utf-8", (STATIC / "viewer.html").read_bytes()
            for m, pattern, fn in self.routes:
                match = pattern.match(path)
                if match and m == method:
                    if path not in OPEN_PATHS:
                        self._auth(headers)
                    if method == "POST":
                        body = decode_body(body, headers.get("content-encoding"))
                    return 200 if method == "GET" else 202, "application/json", self._json(
                        fn(q, body, *[unquote(g) for g in match.groups()]))
            if any(p.match(path) for _, p, _ in self.routes):
                raise HttpError(405, "method not allowed")
            raise HttpError(404, "not found")
        except (HttpError, BodyError) as exc:
            return exc.status, "application/json", self._json({"detail": exc.detail})
        except ImportConflict as exc:
            return 409, "application/json", self._json({"detail": str(exc)})
        except KeyError as exc:
            return 404, "application/json", self._json({"detail": str(exc.args[0]) if exc.args else "not found"})
        except ValueError as exc:                        # ConfigError and bad arguments are ValueErrors
            return 400, "application/json", self._json({"detail": str(exc)})

    @staticmethod
    def _json(obj: Any) -> bytes:
        return json.dumps(obj, default=str).encode("utf-8")

    def _auth(self, headers: Dict[str, str]) -> None:
        if not key_ok(headers.get("authorization"), self.api_key):
            raise HttpError(401, "invalid API key")

    # -- handlers (query, body, *path groups) ---------------------------------
    def _health(self, q, body):
        return {"ok": True}

    def _import(self, q, body):
        try:
            payload = json.loads(body or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HttpError(400, "body must be JSON") from None
        return self.experiments.import_experiment(payload)

    def _ingest(self, q, body):
        if len(body) > MAX_BODY:
            raise HttpError(413, "request body too large")
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            raise HttpError(400, "body must be JSON") from None
        spans = data.get("spans") if isinstance(data, dict) else None
        if not isinstance(spans, list) or len(spans) > 1000:
            raise HttpError(422, "body must be {\"spans\": [...]} with at most 1000 spans")
        for i, s in enumerate(spans):
            if not isinstance(s, dict) or any(k not in s for k in _REQUIRED):
                raise HttpError(422, f"span {i} is missing one of {list(_REQUIRED)}")
        return {"accepted": self.spans.ingest(spans)}

    def _traces(self, q, body):
        return {"traces": self.spans.list_traces(_num(q, "limit", int, 50, 1, 200), _num(q, "offset", int, 0, 0),
                                                 q.get("service") or None)}

    def _trace(self, q, body, trace_id):
        spans = self.spans.get_trace(trace_id)
        if not spans:
            raise HttpError(404, "trace not found")
        return {"trace_id": trace_id, "spans": spans}

    def _experiments(self, q, body):
        return {"experiments": self.experiments.list_experiments(_num(q, "limit", int, 50, 1, 200))}

    def _experiment(self, q, body, exp_id):
        return experiment_overview(self.experiments, exp_id)

    def _compare(self, q, body, exp_id):
        return compare_payload(self.experiments, exp_id, q.get("baseline") or None, q.get("metric") or None,
                               n_boot=1000, top_k=_num(q, "top_k", int, 5, 1, 50))

    def _examples(self, q, body, exp_id):
        return examples_payload(self.experiments, exp_id, q.get("metric") or None, q.get("sort") or "worst",
                                _num(q, "limit", int, 25, 1, 200), _num(q, "offset", int, 0, 0))

    def _judge_check(self, q, body, exp_id):
        return judge_check_payload(self.experiments, exp_id, q.get("metric") or None, _num(q, "top_k", int, 15, 0, 100))

    def _calibration(self, q, body, exp_id):
        return calibration_payload(self.experiments, exp_id, q.get("metric") or None, n_boot=500)

    def _patterns(self, q, body, exp_id):
        labels = [v["label"] for v in self.experiments.get_experiment(exp_id)["config"]["variants"]]
        baseline = q.get("baseline") or None
        variant = q.get("variant") or next((l for l in labels if l != (baseline or labels[0])), labels[0])
        return failure_patterns(self.experiments, exp_id, variant, baseline=baseline, metric=q.get("metric") or None,
                                threshold=_num(q, "threshold", float, 0.5, 0.0001, 1.0),
                                group_by=q.get("group_by") or "category", cluster=_bool(q, "cluster", True),
                                k=_num(q, "k", int, None, 2, 30))
