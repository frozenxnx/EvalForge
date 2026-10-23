"""Core tracing primitives: Span, Tracer, the span() context manager and @trace."""
from __future__ import annotations

import contextvars
import dataclasses
import functools
import inspect
import logging
import math
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from .exporters import Exporter, NoopExporter

log = logging.getLogger("llmobs")

# Span kinds the viewer knows how to color: chain, llm, retrieval, tool, embedding
# Attribute names follow the OpenTelemetry GenAI semantic conventions where one exists.
ATTR_MODEL = "gen_ai.request.model"
ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

_current: "contextvars.ContextVar[Optional[Span]]" = contextvars.ContextVar(
    "llmobs_current_span", default=None
)


def _new_id(nbytes: int) -> str:
    return os.urandom(nbytes).hex()


def _safe(v: Any, max_str: int = 8000, depth: int = 0) -> Any:
    """Convert arbitrary Python values into truncated, JSON-safe structures."""
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        if len(v) <= max_str:
            return v
        return v[:max_str] + f"...[truncated {len(v) - max_str} chars]"
    if depth > 6:
        return repr(v)[:200]
    if isinstance(v, dict):
        return {str(k): _safe(x, max_str, depth + 1) for k, x in list(v.items())[:100]}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_safe(x, max_str, depth + 1) for x in list(v)[:100]]
    if hasattr(v, "model_dump"):  # pydantic v2
        try:
            return _safe(v.model_dump(), max_str, depth + 1)
        except Exception:
            pass
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        try:
            return _safe(dataclasses.asdict(v), max_str, depth + 1)
        except Exception:
            pass
    return repr(v)[:max_str]


@dataclasses.dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    name: str
    kind: str
    service: str
    start_ns: int
    end_ns: Optional[int] = None
    status: str = "ok"
    error: Optional[str] = None
    input: Any = None
    output: Any = None
    attributes: Dict[str, Any] = dataclasses.field(default_factory=dict)
    _t0: int = dataclasses.field(default=0, repr=False)  # monotonic start, for accurate durations

    # -- convenience recorders ---------------------------------------------
    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, attrs: Dict[str, Any]) -> None:
        self.attributes.update(attrs)

    def record_llm(self, model: str, input_tokens: Optional[int] = None,
                   output_tokens: Optional[int] = None, **extra: Any) -> None:
        self.kind = "llm" if self.kind == "chain" else self.kind
        self.attributes[ATTR_MODEL] = model
        if input_tokens is not None:
            self.attributes[ATTR_INPUT_TOKENS] = input_tokens
        if output_tokens is not None:
            self.attributes[ATTR_OUTPUT_TOKENS] = output_tokens
        self.attributes.update(extra)

    def record_retrieval(self, query: str, documents: List[Dict[str, Any]]) -> None:
        """documents: [{"id": ..., "text": ..., "score": ...}, ...] in rank order."""
        self.kind = "retrieval" if self.kind == "chain" else self.kind
        self.attributes["retrieval.query"] = query
        self.attributes["retrieval.documents"] = documents

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "service": self.service,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "status": self.status,
            "error": self.error,
            "input": _safe(self.input),
            "output": _safe(self.output),
            "attributes": _safe(self.attributes),
        }


class Tracer:
    def __init__(self, exporter: Optional[Exporter] = None, service: str = "app") -> None:
        self.exporter = exporter or NoopExporter()
        self.service = service

    @contextmanager
    def span(self, name: str, kind: str = "chain",
             attributes: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        parent = _current.get()
        s = Span(
            trace_id=parent.trace_id if parent else _new_id(16),
            span_id=_new_id(8),
            parent_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            service=self.service,
            start_ns=time.time_ns(),
            attributes=dict(attributes or {}),
            _t0=time.perf_counter_ns(),
        )
        token = _current.set(s)
        try:
            yield s
        except BaseException as exc:
            s.status = "error"
            s.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            s.end_ns = s.start_ns + (time.perf_counter_ns() - s._t0)
            _current.reset(token)
            self._emit(s)

    def _emit(self, s: Span) -> None:
        try:
            self.exporter.export(s.to_dict())
        except Exception:  # tracing must never break the application
            log.exception("llmobs: failed to export span %s", s.name)


# ---------------------------------------------------------------------------
# Module-level API (what applications import)
# ---------------------------------------------------------------------------
_tracer = Tracer()


def get_tracer() -> Tracer:
    return _tracer


def configure(endpoint: Optional[str] = None, service: str = "app",
              exporter: Optional[Exporter] = None, **exporter_kwargs: Any) -> Tracer:
    """Configure the global tracer. Pass `endpoint` for HTTP export or a custom `exporter`."""
    global _tracer
    if exporter is None and endpoint:
        from .exporters import HTTPExporter
        exporter = HTTPExporter(endpoint, **exporter_kwargs)
    _tracer = Tracer(exporter=exporter, service=service)
    return _tracer


def span(name: str, kind: str = "chain", **attributes: Any):
    """Context manager: `with span("rerank", kind="tool") as s: ...`"""
    return get_tracer().span(name, kind, attributes)


def current_span() -> Optional[Span]:
    """The active span, or None outside a trace. Prefer record_llm / record_retrieval for annotating it."""
    return _current.get()


def record_llm(model: str, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
               **extra: Any) -> None:
    """Record model and token usage on the current span. A no-op outside a trace, so instrumented
    code keeps working when called without tracing."""
    s = _current.get()
    if s is not None:
        s.record_llm(model, input_tokens, output_tokens, **extra)


def record_retrieval(query: str, documents: List[Dict[str, Any]]) -> None:
    """Record a retrieval's query and ranked documents on the current span. A no-op outside a trace."""
    s = _current.get()
    if s is not None:
        s.record_retrieval(query, documents)


def trace(fn: Optional[Callable] = None, *, name: Optional[str] = None,
          kind: str = "chain", capture_io: bool = True):
    """Decorator for sync or async functions. Usable as `@trace` or `@trace(kind="llm")`."""

    def decorate(f: Callable) -> Callable:
        span_name = name or f.__qualname__.split("<locals>.")[-1]
        try:
            sig = inspect.signature(f)
        except (TypeError, ValueError):
            sig = None

        def inputs(args, kwargs):
            if sig is None:
                return None
            try:
                bound = sig.bind_partial(*args, **kwargs)
            except TypeError:
                return None
            return {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}

        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def awrapper(*args, **kwargs):
                with get_tracer().span(span_name, kind) as s:
                    if capture_io:
                        s.input = inputs(args, kwargs)
                    result = await f(*args, **kwargs)
                    if capture_io:
                        s.output = result
                    return result
            return awrapper

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with get_tracer().span(span_name, kind) as s:
                if capture_io:
                    s.input = inputs(args, kwargs)
                result = f(*args, **kwargs)
                if capture_io:
                    s.output = result
                return result
        return wrapper

    return decorate(fn) if fn is not None else decorate
