"""Span exporters. Exporters must never raise into, or block, application code."""
from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("llmobs")


class Exporter:
    def export(self, span: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def flush(self, timeout: Optional[float] = None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def stats(self) -> Dict[str, int]:
        """Counters such as sent / failed / dropped. Exporters that keep none return an empty dict."""
        return {}

    def last_error(self) -> Optional[str]:
        """Why the most recent delivery failed, if any did."""
        return None


class NoopExporter(Exporter):
    def export(self, span):
        pass


class InMemoryExporter(Exporter):
    """Collects spans in a list. Handy for tests."""

    def __init__(self) -> None:
        self.spans: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def export(self, span):
        with self._lock:
            self.spans.append(span)


def _is_loopback(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    return host in ("localhost", "::1") or host.startswith("127.")


class HTTPExporter(Exporter):
    """Queues spans and ships them in batches from a background thread.

    - export() is O(1) and non-blocking; if the queue is full the span is dropped and counted
      (tracing must never slow down or crash the app).
    - Transient failures (network errors, 5xx, 429, 408) are retried with exponential backoff. A 4xx will
      not fix itself, so it is not retried.
    - After `breaker_threshold` consecutive failed batches the exporter stops trying for
      `breaker_cooldown` seconds (spans are counted as failed immediately) and then probes with a single
      attempt, so a dead trace server costs milliseconds instead of ~20 s per batch.
    - The reason for a failure is kept (`last_error()`) and logged once, not once per batch.
    - Loopback endpoints bypass system proxy settings.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        batch_size: int = 50,
        flush_interval: float = 1.0,
        max_queue: int = 10_000,
        max_retries: int = 3,
        backoff_base: float = 0.2,
        timeout: float = 5.0,
        autostart: bool = True,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown = breaker_cooldown
        self._clock = clock
        self._opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
                        if _is_loopback(endpoint) else urllib.request.build_opener())
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._stats = {"sent": 0, "failed": 0, "dropped": 0}
        self._stats_lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._warned = False
        self._thread = threading.Thread(target=self._run, name="llmobs-exporter", daemon=True)
        if autostart:
            self._thread.start()
            atexit.register(self.shutdown)

    # -- public API ---------------------------------------------------------
    def export(self, span):
        try:
            self._q.put_nowait(span)
        except queue.Full:
            self._bump("dropped", 1)

    def flush(self, timeout: Optional[float] = None) -> None:
        """Block until everything queued so far has been sent or given up on (or `timeout` seconds pass)."""
        if timeout is None:
            self._q.join()
            return
        deadline = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.02)

    def shutdown(self) -> None:
        if self._stop.is_set():
            return
        self.flush(timeout=self.timeout * 2 + 1)
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.timeout + 1)

    def stats(self) -> Dict[str, int]:
        """sent: delivered; failed: not delivered (after retries, or while paused); dropped: queue was full."""
        with self._stats_lock:
            return dict(self._stats)

    def last_error(self) -> Optional[str]:
        with self._stats_lock:
            return self._last_error

    # -- internals ----------------------------------------------------------
    def _bump(self, key: str, n: int) -> None:
        with self._stats_lock:
            self._stats[key] += n

    def _run(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            batch = self._drain()
            if not batch:
                continue
            try:
                self._send(batch)
            finally:
                for _ in batch:
                    self._q.task_done()

    def _drain(self) -> List[Dict[str, Any]]:
        batch: List[Dict[str, Any]] = []
        deadline = time.monotonic() + self.flush_interval
        while len(batch) < self.batch_size:
            remaining = 0.05 if self._stop.is_set() else deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self._q.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _record_failure(self, n: int, err: str) -> None:
        with self._stats_lock:
            self._stats["failed"] += n
            self._last_error = err
            self._consecutive_failures += 1
            opened = self._consecutive_failures >= self.breaker_threshold
            if opened:
                self._open_until = self._clock() + self.breaker_cooldown
            first = not self._warned
            self._warned = True
        if first:
            log.warning("llmobs: could not deliver spans to %s: %s "
                        "(further failures are counted, not logged; see exporter.stats())", self.endpoint, err)
        if opened and self._consecutive_failures == self.breaker_threshold:
            log.warning("llmobs: pausing trace export for %.0fs after %d consecutive failed batches",
                        self.breaker_cooldown, self.breaker_threshold)

    def _send(self, batch: List[Dict[str, Any]]) -> None:
        n = len(batch)
        with self._stats_lock:
            paused = self._clock() < self._open_until
            probing = self._consecutive_failures >= self.breaker_threshold and not paused
        if paused:
            self._bump("failed", n)                  # fail fast: the server was just unreachable
            return
        retries = 0 if probing else self.max_retries
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps({"spans": batch}).encode()
        err = "unknown error"
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
                with self._opener.open(req, timeout=self.timeout) as resp:
                    if 200 <= resp.status < 300:
                        self._bump("sent", n)
                        with self._stats_lock:
                            self._consecutive_failures, self._open_until = 0, 0.0
                        return
                    err = f"HTTP {resp.status}"
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read()[:200].decode("utf-8", "replace")
                except Exception:
                    detail = ""
                err = f"HTTP {exc.code}: {detail}".rstrip(": ")
                if not (exc.code >= 500 or exc.code in (408, 429)):
                    break                            # a 4xx will not fix itself
            except Exception as exc:                 # connection refused, timeout, DNS, ...
                err = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(self.backoff_base * (2 ** attempt))
        self._record_failure(n, err)
