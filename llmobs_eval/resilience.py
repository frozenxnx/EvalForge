"""Rate limiting and retry policy, kept separate from provider code."""
from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional, Tuple, TypeVar

from .providers import ProviderError

T = TypeVar("T")


class RateLimiter:
    """Spaces calls at most `rps` per second across all threads. rps=None disables it."""

    def __init__(self, rps: Optional[float]) -> None:
        self.interval = 1.0 / rps if rps else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait:
            time.sleep(wait)


def call_with_retries(
    fn: Callable[[], T], max_retries: int, base: float = 0.5, cap: float = 30.0,
    sleep: Callable[[float], None] = time.sleep, rng: Callable[[], float] = random.random,
) -> Tuple[T, int]:
    """Call fn(); retry retryable ProviderErrors with exponential backoff + jitter.

    A server-provided Retry-After wins over our own backoff. Returns (result, attempts).
    On final failure the ProviderError is re-raised with `.attempts` set.
    """
    attempt = 0
    while True:
        try:
            return fn(), attempt + 1
        except ProviderError as exc:
            exc.attempts = attempt + 1
            if not exc.retryable or attempt >= max_retries:
                raise
            if exc.retry_after is not None:
                delay = exc.retry_after
            else:
                delay = base * (2 ** attempt) * (0.5 + rng())
            sleep(min(delay, cap))
            attempt += 1
