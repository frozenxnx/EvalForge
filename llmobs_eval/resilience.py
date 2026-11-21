"""Rate limiting and retry policy, kept separate from provider code."""
from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional, Tuple, TypeVar

from .providers import ProviderError

T = TypeVar("T")

# A timeout is ambiguous: the server took the request and may still be working on it. Retrying only adds a second
# request to its queue (and a 5-minute timeout retried 4 times is 25 minutes), so a timed-out call gets one more try.
MAX_TIMEOUT_RETRIES = 1


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
    attempt = timeouts = 0
    while True:
        try:
            return fn(), attempt + 1
        except ProviderError as exc:
            exc.attempts = attempt + 1
            if exc.kind == "timeout":
                timeouts += 1
                if timeouts > MAX_TIMEOUT_RETRIES:
                    raise
            if not exc.retryable or attempt >= max_retries:
                raise
            if exc.retry_after is not None:
                delay = exc.retry_after
            else:
                delay = base * (2 ** attempt) * (0.5 + rng())
            sleep(min(delay, cap))
            attempt += 1
