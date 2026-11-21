"""Model provider adapters (stdlib HTTP only).

Adapters take chat-style `messages` and return a `Completion`. Failures raise
`ProviderError` with `retryable` set, so the runner's retry policy stays
provider-agnostic. API keys are never read from config files, only from env vars.
"""
from __future__ import annotations

import json
import os
import random
import socket
import threading
import time
import zlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple


class ConfigError(ValueError):
    pass


@dataclass
class Completion:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


class ProviderError(Exception):
    def __init__(self, message: str, status: Optional[int] = None,
                 retry_after: Optional[float] = None, retryable: bool = False,
                 kind: Optional[str] = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.retryable = retryable
        self.kind = kind          # "timeout" | "connection" | None (an HTTP answer or something else)
        self.attempts = 1


class Provider:
    def complete(self, model: str, messages: List[Dict[str, str]],
                 params: Dict[str, Any]) -> Completion:  # pragma: no cover - interface
        raise NotImplementedError

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        raise ConfigError(f"{type(self).__name__} does not support embeddings")


def _post_json(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:500].decode("utf-8", "replace")
        ra = exc.headers.get("Retry-After")
        retry_after = float(ra) if ra and ra.replace(".", "", 1).isdigit() else None
        raise ProviderError(
            f"HTTP {exc.code}: {detail}", status=exc.code, retry_after=retry_after,
            retryable=exc.code in (408, 409, 425, 429) or exc.code >= 500,
        ) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        # urllib wraps errors raised while connecting/sending in URLError, but lets a timeout while waiting
        # for the response through bare. So a URLError whose reason is a timeout means the connection was
        # never established (e.g. Windows takes ~2s to report a refused port), not a slow server.
        if isinstance(exc, (TimeoutError, socket.timeout)):
            raise ProviderError(
                f"timed out after {timeout:g}s waiting for {url}: the server accepted the request but did not answer "
                "(still loading the model, busy with earlier requests, or reasoning for a long time)",
                retryable=True, kind="timeout") from None
        if isinstance(reason, (TimeoutError, socket.timeout)):
            reason = f"no connection within {timeout:g}s"
        raise ProviderError(f"cannot reach {url}: {reason}", retryable=True, kind="connection") from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ProviderError(f"non-JSON response: {raw[:200]!r}") from None


class OpenAICompatProvider(Provider):
    """Anything that speaks POST {base_url}/chat/completions: OpenAI, Ollama, vLLM, OpenRouter..."""

    def __init__(self, base_url: str = "https://api.openai.com/v1",
                 api_key: Optional[str] = None, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, model, messages, params):
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"model": model, "messages": messages, **params}
        data = _post_json(f"{self.base_url}/chat/completions", headers, body, self.timeout)
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError(f"unexpected response shape: {str(data)[:200]}") from None
        usage = data.get("usage") or {}
        return Completion(text, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                          choice.get("finish_reason"))

    def embed(self, model, texts):
        data = _post_json(f"{self.base_url}/embeddings",
                          {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                          {"model": model, "input": texts}, self.timeout)
        try:
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            vecs = [list(map(float, d["embedding"])) for d in items]
        except (KeyError, TypeError, ValueError):
            raise ProviderError(f"unexpected embeddings response: {str(data)[:200]}") from None
        if len(vecs) != len(texts):
            raise ProviderError(f"expected {len(texts)} embeddings, got {len(vecs)}")
        return vecs


class AnthropicProvider(Provider):
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com",
                 timeout: float = 60.0, version: str = "2023-06-01") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.version = version

    def complete(self, model, messages, params):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body: Dict[str, Any] = {"model": model, "max_tokens": 1024, **params,
                                "messages": [m for m in messages if m["role"] != "system"]}
        if system:
            body["system"] = system
        headers = {"x-api-key": self.api_key, "anthropic-version": self.version}
        data = _post_json(f"{self.base_url}/v1/messages", headers, body, self.timeout)
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ProviderError(f"unexpected response shape: {str(data)[:200]}")
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage") or {}
        return Completion(text, usage.get("input_tokens"), usage.get("output_tokens"),
                          data.get("stop_reason"))


class OllamaProvider(Provider):
    """Ollama's native /api/chat (not the OpenAI-compatible endpoint).

    It can switch a reasoning model's thinking off (`think: false`), keeps the model loaded between calls
    (`keep_alive`, so it is not reloaded after Ollama's short default idle timeout) and reports exact token
    counts. `params` become Ollama `options` (temperature, num_predict, num_ctx ...); `max_tokens` is accepted
    as an alias for `num_predict`, and `think` / `keep_alive` in `params` override the provider's settings.
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 60.0,
                 think: Optional[bool] = None, keep_alive: Optional[str] = "30m") -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.think = think
        self.keep_alive = keep_alive

    def complete(self, model, messages, params):
        options = dict(params)
        think = options.pop("think", self.think)
        keep_alive = options.pop("keep_alive", self.keep_alive)
        if "max_tokens" in options:
            options["num_predict"] = options.pop("max_tokens")
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if think is not None:
            body["think"] = think
        if keep_alive:
            body["keep_alive"] = keep_alive
        if options:
            body["options"] = options
        data = _post_json(f"{self.base_url}/api/chat", {}, body, self.timeout)
        msg = data.get("message")
        if not isinstance(msg, dict):
            raise ProviderError(f"unexpected response shape: {str(data)[:200]}")
        return Completion(msg.get("content") or "", data.get("prompt_eval_count"), data.get("eval_count"),
                          data.get("done_reason"))

    def embed(self, model, texts):
        body: Dict[str, Any] = {"model": model, "input": texts}
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        data = _post_json(f"{self.base_url}/api/embed", {}, body, self.timeout)
        vecs = data.get("embeddings")
        if not isinstance(vecs, list) or len(vecs) != len(texts):
            raise ProviderError(f"unexpected embeddings response: {str(data)[:200]}")
        return [list(map(float, v)) for v in vecs]


class MockProvider(Provider):
    """Deterministic, keyless provider for tests, CI and demos."""

    def __init__(self, latency_ms: Tuple[float, float] = (20, 80),
                 error_rate: float = 0.0, seed: int = 0) -> None:
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def complete(self, model, messages, params):
        with self._lock:
            latency = self._rng.uniform(*self.latency_ms)
            fail = self._rng.random() < self.error_rate
        time.sleep(latency / 1000)
        if fail:
            raise ProviderError("mock transient failure", status=503, retryable=True)
        text = f"[{model}] {messages[-1]['content'][:80]}"
        return Completion(
            text,
            input_tokens=sum(len(m["content"].split()) for m in messages),
            output_tokens=len(text.split()),
            finish_reason="stop",
        )

    def embed(self, model, texts):
        """Hashed bag-of-words vectors: lexical, offline, deterministic. For tests/demos only."""
        out = []
        for t in texts:
            v = [0.0] * 64
            for tok in t.lower().split():
                v[zlib.crc32(tok.encode()) % 64] += 1.0
            out.append(v)
        return out


def build_provider(name: str, spec: Mapping[str, Any], env: Mapping[str, str] = os.environ) -> Provider:
    kind = spec.get("type")
    key_env = spec.get("api_key_env")
    key = env.get(key_env) if key_env else None
    if key_env and not key:
        raise ConfigError(f"provider {name!r}: environment variable {key_env} is not set")
    timeout = float(spec.get("timeout", 60.0))
    if kind == "mock":
        lo, hi = spec.get("latency_ms", (20, 80))
        return MockProvider((lo, hi), float(spec.get("error_rate", 0.0)), int(spec.get("seed", 0)))
    if kind == "anthropic":
        if not key:
            raise ConfigError(f"provider {name!r}: anthropic needs api_key_env")
        return AnthropicProvider(key, spec.get("base_url", "https://api.anthropic.com"), timeout)
    if kind == "openai_compat":
        return OpenAICompatProvider(spec.get("base_url", "https://api.openai.com/v1"), key, timeout)
    if kind == "ollama":
        return OllamaProvider(spec.get("base_url", "http://localhost:11434"), timeout, spec.get("think"),
                              spec.get("keep_alive", "30m"))
    raise ConfigError(f"provider {name!r}: unknown type {kind!r} (use mock, anthropic, openai_compat, ollama)")
