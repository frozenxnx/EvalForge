"""Retrievers for RAG experiments (BM25, embedding, oracle) and retrieval-quality metrics."""
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .datasets import Doc, Example
from .metrics import cosine
from .providers import Provider
from .resilience import RateLimiter, call_with_retries


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float


# --------------------------------------------------------------------------- text analysis
_STOP = frozenset(
    "a an the of to in on for and or is are was were be been it its this that these those with as at by "
    "from how what when where which who whom why do does did can could should would will you your we our "
    "they their i me my not if then than so such into about over under between per".split())
_SUFFIXES = (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""))


def stem(word: str) -> str:
    """A deliberately light stemmer: refunds/refunded -> refund, policies -> policy. Not Porter."""
    for suffix, repl in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith("ss"):
                continue
            return word[: -len(suffix)] + repl
    return word


def tokenize(text: str) -> List[str]:
    return [stem(w) for w in re.findall(r"[^\W_]+", text.lower()) if w not in _STOP]


def _doc_text(d: Doc) -> str:
    return f"{d.title}. {d.text}" if d.title else d.text


# --------------------------------------------------------------------------- retrievers
class BM25Retriever:
    """Okapi BM25 over title + text. Pure Python, deterministic, no dependencies."""

    def __init__(self, docs: Sequence[Doc], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs, self.k1, self.b = list(docs), k1, b
        self._len: List[int] = []
        self._post: Dict[str, List[Tuple[int, int]]] = {}
        for i, d in enumerate(self.docs):
            toks = tokenize(_doc_text(d))
            self._len.append(len(toks))
            counts: Dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                self._post.setdefault(t, []).append((i, tf))
        n = len(self.docs)
        self._avgdl = (sum(self._len) / n) if n else 0.0
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._post.items()}

    def retrieve(self, example: Example, k: int) -> List[Hit]:
        scores: Dict[int, float] = {}
        for term in set(tokenize(example.input)):
            for i, tf in self._post.get(term, ()):
                norm = 1 - self.b + self.b * self._len[i] / (self._avgdl or 1.0)
                scores[i] = scores.get(i, 0.0) + self._idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(self.docs[i].id, s) for i, s in ranked if s > 0]


class EmbeddingRetriever:
    """Dense retrieval through a provider's embeddings endpoint (e.g. Ollama nomic-embed-text)."""

    def __init__(self, provider: Provider, model: str, docs: Sequence[Doc], max_retries: int = 4,
                 limiter: Optional[RateLimiter] = None, sleep: Callable[[float], None] = time.sleep,
                 batch: int = 32) -> None:
        self.provider, self.model, self.docs = provider, model, list(docs)
        self.max_retries, self.limiter, self.sleep = max_retries, limiter, sleep
        self._qcache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        texts = [_doc_text(d) for d in self.docs]
        self._vecs: List[List[float]] = []
        for i in range(0, len(texts), batch):          # raises ProviderError if the endpoint is unusable
            self._vecs.extend(self._embed(texts[i:i + batch]))

    def _embed(self, texts: List[str]) -> List[List[float]]:
        def call():
            if self.limiter:
                self.limiter.acquire()
            return self.provider.embed(self.model, texts)
        return call_with_retries(call, self.max_retries, sleep=self.sleep)[0]

    def retrieve(self, example: Example, k: int) -> List[Hit]:
        with self._lock:
            qvec = self._qcache.get(example.input)
        if qvec is None:
            qvec = self._embed([example.input])[0]
            with self._lock:
                self._qcache[example.input] = qvec
        sims = sorted(((cosine(qvec, v), i) for i, v in enumerate(self._vecs)), key=lambda t: (-t[0], t[1]))
        return [Hit(self.docs[i].id, s) for s, i in sims[:k]]


class OracleRetriever:
    """Returns the gold documents named in `relevant_doc_ids`: the upper bound for any retriever."""

    def __init__(self, docs: Sequence[Doc]) -> None:
        self._ids = {d.id for d in docs}

    def retrieve(self, example: Example, k: int) -> List[Hit]:
        return [Hit(i, 1.0) for i in example.relevant_doc_ids if i in self._ids][:k]


def build_retrievers(cfg: Mapping, docs: Sequence[Doc], providers: Mapping[str, Provider],
                     limiters: Mapping[str, RateLimiter], sleep: Callable[[float], None] = time.sleep):
    """variant label -> (retriever, k). Identical retriever settings share one index."""
    out, cache = {}, {}
    for v in cfg["variants"]:
        spec = v.get("retriever")
        if not spec:
            continue
        key = (spec["type"], spec.get("provider"), spec.get("model"))
        if key not in cache:
            if spec["type"] == "bm25":
                cache[key] = BM25Retriever(docs)
            elif spec["type"] == "oracle":
                cache[key] = OracleRetriever(docs)
            else:
                name = spec["provider"]
                cache[key] = EmbeddingRetriever(providers[name], spec["model"], docs, cfg["max_retries"],
                                                limiters.get(name), sleep)
        out[v["label"]] = (cache[key], spec["k"])
    return out


# --------------------------------------------------------------------------- metrics
def retrieval_metrics(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> Dict[str, float]:
    """recall@k, precision@k, hit@k and MRR against labeled relevant documents.

    precision@k divides by k, not by how many documents came back, so a retriever that returns
    nothing is not rewarded for being 'precise'.
    """
    relevant = set(relevant_ids)
    top = list(retrieved_ids)[:k]
    hits = len(relevant & set(top))
    first = next((rank for rank, d in enumerate(top, 1) if d in relevant), None)
    return {f"recall@{k}": hits / len(relevant), f"precision@{k}": hits / k,
            f"hit@{k}": float(hits > 0), "mrr": 1.0 / first if first else 0.0}


def format_context(docs: Sequence[Doc], max_doc_chars: int = 1500, max_total: int = 6000) -> str:
    """The text that goes into a prompt's {context} slot."""
    if not docs:
        return "(no documents were retrieved)"
    parts, used = [], 0
    for d in docs:
        body = d.text if len(d.text) <= max_doc_chars else d.text[:max_doc_chars] + " ..."
        part = f"[{d.id}] " + (f"{d.title}: " if d.title else "") + body
        if used + len(part) > max_total:
            break
        parts.append(part)
        used += len(part)
    return "\n\n".join(parts)
