"""Experiment runner: every (example x variant) pair becomes one traced, retried call.

A variant may declare a `retriever` (bm25 | embedding | oracle). The runner then retrieves documents
for each example, records which ones (rank + score) in the `retrievals` table, and fills the prompt
template's {context} slot with them. Retrieval and generation share one trace.
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Mapping, Optional

import llmobs

from .datasets import Corpus, Dataset, Example
from .providers import ConfigError, Provider, ProviderError, build_provider
from .resilience import RateLimiter, call_with_retries
from .retrieval import build_retrievers, format_context
from .store import ExperimentStore

RETRIEVER_TYPES = ("bm25", "embedding", "oracle")
ABORT_AFTER_CONNECTION_FAILURES = 5
_SLOT = re.compile(r"\{(input|context)\}")


def normalize_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and fill defaults. The result is what gets frozen into the experiment."""
    cfg = json.loads(json.dumps(cfg))  # deep copy, JSON-safe
    providers, variants = cfg.get("providers") or {}, cfg.get("variants") or []
    if not providers or not variants:
        raise ConfigError("config needs non-empty 'providers' and 'variants'")
    labels = set()
    for v in variants:
        for key in ("label", "provider", "model"):
            if not v.get(key):
                raise ConfigError(f"variant is missing {key!r}: {v}")
        label = v["label"]
        if v["provider"] not in providers:
            raise ConfigError(f"variant {label!r} uses unknown provider {v['provider']!r}")
        if label in labels:
            raise ConfigError(f"duplicate variant label {label!r}")
        labels.add(label)
        v.setdefault("system", None)
        v.setdefault("prompt_template", "{input}")
        v.setdefault("params", {})
        template = v["prompt_template"]
        if "{input}" not in template:
            raise ConfigError(f"variant {label!r}: prompt_template must contain {{input}}")
        retr = v.get("retriever")
        if retr:
            if not isinstance(retr, dict) or retr.get("type") not in RETRIEVER_TYPES:
                raise ConfigError(f"variant {label!r}: retriever.type must be one of {RETRIEVER_TYPES}")
            retr.setdefault("k", 3)
            if not isinstance(retr["k"], int) or isinstance(retr["k"], bool) or retr["k"] < 1:
                raise ConfigError(f"variant {label!r}: retriever.k must be a positive integer")
            if retr["type"] == "embedding":
                if retr.get("provider") not in providers:
                    raise ConfigError(f"variant {label!r}: embedding retriever needs 'provider' from the providers table")
                if not retr.get("model"):
                    raise ConfigError(f"variant {label!r}: embedding retriever needs 'model'")
            if "{context}" not in template:
                raise ConfigError(f"variant {label!r} has a retriever, so its prompt_template must contain {{context}}")
        elif "{context}" in template:
            raise ConfigError(f"variant {label!r} uses {{context}} but has no retriever")
    cfg.setdefault("name", "experiment")
    cfg.setdefault("concurrency", 8)
    cfg.setdefault("max_retries", 4)
    cfg.setdefault("pricing", {})
    return cfg


def create_experiment(store: ExperimentStore, dataset: Dataset, config: Mapping[str, Any],
                      name: Optional[str] = None, corpus: Optional[Corpus] = None) -> str:
    cfg = normalize_config(config)
    retrieving = [v for v in cfg["variants"] if v.get("retriever")]
    if retrieving and corpus is None:
        raise ConfigError("some variants use a retriever, so a corpus is required (CLI: --corpus PATH)")
    if any(v["retriever"]["type"] == "oracle" for v in retrieving) and \
            not any(e.relevant_doc_ids for e in dataset.examples):
        raise ConfigError("an oracle retriever needs examples with relevant_doc_ids")
    if corpus is not None:
        known = {d.id for d in corpus.docs}
        unknown = sorted({i for e in dataset.examples for i in e.relevant_doc_ids} - known)
        if unknown:
            raise ConfigError(f"dataset names relevant_doc_ids that are not in the corpus: {unknown[:5]}")
    for pname, spec in cfg["providers"].items():  # fail before writing anything (experiments can't be deleted)
        build_provider(pname, spec)
    return store.create_experiment(name or cfg["name"], dataset, cfg, corpus)


def render_messages(variant: Mapping[str, Any], example: Example,
                    context: Optional[str] = None) -> List[Dict[str, str]]:
    values = {"input": example.input, "context": context or ""}
    msgs: List[Dict[str, str]] = []
    if variant.get("system"):
        msgs.append({"role": "system", "content": variant["system"]})
    # single pass, so text inside the question or documents is never re-interpreted as a placeholder
    msgs.append({"role": "user", "content": _SLOT.sub(lambda m: values[m.group(1)], variant["prompt_template"])})
    return msgs


def compute_cost(pricing: Mapping[str, Any], model: str, in_tok: Optional[int],
                 out_tok: Optional[int]) -> Optional[float]:
    """USD cost from a per-million-token price table; None if price or token counts are unknown."""
    p = pricing.get(model)
    if not p or in_tok is None or out_tok is None:
        return None
    return (in_tok * p["input_per_mtok"] + out_tok * p["output_per_mtok"]) / 1_000_000


def run_experiment(
    store: ExperimentStore, experiment_id: str, *,
    progress: Optional[Callable[[int, int], None]] = None,
    max_tasks: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
    providers: Optional[Mapping[str, Provider]] = None,   # test hook: pre-built providers by name
) -> Dict[str, int]:
    """Run (or resume) an experiment. Already-recorded pairs are skipped, so this is idempotent."""
    exp = store.get_experiment(experiment_id)
    cfg = exp["config"]
    examples = store.get_examples(experiment_id)
    docs = store.get_docs(experiment_id)
    docs_by_id = {d.id: d for d in docs}
    built = {n: (providers[n] if providers and n in providers else build_provider(n, s))
             for n, s in cfg["providers"].items()}          # fail fast on missing keys
    limiters = {n: RateLimiter(s.get("rps")) for n, s in cfg["providers"].items()}
    retrievers = build_retrievers(cfg, docs, built, limiters, sleep)   # embeds the corpus once, up front

    all_pairs = [(v, ex) for ex in examples for v in cfg["variants"]]  # interleaved: partial runs stay balanced
    done = store.done_keys(experiment_id)
    todo = [(v, ex) for v, ex in all_pairs if (v["label"], ex.id) not in done]
    if max_tasks is not None:
        todo = todo[:max_tasks]

    abort = threading.Event()
    infra = {"consecutive": 0, "total": 0, "first": None}
    infra_lock = threading.Lock()

    def task(variant: Dict[str, Any], ex: Example) -> None:
        if abort.is_set():                       # the provider is down: do not pile up pointless attempts
            return
        provider, limiter = built[variant["provider"]], limiters[variant["provider"]]
        started = time.perf_counter()
        unrecorded: Optional[str] = None
        last_latency: List[float] = []
        rec: Dict[str, Any] = {"experiment_id": experiment_id, "variant": variant["label"],
                               "example_id": ex.id}
        with llmobs.span(f"{variant['label']}", kind="chain", experiment_id=experiment_id,
                         variant=variant["label"], example_id=ex.id) as root:
            root.input = ex.input
            rec["trace_id"] = root.trace_id
            try:
                context = None
                found = retrievers.get(variant["label"])
                if found:
                    retriever, k = found
                    with llmobs.span("retrieve", kind="retrieval", retriever=variant["retriever"]["type"],
                                     k=k) as rs:
                        rs.input = ex.input
                        hits, _ = call_with_retries(lambda: retriever.retrieve(ex, k), cfg["max_retries"],
                                                    sleep=sleep)
                        pairs = [(h, docs_by_id[h.doc_id]) for h in hits if h.doc_id in docs_by_id]
                        rs.record_retrieval(ex.input, [{"id": d.id, "text": d.text[:500], "score": h.score}
                                                       for h, d in pairs])
                    rec["retrieved"] = [(h.doc_id, h.score) for h, _ in pairs]
                    context = format_context([d for _, d in pairs])
                messages = render_messages(variant, ex, context)

                def attempt():
                    limiter.acquire()
                    with llmobs.span("llm_call", kind="llm", provider=variant["provider"]) as s:
                        s.input = messages
                        t0 = time.perf_counter()
                        comp = provider.complete(variant["model"], messages, variant["params"])
                        last_latency[:] = [(time.perf_counter() - t0) * 1000]
                        s.record_llm(variant["model"], comp.input_tokens, comp.output_tokens)
                        s.output = comp.text
                        return comp

                comp, attempts = call_with_retries(attempt, cfg["max_retries"], sleep=sleep)
                root.output = comp.text
                rec.update(status="ok", output=comp.text, attempts=attempts,
                           latency_ms=last_latency[0], input_tokens=comp.input_tokens,
                           output_tokens=comp.output_tokens,
                           cost_usd=compute_cost(cfg["pricing"], variant["model"],
                                                 comp.input_tokens, comp.output_tokens))
            except ProviderError as exc:
                root.set_attribute("error.retryable", exc.retryable)
                if exc.status is None and exc.retryable:
                    # No HTTP answer at all (connection refused, DNS failure, timeout). That says nothing about the
                    # candidate, so it is NOT recorded: results are immutable, and `resume` must be able to redo it.
                    unrecorded = str(exc)
                else:
                    rec.update(status="error", error=str(exc)[:500], attempts=exc.attempts)
            except Exception as exc:  # a bug in an adapter must not abort the whole experiment
                rec.update(status="error", error=f"internal: {type(exc).__name__}: {exc}"[:500], attempts=1)
        if unrecorded is not None:
            with infra_lock:
                infra["consecutive"] += 1
                infra["total"] += 1
                infra["first"] = infra["first"] or unrecorded
                if infra["consecutive"] >= ABORT_AFTER_CONNECTION_FAILURES:
                    abort.set()
            return
        with infra_lock:
            infra["consecutive"] = 0
        rec["wall_ms"] = (time.perf_counter() - started) * 1000
        rec["created_at"] = time.time()
        store.add_result(rec)

    store.set_status(experiment_id, "running")
    pool = ThreadPoolExecutor(max_workers=cfg["concurrency"], thread_name_prefix="exp")
    completed = 0
    try:
        futures = [pool.submit(task, v, ex) for v, ex in todo]
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            if progress:
                progress(completed, len(todo))
    except BaseException:  # Ctrl+C etc: keep what we have, stop scheduling more
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True)
        remaining = len(all_pairs) - len(store.done_keys(experiment_id))
        store.set_status(experiment_id, "complete" if remaining == 0 else "partial")
    if infra["total"]:
        raise ProviderError(
            f"{infra['total']} call(s) could not reach the provider and were not recorded (first error: "
            f"{infra['first']}). The experiment is partial: fix the connection, then run "
            f"`python -m llmobs_eval resume {experiment_id}`")
    return {"ran": completed, "remaining": remaining}
