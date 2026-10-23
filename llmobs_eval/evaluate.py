"""Score the outputs of an experiment. Safe to re-run: only missing scores are computed."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import llmobs

from . import faithfulness as faith
from .judge import RETRY_NUDGE, RUBRICS, VERDICT_VALUES, build_prompt, judge_metric_name, parse_verdict
from .metrics import LEXICAL, cosine
from .providers import ConfigError, Provider, ProviderError, build_provider
from .resilience import RateLimiter, call_with_retries
from .retrieval import format_context, retrieval_metrics
from .store import ExperimentStore

EMBED_BATCH = 32


ABORT_AFTER_CONNECTION_FAILURES = 5


def _run_parallel(fn, items: Sequence[tuple], workers: int, progress, on_ok, on_provider_error) -> None:
    """Run fn over items in a thread pool. A provider error is handed to on_provider_error (the item stays
    unscored, so a re-run retries it). After several CONSECUTIVE connection failures (no HTTP answer at all:
    refused, DNS, timeout) the provider is evidently down: stop instead of retrying every remaining item."""
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval")
    done = consecutive = 0
    try:
        futures = [pool.submit(fn, *it) for it in items]
        for fut in as_completed(futures):
            done += 1
            try:
                res = fut.result()
            except ProviderError as exc:
                on_provider_error(exc)
                consecutive = consecutive + 1 if (exc.status is None and exc.retryable) else 0
                if consecutive >= ABORT_AFTER_CONNECTION_FAILURES:
                    raise ProviderError(
                        f"stopped after {consecutive} consecutive connection failures ({exc}). Nothing was scored for "
                        "the remaining answers; fix the connection and re-run (answers already scored are skipped).")
            else:
                on_ok(res)
                consecutive = 0
            if progress:
                progress(done, len(items))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True)


def evaluate_experiment(
    store: ExperimentStore, experiment_id: str, *,
    lexical: bool = True,
    retrieval: bool = True,                      # recall@k / precision@k / hit@k / mrr for retriever variants
    judge: Optional[Mapping[str, Any]] = None,   # {"provider": name, "model": model, "params": {...}}
    faithfulness: bool = False,                  # needs `judge`; checks answers against the retrieved context
    correctness: bool = True,                    # with `judge`: grade answers against the reference
    only_labeled: bool = False,                  # judge only the answers a human has labeled (a cheap calibration run)
    embed: Optional[Mapping[str, Any]] = None,   # {"provider": name, "model": model}
    providers: Optional[Mapping[str, Provider]] = None,  # test hook: pre-built providers by name
    progress: Optional[Callable[[int, int], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    if faithfulness and not judge:
        raise ConfigError("faithfulness needs a judge model (--judge PROVIDER:MODEL)")
    if judge and not correctness and not faithfulness:
        raise ConfigError("nothing to judge: enable correctness and/or faithfulness")
    exp = store.get_experiment(experiment_id)
    cfg = exp["config"]
    examples = {e.id: e for e in store.get_examples(experiment_id)}
    docs_by_id = {d.id: d for d in store.get_docs(experiment_id)}
    retrievals = store.get_retrievals(experiment_id)
    var_cfg = {v["label"]: v for v in cfg["variants"]}
    all_results = store.get_results(experiment_id)
    ok = [r for r in all_results if r["status"] == "ok"]
    refd = [(r, examples[r["example_id"]]) for r in ok
            if (examples[r["example_id"]].reference or "").strip()]
    have = store.scored_keys(experiment_id)
    built: Dict[str, Provider] = dict(providers or {})
    stats: Dict[str, Any] = {
        "scored": 0, "skipped_no_reference": len(ok) - len(refd), "skipped_no_gold_docs": 0,
        "skipped_no_context": 0, "judge_unparseable": 0, "judge_failed": 0,
        "faith_unparseable": 0, "faith_failed": 0, "embed_failed": 0, "warnings": [], "error_samples": []}

    def sample(exc: Exception) -> None:  # keep a few distinct failure messages so the CLI can say *why*
        msg = str(exc)[:200]
        if msg not in stats["error_samples"] and len(stats["error_samples"]) < 3:
            stats["error_samples"].append(msg)

    def resolve(spec: Mapping[str, Any]):
        name = spec["provider"]
        if name not in cfg["providers"]:
            raise ConfigError(f"provider {name!r} is not in this experiment's config "
                              f"(available: {', '.join(cfg['providers'])})")
        if name not in built:
            built[name] = build_provider(name, cfg["providers"][name])
        return built[name]

    def row(r, metric, value, detail=None):
        return {"experiment_id": experiment_id, "variant": r["variant"], "example_id": r["example_id"],
                "metric": metric, "value": value, "detail": detail}

    def todo_for(metric):
        return [(r, ex) for r, ex in refd if (r["variant"], r["example_id"], metric) not in have]

    # -- lexical -----------------------------------------------------------
    if lexical:
        rows = [row(r, name, fn(r["output"], ex.reference))
                for name, fn in LEXICAL.items() for r, ex in todo_for(name)]
        stats["scored"] += store.add_scores(rows)

    # -- retrieval quality (needs labels; skips oracle, whose recall is 1 by construction) ---------------
    if retrieval:
        rows = []
        for r in all_results:
            spec = (var_cfg.get(r["variant"]) or {}).get("retriever")
            if not spec or spec["type"] == "oracle":
                continue
            found = retrievals.get((r["variant"], r["example_id"]))
            if r["status"] != "ok" and not found:
                continue                                   # retrieval itself failed: nothing to score
            ex = examples[r["example_id"]]
            if not ex.relevant_doc_ids:
                stats["skipped_no_gold_docs"] += 1
                continue
            ids = [d for d, _ in (found or [])]
            detail = {"retrieved": ids, "relevant": list(ex.relevant_doc_ids)}
            for name, val in retrieval_metrics(ids, ex.relevant_doc_ids, spec["k"]).items():
                if (r["variant"], r["example_id"], name) not in have:
                    rows.append(row(r, name, val, detail))
        stats["scored"] += store.add_scores(rows)

    # -- embedding similarity ---------------------------------------------
    if embed:
        prov, model = resolve(embed), embed["model"]
        metric = f"embed_sim:{model}"
        todo = todo_for(metric)
        texts = sorted({t for r, ex in todo for t in (r["output"], ex.reference) if t.strip()})
        vectors: Dict[str, List[float]] = {}
        for i in range(0, len(texts), EMBED_BATCH):
            chunk = texts[i:i + EMBED_BATCH]
            try:
                vecs, _ = call_with_retries(lambda: prov.embed(model, chunk), cfg["max_retries"], sleep=sleep)
                vectors.update(zip(chunk, vecs))
            except ProviderError as exc:
                stats["embed_failed"] += len(chunk)
                sample(exc)
        rows = []
        for r, ex in todo:
            if not r["output"].strip():
                rows.append(row(r, metric, 0.0, {"note": "empty output"}))
            elif r["output"] in vectors and ex.reference in vectors:
                rows.append(row(r, metric, cosine(vectors[r["output"]], vectors[ex.reference])))
        stats["scored"] += store.add_scores(rows)

    # -- LLM judge and faithfulness (same judge model, same plumbing) -------
    if judge:
        prov, model = resolve(judge), judge["model"]
        params = dict(judge.get("params") or {"temperature": 0})
        rubric = judge.get("rubric", "v1")
        suffix = judge.get("suffix")             # e.g. "/no_think" to switch off a reasoning model's thinking
        if rubric not in RUBRICS:
            raise ConfigError(f"unknown rubric {rubric!r}; use one of {RUBRICS}")
        limiter = RateLimiter(cfg["providers"][judge["provider"]].get("rps"))
        if any(v["model"] == model for v in cfg["variants"]):
            stats["warnings"].append(
                f"judge model {model!r} is also a candidate: judges tend to favor their own outputs; "
                "prefer a different model for judging")

        def ask(messages, span_name):
            def once():
                limiter.acquire()
                with llmobs.span(span_name, kind="llm", provider=judge["provider"]) as s:
                    s.input = messages
                    sent = messages if not suffix else messages[:-1] + [
                        {**messages[-1], "content": messages[-1]["content"] + "\n\n" + suffix}]
                    comp = prov.complete(model, sent, params)
                    s.record_llm(model, comp.input_tokens, comp.output_tokens)
                    s.output = comp.text
                    return comp
            return call_with_retries(once, cfg["max_retries"], sleep=sleep)[0]

        def ask_parsed(messages, parse, nudge, span_name):
            """Ask; if the reply can't be parsed, re-ask once with a nudge. Returns (parsed|None, last text)."""
            comp = ask(messages, span_name)
            parsed = parse(comp.text)
            if parsed is None:
                messages = messages + [{"role": "assistant", "content": comp.text or "(empty)"},
                                       {"role": "user", "content": nudge}]
                comp = ask(messages, span_name)
                parsed = parse(comp.text)
            return parsed, (comp.text or "")

        # ---- correctness ----
        metric = judge_metric_name(model, rubric)

        def judge_one(r, ex) -> Dict[str, Any]:
            with llmobs.span("judge", kind="chain", experiment_id=experiment_id, variant=r["variant"],
                             example_id=r["example_id"], metric=metric) as root:
                messages = [{"role": "user", "content": build_prompt(ex.input, ex.reference, r["output"], rubric=rubric)}]
                parsed, raw = ask_parsed(messages, parse_verdict, RETRY_NUDGE, "judge_call")
                detail = {"rubric": rubric, "judge_model": model, "trace_id": root.trace_id}
                if parsed is None:
                    detail.update(error="unparseable judge output", raw=raw[:500])
                    root.output = "unparseable"
                    return row(r, metric, None, detail)
                detail.update(parsed)
                root.output = parsed["verdict"]
                return row(r, metric, VERDICT_VALUES[parsed["verdict"]], detail)

        def judge_done(res):
            stats["scored"] += store.add_scores([res])
            if res["value"] is None:
                stats["judge_unparseable"] += 1

        def judge_failed(exc):
            stats["judge_failed"] += 1
            sample(exc)

        if correctness:
            todo = todo_for(metric)
            if only_labeled:
                labeled = {(l["variant"], l["example_id"]) for l in store.get_labels(experiment_id)}
                if not labeled:
                    raise ConfigError("--only-labeled needs human labels: run `python -m llmobs_eval label <id>` first")
                todo = [(r, ex) for r, ex in todo if (r["variant"], r["example_id"]) in labeled]
            _run_parallel(judge_one, todo, cfg["concurrency"], progress, judge_done, judge_failed)

        # ---- faithfulness to the retrieved context ----
        if faithfulness:
            fmetric = f"faithfulness:{model}"

            def faith_one(r) -> Dict[str, Any]:
                ex = examples[r["example_id"]]
                hits = retrievals.get((r["variant"], r["example_id"]), [])
                context = format_context([docs_by_id[d] for d, _ in hits if d in docs_by_id])
                with llmobs.span("faithfulness", kind="chain", experiment_id=experiment_id,
                                 variant=r["variant"], example_id=r["example_id"], metric=fmetric) as root:
                    detail = {"rubric": faith.FAITH_VERSION, "judge_model": model, "trace_id": root.trace_id}
                    claims, raw = ask_parsed(
                        [{"role": "user", "content": faith.build_extract_prompt(ex.input, r["output"])}],
                        faith.parse_claims, faith.NUDGE_CLAIMS, "faith_extract")
                    if claims is None:
                        detail.update(error="unparseable claim list", raw=raw[:500])
                        return row(r, fmetric, None, detail)
                    if not claims:
                        detail.update(no_claims=True, n_claims=0, reason="no factual claims to check")
                        root.output = "no claims"
                        return row(r, fmetric, 1.0, detail)
                    verdicts, raw = ask_parsed(
                        [{"role": "user", "content": faith.build_verify_prompt(context, claims)}],
                        lambda t: faith.parse_verdicts(t, len(claims)), faith.NUDGE_VERDICTS, "faith_verify")
                    if verdicts is None:
                        detail.update(error="unparseable verdicts", raw=raw[:500], claims=claims)
                        return row(r, fmetric, None, detail)
                    value, reason = faith.score(claims, verdicts)
                    detail.update(n_claims=len(claims), reason=reason,
                                  claims=[{"claim": c, "verdict": v} for c, v in zip(claims, verdicts)])
                    root.output = f"{value:.2f}"
                    return row(r, fmetric, value, detail)

            def faith_done(res):
                stats["scored"] += store.add_scores([res])
                if res["value"] is None:
                    stats["faith_unparseable"] += 1

            def faith_failed(exc):
                stats["faith_failed"] += 1
                sample(exc)

            items = []
            for r in ok:
                if not (var_cfg.get(r["variant"]) or {}).get("retriever"):
                    stats["skipped_no_context"] += 1       # closed-book variants have no context to be faithful to
                elif (r["variant"], r["example_id"], fmetric) not in have:
                    items.append((r,))
            _run_parallel(faith_one, items, cfg["concurrency"], progress, faith_done, faith_failed)
    return stats
