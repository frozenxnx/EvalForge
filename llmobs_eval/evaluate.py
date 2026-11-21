"""Score the outputs of an experiment. Safe to re-run: only missing scores are computed."""
from __future__ import annotations

import random
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
ABORT_AFTER_TIMEOUTS = 3                      # each timed-out answer already cost a full timeout (twice): stop sooner


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
                limit = ABORT_AFTER_TIMEOUTS if exc.kind == "timeout" else ABORT_AFTER_CONNECTION_FAILURES
                if consecutive >= limit:
                    what = "timeouts" if exc.kind == "timeout" else "connection failures"
                    raise ProviderError(
                        f"stopped after {consecutive} consecutive {what} ({exc}). Nothing was scored for the remaining "
                        "answers; fix the problem and re-run (answers already scored are skipped).")
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
    extra_providers: Optional[Mapping[str, Mapping[str, Any]]] = None,   # judge/embed providers beyond the experiment's
    max_items: Optional[int] = None,             # judge at most this many answers now; re-run to continue
    concurrency: Optional[int] = None,           # parallel judge calls (default: the experiment's setting)
    timeout: Optional[float] = None,             # per-call timeout for the judge/embedding provider, in seconds
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
    specs = {**cfg["providers"], **(extra_providers or {})}      # extras are used for this evaluation only, never stored
    workers = concurrency or cfg["concurrency"]
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
        "skipped_no_context": 0, "deferred": 0, "judge_unparseable": 0, "judge_failed": 0,
        "faith_unparseable": 0, "faith_failed": 0, "embed_failed": 0, "warnings": [], "error_samples": []}

    def sample(exc: Exception) -> None:  # keep a few distinct failure messages so the CLI can say *why*
        msg = str(exc)[:200]
        if msg not in stats["error_samples"] and len(stats["error_samples"]) < 3:
            stats["error_samples"].append(msg)

    def resolve(spec: Mapping[str, Any]):
        name = spec["provider"]
        if name not in specs:
            raise ConfigError(f"provider {name!r} is not in this experiment's config "
                              f"(available: {', '.join(specs)}; add more with --extra-providers)")
        if name not in built:
            spec = specs[name] if not timeout else {**specs[name], "timeout": timeout}
            built[name] = build_provider(name, spec)
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
        limiter = RateLimiter(specs[judge["provider"]].get("rps"))
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
            if max_items is not None and len(todo) > max_items:
                stats["deferred"] += len(todo) - max_items
                todo = todo[:max_items]
            _run_parallel(judge_one, todo, workers, progress, judge_done, judge_failed)

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
            if max_items is not None and len(items) > max_items:
                stats["deferred"] += len(items) - max_items
                items = items[:max_items]
            _run_parallel(faith_one, items, workers, progress, faith_done, faith_failed)
    return stats


def probe_judge(store: ExperimentStore, experiment_id: str, judge: Mapping[str, Any], *, n: int = 3,
                extra_providers: Optional[Mapping[str, Mapping[str, Any]]] = None,
                providers: Optional[Mapping[str, Provider]] = None,
                concurrency: Optional[int] = None, timeout: Optional[float] = 60.0,
                max_tokens: Optional[int] = 300) -> Dict[str, Any]:
    """Time a few real judge calls with the given settings (nothing is stored) and project the total.

    Answers "why is this slow?" quickly: the probe uses a short timeout and caps the output (a verdict needs ~40
    tokens), so a model that reasons at length is caught in seconds instead of minutes. It reports per-call
    latency, generated tokens and tokens/second, and whether replies parse."""
    exp = store.get_experiment(experiment_id)
    cfg = exp["config"]
    specs = {**cfg["providers"], **(extra_providers or {})}
    name, model = judge["provider"], judge["model"]
    if name not in specs:
        raise ConfigError(f"provider {name!r} is not in this experiment's config "
                          f"(available: {', '.join(specs)}; add more with --extra-providers)")
    rubric = judge.get("rubric", "v1")
    if rubric not in RUBRICS:
        raise ConfigError(f"unknown rubric {rubric!r}; use one of {RUBRICS}")
    spec = dict(specs[name])
    if timeout:
        spec["timeout"] = min(float(spec.get("timeout", 60.0)), float(timeout))
    prov = (providers or {}).get(name) or build_provider(name, spec)
    params, suffix = dict(judge.get("params") or {"temperature": 0}), judge.get("suffix")
    if max_tokens and not any(k in params for k in ("max_tokens", "num_predict", "max_completion_tokens")):
        params["max_tokens"] = max_tokens
    cap = params.get("max_tokens") or params.get("num_predict") or params.get("max_completion_tokens")
    metric = judge_metric_name(model, rubric)
    examples = {e.id: e for e in store.get_examples(experiment_id)}
    have = store.scored_keys(experiment_id)
    todo = [r for r in store.get_results(experiment_id)
            if r["status"] == "ok" and (examples[r["example_id"]].reference or "").strip()
            and (r["variant"], r["example_id"], metric) not in have]
    sample = random.Random(0).sample(todo, min(n, len(todo)))
    calls: List[Dict[str, Any]] = []
    for r in sample:
        ex = examples[r["example_id"]]
        content = build_prompt(ex.input, ex.reference, r["output"], rubric=rubric)
        if suffix:
            content += "\n\n" + suffix
        started = time.perf_counter()
        try:
            comp = prov.complete(model, [{"role": "user", "content": content}], params)
        except ProviderError as exc:
            calls.append({"error": str(exc)[:300], "kind": exc.kind, "latency_s": time.perf_counter() - started})
            if exc.status is None and exc.retryable:      # the server is not answering: more probes would only wait
                break
            continue
        text = comp.text or ""
        latency = time.perf_counter() - started
        calls.append({"latency_s": latency, "input_tokens": comp.input_tokens, "output_tokens": comp.output_tokens,
                      "tokens_per_s": (comp.output_tokens / latency) if comp.output_tokens and latency > 0 else None,
                      "thinking_tag": "<think" in text.lower(), "parsed": parse_verdict(text) is not None})
    good = [c for c in calls if "error" not in c]
    mean = sum(c["latency_s"] for c in good) / len(good) if good else None
    workers = concurrency or cfg["concurrency"]
    advice: List[str] = []
    if not todo:
        advice.append("nothing is left to judge with this metric")
    errors = [c for c in calls if "error" in c]
    if errors:
        advice.append(f"{len(errors)} of {len(calls)} probe calls failed: {errors[0]['error']}")
    if any(c.get("kind") == "timeout" for c in errors):
        advice.append(
            f"a call did not finish within {spec.get('timeout', timeout):g}s although the server accepted it. Likely causes, "
            "in order: (1) the model is reasoning at length: use the native Ollama provider with think=false "
            "(--extra-providers examples/providers_ollama_native.json) and probe again; (2) Ollama is still busy with "
            "requests from an earlier, interrupted run: quit and restart Ollama to clear them; (3) the model does not fit in "
            "memory and is swapping: `ollama ps` shows what is loaded and how big it is; close other programs or use a "
            "smaller judge. Check CPU use in the task manager: busy means it is generating, idle means it is stuck")
    out_tokens = [c["output_tokens"] for c in good if c.get("output_tokens") is not None]
    hit_cap = bool(cap) and any(c.get("output_tokens") is not None and c["output_tokens"] >= cap and not c["parsed"]
                                for c in good)
    if any(c["thinking_tag"] for c in good) or hit_cap or (out_tokens and sum(out_tokens) / len(out_tokens) > 150):
        avg = f"about {sum(out_tokens) / len(out_tokens):.0f}" if out_tokens else "many"
        why = (f"it reached the {cap}-token limit without giving a verdict" if hit_cap else
               f"the model generates {avg} tokens per call; a verdict needs ~40")
        advice.append(f"{why}: it is reasoning ('thinking') behind the scenes, which is what makes it slow. Switch "
                      "thinking off: use the native Ollama provider (examples/providers_ollama_native.json: think=false), or "
                      "--judge-suffix /no_think, or --judge-param reasoning_effort=none, then probe again")
    if mean is not None and mean > 8:
        advice.append(f"each call takes {mean:.1f}s. To get results sooner: --max-items N scores N answers per run "
                      "(re-run to continue), --concurrency 2 helps only if the server can run requests in parallel, and a "
                      "smaller model judges faster")
    bad = [c for c in good if not c["parsed"] and not (hit_cap and c.get("output_tokens") is not None
                                                         and c["output_tokens"] >= cap)]
    if bad:
        advice.append(f"{len(bad)} of {len(good)} replies could not be parsed as a verdict")
    return {"metric": metric, "model": model, "sampled": len(sample), "calls": calls, "mean_latency_s": mean,
            "remaining": len(todo), "workers": workers, "max_tokens": cap, "timeout": spec.get("timeout", timeout),
            "estimate_s": (len(todo) * mean / workers) if mean is not None else None, "advice": advice}


def format_probe(p: Mapping[str, Any]) -> str:
    def dur(s: float) -> str:
        s = int(round(s))
        return f"{s}s" if s < 60 else (f"{s // 60}m{s % 60:02d}s" if s < 3600 else f"{s // 3600}h{(s % 3600) // 60:02d}m")
    limits = "".join([f"; output capped at {p['max_tokens']} tokens" if p.get("max_tokens") else "",
                      f"; timeout {p['timeout']:g}s" if p.get("timeout") else ""])
    out = [f"judge speed probe: {p['metric']}   ({p['sampled']} real calls, nothing stored{limits})", ""]
    for i, c in enumerate(p["calls"], 1):
        if "error" in c:
            out.append(f"  call {i}: FAILED after {c['latency_s']:.1f}s: {c['error']}")
        else:
            tok = "-" if c.get("output_tokens") is None else str(c["output_tokens"])
            inp = "-" if c.get("input_tokens") is None else str(c["input_tokens"])
            speed = "" if c.get("tokens_per_s") is None else f"({c['tokens_per_s']:.1f} tok/s)   "
            out.append(f"  call {i}: {c['latency_s']:.1f}s   prompt tokens {inp}   generated tokens {tok} {speed}"
                       f"{'<think> seen   ' if c['thinking_tag'] else ''}{'parsed' if c['parsed'] else 'NOT PARSED'}")
    out.append("")
    if p["mean_latency_s"] is not None:
        out.append(f"mean {p['mean_latency_s']:.1f}s per call; {p['remaining']} answers left to judge with {p['workers']} "
                   f"worker(s): about {dur(p['estimate_s'])}")
    for a in p["advice"]:
        out.append(f"- {a}")
    return "\n".join(out)
