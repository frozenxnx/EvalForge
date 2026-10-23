"""Experiment comparison: per-variant metrics with CIs, paired deltas vs a baseline, regressions."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .judgecheck import judge_check_payload
from .report import summarize
from .stats import MIN_PAIRS_FOR_SIGNIFICANCE, mean_ci, paired_bootstrap, seed_for
from .store import ExperimentStore

_ORDER = ("exact_match", "contains", "token_f1", "embed_sim", "recall", "precision", "hit", "mrr",
          "judge", "faithfulness")


def _rank(metric: str):
    prefix = re.split(r"[:@]", metric)[0]
    return (_ORDER.index(prefix) if prefix in _ORDER else 99, metric)


def default_metric(metrics: List[str]) -> Optional[str]:
    for m in metrics:
        if m.startswith("judge:"):
            return m
    return "token_f1" if "token_f1" in metrics else (metrics[0] if metrics else None)


def _clip(s: Optional[str], n: int = 600) -> Optional[str]:
    return s if s is None or len(s) <= n else s[:n] + "..."


def compare_payload(store: ExperimentStore, experiment_id: str, baseline: Optional[str] = None,
                    metric: Optional[str] = None, n_boot: int = 2000, top_k: int = 5) -> Dict[str, Any]:
    exp = store.get_experiment(experiment_id)  # KeyError if unknown
    cfg = exp["config"]
    labels = [v["label"] for v in cfg["variants"]]
    baseline = baseline or labels[0]
    if baseline not in labels:
        raise ValueError(f"unknown baseline {baseline!r}; variants are {labels}")
    examples = {e.id: e for e in store.get_examples(experiment_id)}
    order = list(examples)
    results = {(r["variant"], r["example_id"]): r for r in store.get_results(experiment_id)}
    summary = {r["variant"]: r for r in summarize(store, experiment_id)}

    scores: Dict[str, Dict[str, Dict[str, Any]]] = {}   # metric -> variant -> example -> (value, detail)
    nulls: Dict[str, Dict[str, int]] = {}
    for s in store.get_scores(experiment_id):
        if s["value"] is None:
            nulls.setdefault(s["metric"], {}).setdefault(s["variant"], 0)
            nulls[s["metric"]][s["variant"]] += 1
        else:
            scores.setdefault(s["metric"], {}).setdefault(s["variant"], {})[s["example_id"]] = (s["value"], s["detail"])
    metrics = sorted(set(scores) | set(nulls), key=_rank)
    if metric and metrics and metric not in metrics:
        raise ValueError(f"unknown metric {metric!r}; available: {metrics}")
    primary = metric or default_metric(metrics)

    warnings: List[str] = []
    expected = len(labels) * len(examples)
    if exp["status"] != "complete":
        warnings.append(f"experiment is {exp['status']}: {expected - len(results)} of {expected} "
                        "results are missing (use resume)")
    if not metrics:
        warnings.append("no scores yet: run `python -m llmobs_eval evaluate <id>` first")
    models = {v["model"] for v in cfg["variants"]}
    for m in metrics:
        if m.split(":", 1)[0] in ("judge", "faithfulness") and ":" in m and m.split(":", 1)[1].split("#")[0] in models:
            warnings.append(f"{m}: the judge model is also a candidate, which biases scores toward it")
    for m in metrics:
        if m.startswith("judge:"):
            t = judge_check_payload(store, experiment_id, m, top_k=0)["totals"]
            if t["checkable"] >= 10 and t["agreement"] < 0.8:
                warnings.append(
                    f"{m} disagrees with `contains` on {t['lenient'] + t['strict']} of {t['checkable']} answers that "
                    f"have a short reference ({t['lenient']} judge too lenient, {t['strict']} too strict): "
                    f"run `python -m llmobs_eval judge-check {experiment_id}` to see which side is right")

    variants, regressions = [], {}
    gaps: Dict[tuple, List[str]] = {}       # one warning per variant/coverage, not one per metric
    for v in cfg["variants"]:
        label = v["label"]
        row = dict(summary[label])
        if row["errors"]:
            warnings.append(f"{label}: {row['errors']} generation errors are excluded from metric means")
        row["scores"], row["vs_baseline"] = {}, {}
        for m in metrics:
            mine = scores.get(m, {}).get(label, {})
            row["scores"][m] = mean_ci([x[0] for x in mine.values()], n_boot, seed=seed_for(label, m))
            unscored = row["ok"] - len(mine)
            n_null = nulls.get(m, {}).get(label, 0)
            if unscored > 0 and (mine or n_null):      # a metric that never applied to this variant is not a gap
                gaps.setdefault((label, len(mine), row["ok"], n_null), []).append(m)
            if label != baseline:
                base = scores.get(m, {}).get(baseline, {})
                ids = [e for e in order if e in base and e in mine]
                row["vs_baseline"][m] = paired_bootstrap(
                    [base[e][0] for e in ids], [mine[e][0] for e in ids], n_boot,
                    seed=seed_for(baseline, label, m))
        if label != baseline and primary:
            base = scores.get(primary, {}).get(baseline, {})
            mine = scores.get(primary, {}).get(label, {})
            worse = sorted(((mine[e][0] - base[e][0], i, e) for i, e in enumerate(order)
                            if e in base and e in mine and mine[e][0] < base[e][0]))
            items = []
            for delta, _, e in worse[:top_k]:
                rb, rc = results.get((baseline, e)), results.get((label, e))
                items.append({
                    "example_id": e, "input": _clip(examples[e].input), "reference": _clip(examples[e].reference),
                    "baseline_score": base[e][0], "candidate_score": mine[e][0], "delta": delta,
                    "baseline_output": _clip(rb["output"] if rb else None),
                    "candidate_output": _clip(rc["output"] if rc else None),
                    "reason": (mine[e][1] or {}).get("reason"),
                })
            regressions[label] = {"metric": primary, "count": len(worse), "items": items}
        variants.append(row)

    for (label, n_scored, n_ok, n_null), ms in gaps.items():
        warnings.append(f"{', '.join(ms)} / {label}: scored {n_scored} of {n_ok} successful results "
                        f"({n_null} unparseable; the rest lacked a reference, labels or context)")
    if any(any(x["n_paired"] > 0 and not x["enough_pairs"] for x in r["vs_baseline"].values())
           for r in variants):       # metrics with no pairs at all (baseline lacks them) don't count
        warnings.append(f"comparisons with fewer than {MIN_PAIRS_FOR_SIGNIFICANCE} paired examples are "
                        "never flagged significant")
    return {
        "experiment": {"id": exp["id"], "name": exp["name"], "status": exp["status"],
                       "dataset_name": exp["dataset_name"], "dataset_hash": exp["dataset_hash"],
                       "n_examples": len(examples)},
        "baseline": baseline, "metrics": metrics, "primary_metric": primary,
        "variants": variants, "regressions": regressions,
        "warnings": list(dict.fromkeys(warnings)),
    }


def format_compare(p: Dict[str, Any]) -> str:
    """Plain-ASCII rendering for the terminal (Windows consoles choke on fancy glyphs)."""
    f = lambda v, spec=".2f": "-" if v is None else format(v, spec)
    sg = lambda v: "-" if v is None else format(v, "+.2f")
    out = [f"experiment {p['experiment']['id']}  [{p['experiment']['status']}]  "
           f"{p['experiment']['n_examples']} examples  baseline = {p['baseline']}", ""]
    head = ["variant", "n", "err", "p50 ms", "p95 ms", "cost USD"] + p["metrics"]
    body = []
    for r in p["variants"]:
        cells = [r["variant"], str(r["n"]), str(r["errors"]), f(r["p50_ms"], ".0f"),
                 f(r["p95_ms"], ".0f"), f(r["cost_usd"], ".4f")]
        for m in p["metrics"]:
            s = r["scores"][m]
            partial = f" ({s['n']}/{r['ok']})" if s["n"] and s["n"] < r["ok"] else ""      # scored on a subset only
            cells.append("-" if s["mean"] is None else
                         f"{s['mean']:.2f}" + ("" if s["ci_low"] is None else f" [{s['ci_low']:.2f},{s['ci_high']:.2f}]") + partial)
        body.append(cells)
    widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    line = lambda c: "  ".join(x.ljust(w) for x, w in zip(c, widths))
    out += [line(head), line(["-" * w for w in widths])] + [line(b) for b in body]
    deltas = [(r["variant"], m, d) for r in p["variants"] for m, d in r["vs_baseline"].items()
              if d["n_paired"] > 0]                      # nothing to pair when the baseline lacks the metric
    if deltas:
        out += ["", "Paired deltas vs baseline (95% bootstrap CI;  * = CI excludes 0 and >= "
                f"{MIN_PAIRS_FOR_SIGNIFICANCE} pairs):"]
        for label, m, d in deltas:
            if not d["enough_pairs"]:                      # a "+1.00 [+1.00,+1.00]" from 2 examples is noise, not a result
                out.append(f"  {label:<18} {m:<28} too few pairs to compare (n={d['n_paired']})")
                continue
            ci = "" if d["ci_low"] is None else f" [{d['ci_low']:+.2f},{d['ci_high']:+.2f}]"
            out.append(f"  {label:<18} {m:<28} {sg(d['delta'])}{ci}  n={d['n_paired']}"
                       f"{'  *' if d['significant'] else ''}")
    for label, reg in p["regressions"].items():
        if reg["count"]:
            shown = f", showing {len(reg['items'])}" if reg["count"] > len(reg["items"]) else ""
            out += ["", f"{label}: {reg['count']} examples worse than baseline on {reg['metric']} (worst first{shown}):"]
            for it in reg["items"]:
                q = (it["input"] or "")[:70].encode("ascii", "replace").decode()
                out.append(f"  {it['example_id']:<18} {it['baseline_score']:.2f} -> {it['candidate_score']:.2f}  {q}")
    if p["warnings"]:
        out += [""] + [f"warning: {w}" for w in p["warnings"]]
    return "\n".join(out)


def examples_payload(store: ExperimentStore, experiment_id: str, metric: Optional[str] = None,
                     sort: str = "worst", limit: int = 25, offset: int = 0) -> Dict[str, Any]:
    """Per-example view: question, reference, and every variant's output with all its scores.

    sort="worst" puts examples where any variant failed to generate first, then by the lowest
    score on the primary metric; sort="dataset" keeps dataset order.
    """
    if sort not in ("worst", "dataset"):
        raise ValueError("sort must be 'worst' or 'dataset'")
    exp = store.get_experiment(experiment_id)
    labels = [v["label"] for v in exp["config"]["variants"]]
    retriever_of = {v["label"]: (v.get("retriever") or {}).get("type") for v in exp["config"]["variants"]}
    examples = store.get_examples(experiment_id)
    docs = {d.id: d for d in store.get_docs(experiment_id)}
    retrievals = store.get_retrievals(experiment_id)
    results = {(r["variant"], r["example_id"]): r for r in store.get_results(experiment_id)}
    table: Dict[str, Dict[str, Dict[str, Any]]] = {}
    found = set()
    for s in store.get_scores(experiment_id):
        found.add(s["metric"])
        table.setdefault(s["example_id"], {}).setdefault(s["variant"], {})[s["metric"]] = (s["value"], s["detail"] or {})
    metrics = sorted(found, key=_rank)
    if metric and metrics and metric not in metrics:
        raise ValueError(f"unknown metric {metric!r}; available: {metrics}")
    primary = metric or default_metric(metrics)

    rows = []
    for idx, ex in enumerate(examples):
        per, vals, failed = {}, [], False
        for label in labels:
            r = results.get((label, ex.id))
            rel = set(ex.relevant_doc_ids)
            retrieved = [{"rank": i, "doc_id": d, "score": sc_, "relevant": (d in rel) if rel else None,
                          "title": docs[d].title if d in docs else None,
                          "text": _clip(docs[d].text, 400) if d in docs else None}
                         for i, (d, sc_) in enumerate(retrievals.get((label, ex.id), []), 1)]
            sc = {m: {"value": v, "verdict": d.get("verdict"),
                      "reason": d.get("reason") or d.get("error") or d.get("note")}
                  for m, (v, d) in table.get(ex.id, {}).get(label, {}).items()}
            if primary in sc and sc[primary]["value"] is not None:
                vals.append(sc[primary]["value"])
            status = r["status"] if r else "missing"
            failed = failed or status != "ok"
            per[label] = {"status": status, "output": _clip(r["output"], 1500) if r else None,
                          "error": r["error"] if r else None, "latency_ms": r["latency_ms"] if r else None,
                          "attempts": r["attempts"] if r else None, "trace_id": r["trace_id"] if r else None,
                          "scores": sc, "retrieved": retrieved, "retriever": retriever_of[label]}
        low = min(vals) if vals else None
        rows.append({"id": ex.id, "index": idx, "input": _clip(ex.input, 1500),
                     "reference": _clip(ex.reference, 1500), "relevant_ids": list(ex.relevant_doc_ids),
                     "primary_min": low, "failed": failed,
                     "variants": per})
    if sort == "worst":
        rows.sort(key=lambda r: (-1.0 if r["failed"] else (2.0 if r["primary_min"] is None else r["primary_min"]),
                                 r["index"]))
    return {"experiment": {"id": exp["id"], "name": exp["name"]}, "metrics": metrics,
            "primary_metric": primary, "variants": labels, "sort": sort,
            "total": len(rows), "offset": offset, "examples": rows[offset:offset + limit]}
