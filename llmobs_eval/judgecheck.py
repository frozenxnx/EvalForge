"""Is the LLM judge trustworthy? Check it against a lexical oracle and against human labels.

1. judge-check: on answers whose reference is a short canonical value, `contains` is a cheap, mostly
   reliable second opinion. Where the two disagree, one of them is wrong; those are the cases to read.
2. label + calibrate: label a sample yourself (blind: the judge's verdict is never shown) and measure
   agreement, Cohen's kappa and leniency bias. Labels drawn at random give unbiased estimates; labels
   drawn from the disagreements are for finding errors and are reported separately.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .metrics import contains, normalize
from .stats import seed_for
from .store import ExperimentStore

LEVELS = (0.0, 0.5, 1.0)
MIN_LABELS = 20
MAX_REFERENCE_TOKENS = 3


def is_checkable(reference: Optional[str], question: str) -> bool:
    """`contains` is a fair second opinion only for a short value that is not already in the question."""
    ref = normalize(reference or "")
    return 1 <= len(ref.split()) <= MAX_REFERENCE_TOKENS and f" {ref} " not in f" {normalize(question)} "


def _clip(s: Optional[str], n: int = 600) -> Optional[str]:
    return s if s is None or len(s) <= n else s[:n] + "..."


def judge_metrics(store: ExperimentStore, experiment_id: str) -> List[str]:
    return sorted({s["metric"] for s in store.get_scores(experiment_id) if s["metric"].startswith("judge:")})


def _pick_metric(available: List[str], requested: Optional[str]) -> Optional[str]:
    if requested:
        if requested not in available:
            raise ValueError(f"unknown judge metric {requested!r}; available: {available}")
        return requested
    return available[0] if available else None


def _judged(store: ExperimentStore, experiment_id: str, metric: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {(s["variant"], s["example_id"]): s for s in store.get_scores(experiment_id)
            if s["metric"] == metric and s["value"] is not None}


# --------------------------------------------------------------------------- judge vs `contains`
def judge_check_payload(store: ExperimentStore, experiment_id: str, metric: Optional[str] = None,
                        top_k: int = 15, only: Optional[Set[Tuple[str, str]]] = None) -> Dict[str, Any]:
    """`only` restricts the check to those (variant, example_id) pairs, so several judges that scored
    different subsets can be compared on the same answers."""
    exp = store.get_experiment(experiment_id)
    available = judge_metrics(store, experiment_id)
    out: Dict[str, Any] = {
        "experiment": {"id": exp["id"], "name": exp["name"]}, "available": available, "metric": None,
        "variants": [], "totals": {"checkable": 0, "agree": 0, "lenient": 0, "strict": 0, "agreement": None},
        "items": []}
    chosen = _pick_metric(available, metric)
    if chosen is None:
        return out
    out["metric"] = chosen
    labels = [v["label"] for v in exp["config"]["variants"]]
    examples = {e.id: e for e in store.get_examples(experiment_id)}
    results = {(r["variant"], r["example_id"]): r for r in store.get_results(experiment_id) if r["status"] == "ok"}
    judged = _judged(store, experiment_id, chosen)
    if only is not None:
        judged = {k: v for k, v in judged.items() if k in only}
    rows = {l: {"variant": l, "checkable": 0, "agree": 0, "lenient": 0, "strict": 0} for l in labels}
    items: Dict[str, List[Dict[str, Any]]] = {"lenient": [], "strict": []}
    for ex_id, ex in examples.items():
        if not is_checkable(ex.reference, ex.input):
            continue
        for label in labels:
            s, r = judged.get((label, ex_id)), results.get((label, ex_id))
            if s is None or r is None:
                continue
            lexical_ok = contains(r["output"], ex.reference or "") == 1.0
            judge_ok = s["value"] >= 0.5
            row = rows[label]
            row["checkable"] += 1
            if judge_ok == lexical_ok:
                row["agree"] += 1
                continue
            kind = "lenient" if judge_ok else "strict"
            row[kind] += 1
            detail = s["detail"] or {}
            items[kind].append({
                "kind": kind, "variant": label, "example_id": ex_id, "input": _clip(ex.input),
                "reference": _clip(ex.reference), "output": _clip(r["output"]), "judge_value": s["value"],
                "judge_verdict": detail.get("verdict"), "judge_reason": detail.get("reason"),
                "contains": 1.0 if lexical_ok else 0.0})
    for row in rows.values():
        row["agreement"] = row["agree"] / row["checkable"] if row["checkable"] else None
    out["variants"] = list(rows.values())
    t = out["totals"]
    for key in ("checkable", "agree", "lenient", "strict"):
        t[key] = sum(r[key] for r in rows.values())
    t["agreement"] = t["agree"] / t["checkable"] if t["checkable"] else None
    items["lenient"].sort(key=lambda i: -i["judge_value"])          # stable: dataset order within a level
    out["items"] = items["lenient"][:top_k] + items["strict"][:top_k]
    return out


# --------------------------------------------------------------------------- agreement with humans
def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> Optional[float]:
    """Chance-corrected agreement. None when it is undefined (no variation to agree about)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == k for x in a) / n) * (sum(y == k for y in b) / n) for k in set(a) | set(b))
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def _stats(pairs: List[Tuple[float, float, str]], n_boot: int, seed: int) -> Optional[Dict[str, Any]]:
    n = len(pairs)
    if n == 0:
        return None
    judge = [p[0] for p in pairs]
    human = [p[1] for p in pairs]
    jb, hb = [x == 1.0 for x in judge], [x == 1.0 for x in human]
    rng, boots = random.Random(seed), []
    for _ in range(n_boot):
        idx = rng.choices(range(n), k=n)
        k = cohen_kappa([jb[i] for i in idx], [hb[i] for i in idx])
        if k is not None:
            boots.append(k)
    boots.sort()
    ci = (boots[int(0.025 * len(boots))], boots[min(len(boots) - 1, int(0.975 * len(boots)))]) \
        if len(boots) >= n_boot / 2 else (None, None)
    per_variant: Dict[str, List[float]] = {}
    for j, h, v in pairs:
        per_variant.setdefault(v, []).append(j - h)
    return {
        "n": n,
        "agreement": sum(x == y for x, y in zip(judge, human)) / n,           # exact match on the 3 levels
        "kappa": cohen_kappa(judge, human),
        "binary_agreement": sum(x == y for x, y in zip(jb, hb)) / n,          # correct vs not correct
        "binary_kappa": cohen_kappa(jb, hb),
        "binary_kappa_ci": list(ci),
        "bias": sum(judge) / n - sum(human) / n,                              # > 0: the judge is more lenient
        "bias_by_variant": {v: sum(d) / len(d) for v, d in per_variant.items()},
        "confusion": {"judge_rows": list(LEVELS), "human_cols": list(LEVELS),
                      "counts": [[sum(1 for j, h, _ in pairs if j == jl and h == hl) for hl in LEVELS]
                                 for jl in LEVELS]},
    }


def calibration_payload(store: ExperimentStore, experiment_id: str, metric: Optional[str] = None,
                        labeler: Optional[str] = None, n_boot: int = 1000) -> Dict[str, Any]:
    exp = store.get_experiment(experiment_id)
    available = judge_metrics(store, experiment_id)
    chosen = _pick_metric(available, metric)
    out: Dict[str, Any] = {"experiment": {"id": exp["id"], "name": exp["name"]}, "available": available,
                           "metric": chosen, "n_labels": 0, "all": None, "random": None, "headline": None,
                           "warnings": []}
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for lab in store.get_labels(experiment_id, labeler):                     # ordered by time: last one wins
        latest[(lab["variant"], lab["example_id"])] = lab
    out["n_labels"] = len(latest)
    if chosen is None:
        out["warnings"].append("no judge scores yet: run `evaluate --judge PROVIDER:MODEL` first")
        return out
    judged = _judged(store, experiment_id, chosen)
    pairs_all, pairs_random = [], []
    for key, lab in latest.items():
        s = judged.get(key)
        if s is None:
            continue
        pair = (s["value"], lab["label"], key[0])
        pairs_all.append(pair)
        if lab["source"] == "random":
            pairs_random.append(pair)
    seed = seed_for(experiment_id, chosen)
    out["all"] = _stats(pairs_all, n_boot, seed)
    out["random"] = _stats(pairs_random, n_boot, seed)
    out["headline"] = "random" if len(pairs_random) >= 10 else ("all" if pairs_all else None)
    if not pairs_all:
        out["warnings"].append("no labels yet: run `python -m llmobs_eval label <id>` to add some")
    elif len(pairs_all) < MIN_LABELS:
        out["warnings"].append(f"only {len(pairs_all)} labels: at least {MIN_LABELS} are needed for a stable "
                               "estimate, and the interval shows how unsure this still is")
    if pairs_all and not pairs_random:
        out["warnings"].append("none of the labels came from a random sample, so agreement is biased toward "
                               "hard cases; label a random sample for an honest estimate")
    elif pairs_random and out["headline"] == "all":
        out["warnings"].append(f"only {len(pairs_random)} random labels; the headline mixes in disagreement-picked ones")
    return out


def judge_table_payload(store: ExperimentStore, experiment_id: str, labeler: Optional[str] = None,
                        n_boot: int = 500) -> Dict[str, Any]:
    """Every judge metric on this experiment, measured against the same human labels (labels belong to
    the answers, not to a judge), so judges and rubrics can be compared without labeling again."""
    exp = store.get_experiment(experiment_id)
    rows = []
    metrics = judge_metrics(store, experiment_id)
    # with several judges, count lenient/strict on the answers ALL of them scored (a judge run with
    # --only-labeled has scored far fewer answers than one run on everything)
    common = set.intersection(*[set(_judged(store, experiment_id, m)) for m in metrics]) if len(metrics) > 1 else None
    for m in metrics:
        cal = calibration_payload(store, experiment_id, m, labeler, n_boot)
        stats = cal[cal["headline"]] if cal["headline"] else None
        totals = judge_check_payload(store, experiment_id, m, top_k=0, only=common)["totals"]
        rows.append({"metric": m, "headline": cal["headline"], "n": stats["n"] if stats else 0,
                     "agreement": stats["agreement"] if stats else None,
                     "binary_agreement": stats["binary_agreement"] if stats else None,
                     "kappa": stats["kappa"] if stats else None,
                     "binary_kappa": stats["binary_kappa"] if stats else None,
                     "binary_kappa_ci": stats["binary_kappa_ci"] if stats else [None, None],
                     "bias": stats["bias"] if stats else None,
                     "bias_by_variant": stats["bias_by_variant"] if stats else {},
                     "checkable": totals["checkable"], "lenient": totals["lenient"], "strict": totals["strict"]})
    return {"experiment": {"id": exp["id"], "name": exp["name"]}, "judges": rows,
            "common_items": None if common is None else len(common),
            "n_labels": len({(l["variant"], l["example_id"]) for l in store.get_labels(experiment_id, labeler)})}


# --------------------------------------------------------------------------- labeling
def pick_label_items(store: ExperimentStore, experiment_id: str, metric: Optional[str] = None, n: int = 20,
                     sample: str = "random", seed: int = 0, labeler: str = "human") -> List[Tuple[str, str, str]]:
    """(variant, example_id, source) triples to label. `random` gives an unbiased calibration set;
    `disagreements` picks only judge/contains disagreements, for finding the judge's mistakes."""
    if sample not in ("random", "disagreements"):
        raise ValueError("sample must be 'random' or 'disagreements'")
    available = judge_metrics(store, experiment_id)
    chosen = _pick_metric(available, metric)
    if chosen is None:
        raise ValueError("no judge scores yet: run `evaluate --judge PROVIDER:MODEL` first")
    done = {(l["variant"], l["example_id"]) for l in store.get_labels(experiment_id, labeler)}
    results = {(r["variant"], r["example_id"]) for r in store.get_results(experiment_id) if r["status"] == "ok"}
    pool = sorted(k for k in _judged(store, experiment_id, chosen) if k in results and k not in done)
    if sample == "disagreements":
        flagged = {(i["variant"], i["example_id"])
                   for i in judge_check_payload(store, experiment_id, chosen, top_k=10 ** 6)["items"]}
        pool = [k for k in pool if k in flagged]
    rng = random.Random(seed)
    rng.shuffle(pool)
    source = "random" if sample == "random" else "disagreement"
    return [(v, e, source) for v, e in pool[:n]]


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode()


def label_session(store: ExperimentStore, experiment_id: str, items: Sequence[Tuple[str, str, str]],
                  ask: Optional[Callable[[str], str]] = None, out: Optional[Callable[[str], None]] = None,
                  labeler: str = "human") -> int:
    """Blind labeling loop: shows question, reference and answer only (never the judge's or contains' verdict).

    `ask` and `out` default to the built-in input() and print(), looked up when called (not when defined),
    so they can be patched."""
    ask = ask if ask is not None else input
    out = out if out is not None else print
    examples = {e.id: e for e in store.get_examples(experiment_id)}
    results = {(r["variant"], r["example_id"]): r for r in store.get_results(experiment_id)}
    keys = {"c": 1.0, "p": 0.5, "i": 0.0}
    saved = 0
    for pos, (variant, ex_id, source) in enumerate(items, 1):
        ex, res = examples[ex_id], results[(variant, ex_id)]
        out(f"\n[{pos}/{len(items)}] " + "-" * 50)
        out("Question : " + _ascii(ex.input))
        out("Reference: " + _ascii(ex.reference or "(none)"))
        out("Answer   : " + _ascii((res["output"] or "")[:800]))
        while True:
            try:
                reply = ask("Is the answer  correct [c]  partially correct [p]  incorrect [i]  skip [s]  quit [q] > ")
            except (EOFError, KeyboardInterrupt):
                return saved
            reply = reply.strip().lower()
            if reply in keys:
                store.add_label(experiment_id, variant, ex_id, keys[reply], labeler, source)
                saved += 1
                break
            if reply == "s":
                break
            if reply == "q":
                return saved
            out("please type c, p, i, s or q")
    return saved


# --------------------------------------------------------------------------- terminal rendering (ASCII)
def format_judge_check(p: Dict[str, Any]) -> str:
    if not p["metric"]:
        return "no judge scores yet: run `python -m llmobs_eval evaluate <id> --judge PROVIDER:MODEL` first"
    t = p["totals"]
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    out = [f"judge check: {p['metric']} vs `contains`, on answers whose reference is a short value "
           f"(experiment {p['experiment']['id']})", ""]
    head = ["variant", "checkable", "agree", "judge too lenient", "judge too strict"]
    body = [[r["variant"], str(r["checkable"]), pct(r["agreement"]), str(r["lenient"]), str(r["strict"])]
            for r in p["variants"]] + [["ALL", str(t["checkable"]), pct(t["agreement"]), str(t["lenient"]), str(t["strict"])]]
    widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    line = lambda c: "  ".join(x.ljust(w) for x, w in zip(c, widths))
    out += [line(head), line(["-" * w for w in widths])] + [line(b) for b in body]
    out += ["", "lenient = the judge credits an answer that does not contain the reference value",
            "strict  = the judge rejects an answer that does. Either side can be the wrong one: read them."]
    for it in p["items"]:
        out += ["", f"[{it['kind']}] {it['variant']} / {it['example_id']}  judge={it['judge_value']:.1f} "
                    f"({it['judge_verdict']})  contains={it['contains']:.0f}",
                "   Q: " + _ascii(it["input"] or ""), "   reference: " + _ascii(it["reference"] or ""),
                "   answer   : " + _ascii((it["output"] or "")[:200].replace("\n", " ")),
                "   judge said: " + _ascii(it["judge_reason"] or "")]
    if len(p["available"]) > 1:
        out += ["", f"other judge metrics available: {', '.join(m for m in p['available'] if m != p['metric'])}"]
    return "\n".join(out)


def format_calibration(p: Dict[str, Any]) -> str:
    out = [f"judge calibration: {p['metric'] or '-'}  (experiment {p['experiment']['id']}, {p['n_labels']} human labels)"]
    fmt = lambda x, spec=".2f": "n/a" if x is None else format(x, spec)
    for name in ("random", "all"):
        s = p[name]
        if not s:
            continue
        lo, hi = s["binary_kappa_ci"]
        ci = "" if lo is None else f" [{lo:.2f}, {hi:.2f}]"
        out += ["", f"{name} sample, n={s['n']}" + ("   <- headline" if p["headline"] == name else ""),
                f"  agreement (3 levels)      {s['agreement']:.0%}     kappa {fmt(s['kappa'])}",
                f"  agreement (correct or not) {s['binary_agreement']:.0%}     kappa {fmt(s['binary_kappa'])}{ci}",
                f"  judge leniency bias       {s['bias']:+.2f}   (> 0: judge more lenient than you)"]
        out.append("  by variant: " + ", ".join(f"{v} {b:+.2f}" for v, b in s["bias_by_variant"].items()))
        c = s["confusion"]["counts"]
        out += ["  confusion (rows = judge, cols = you; levels 0 / 0.5 / 1):"] + \
               [f"     judge {lv:>3}: {row}" for lv, row in zip(s["confusion"]["judge_rows"], c)]
    out += [""] + [f"warning: {w}" for w in p["warnings"]]
    out.append("rule of thumb: kappa > 0.8 is strong agreement, 0.6-0.8 substantial, below 0.4 weak.")
    return "\n".join(out)


def format_judge_table(p: Dict[str, Any]) -> str:
    if not p["judges"]:
        return "no judge scores yet: run `python -m llmobs_eval evaluate <id> --judge PROVIDER:MODEL` first"
    f = lambda x, spec=".2f": "-" if x is None else format(x, spec)
    head = ["judge metric", "labels", "agree", "kappa (correct or not) [95% CI]", "bias", "vs contains: lenient/strict"]
    body = []
    for r in p["judges"]:
        lo, hi = r["binary_kappa_ci"]
        kappa = f(r["binary_kappa"]) + ("" if lo is None else f" [{lo:.2f}, {hi:.2f}]")
        body.append([r["metric"], str(r["n"]), f(r["binary_agreement"], ".0%"), kappa, f(r["bias"], "+.2f"),
                     f"{r['lenient']}/{r['strict']} of {r['checkable']}"])
    widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    line = lambda c: "  ".join(x.ljust(w) for x, w in zip(c, widths))
    out = [f"judges compared on the same {p['n_labels']} human labels (experiment {p['experiment']['id']})", "",
           line(head), line(["-" * w for w in widths])] + [line(b) for b in body]
    if p.get("common_items") is not None:
        out += ["", f"lenient/strict are counted on the {p['common_items']} answers every judge scored, so the judges are comparable"]
    out += ["", "bias by variant (positive = more lenient than you):"]
    out += [f"  {r['metric']}: " + (", ".join(f"{v} {b:+.2f}" for v, b in r["bias_by_variant"].items()) or "-")
            for r in p["judges"]]
    out += ["", "A judge is only comparable on the labels it has scored (see `labels`). With few labels the interval is wide:",
            "prefer a clear difference (a big bias gap or a kappa gap well beyond the intervals) over a small one."]
    return "\n".join(out)
