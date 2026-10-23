"""CI gating: decide whether a candidate is significantly worse than a baseline, and fail the build if so.

Principles
- Compare *paired* (same examples) with a bootstrap interval, so example difficulty cancels out.
- Fail only when the drop is bigger than you allow AND not explained by noise. A drop that is bigger than
  allowed but cannot be told from noise is "inconclusive": a warning by default, a failure with
  `on_inconclusive: fail`.
- A gate that cannot evaluate a rule never passes silently: a missing metric, no overlapping examples or a
  typo in the config is an error (exit code 2), not a green build.
- Generation errors count as failures (score 0), so a change that breaks calls cannot hide behind the
  examples that still worked.
- A judge-based metric whose reliability is unknown or poor cannot produce a clean pass (`judge_trust`).

Exit codes: 0 pass (warnings allowed) | 1 fail | 2 error / misconfiguration.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set

from .judgecheck import calibration_payload
from .providers import ConfigError
from .report import percentile
from .stats import MIN_PAIRS_FOR_SIGNIFICANCE, paired_bootstrap, seed_for
from .store import ExperimentStore

SNAPSHOT_FORMAT = 1
EXIT_CODES = {"pass": 0, "warn": 0, "fail": 1, "error": 2}
_TOP_KEYS = {"description", "baseline", "candidates", "metrics", "max_error_rate", "max_p95_ratio",
             "on_inconclusive", "judge_trust", "judge_calibration", "alpha", "n_boot", "min_pairs",
             "bonferroni", "errors_as_zero"}


# --------------------------------------------------------------------------- rules
@dataclass
class MetricRule:
    name: str
    max_drop: float = 0.0              # the largest drop of the mean that is acceptable
    min: Optional[float] = None        # absolute floor for the candidate's mean


@dataclass
class GateRules:
    metrics: List[MetricRule]
    baseline: Optional[str] = None
    candidates: Optional[List[str]] = None
    max_error_rate: Optional[float] = None
    max_p95_ratio: Optional[float] = None
    on_inconclusive: str = "warn"
    judge_trust: Optional[Dict[str, float]] = None
    judge_calibration: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    alpha: float = 0.05
    n_boot: int = 2000
    min_pairs: int = MIN_PAIRS_FOR_SIGNIFICANCE
    bonferroni: bool = False
    errors_as_zero: bool = True


def _num(value: Any, what: str, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{what} must be a number, got {value!r}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ConfigError(f"{what} must be between {lo} and {hi}, got {value}")
    return float(value)


def parse_rules(cfg: Mapping[str, Any]) -> GateRules:
    """Validate a gate config. Unknown keys are errors: a typo in a gate must not silently weaken it."""
    unknown = set(cfg) - _TOP_KEYS
    if unknown:
        raise ConfigError(f"unknown gate setting(s) {sorted(unknown)}; allowed: {sorted(_TOP_KEYS)}")
    raw = cfg.get("metrics")
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("the gate needs a non-empty 'metrics' object, e.g. {\"contains\": {\"max_drop\": 0.03}}")
    metrics = []
    for name, spec in raw.items():
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            spec = {"max_drop": spec}
        if not isinstance(spec, dict):
            raise ConfigError(f"metrics[{name!r}] must be an object like {{\"max_drop\": 0.03}}")
        bad = set(spec) - {"max_drop", "min"}
        if bad:
            raise ConfigError(f"metrics[{name!r}]: unknown setting(s) {sorted(bad)}; allowed: max_drop, min")
        metrics.append(MetricRule(name, _num(spec.get("max_drop", 0.0), f"metrics[{name!r}].max_drop", 0.0),
                                  _num(spec["min"], f"metrics[{name!r}].min") if "min" in spec else None))
    on = cfg.get("on_inconclusive", "warn")
    if on not in ("warn", "fail"):
        raise ConfigError("on_inconclusive must be 'warn' or 'fail'")
    trust = cfg.get("judge_trust")
    if trust is not None:
        if not isinstance(trust, dict) or set(trust) - {"min_kappa", "min_labels"}:
            raise ConfigError("judge_trust must be an object with min_kappa and/or min_labels")
        trust = {"min_kappa": _num(trust.get("min_kappa", 0.6), "judge_trust.min_kappa", -1.0, 1.0),
                 "min_labels": _num(trust.get("min_labels", 20), "judge_trust.min_labels", 1.0)}
    ledger = cfg.get("judge_calibration") or {}
    if not isinstance(ledger, dict) or any(not isinstance(v, dict) for v in ledger.values()):
        raise ConfigError("judge_calibration must map a judge metric name to an object like {\"kappa\": 0.74, \"labels\": 25}")
    return GateRules(
        metrics=metrics, baseline=cfg.get("baseline"), candidates=cfg.get("candidates"),
        max_error_rate=_num(cfg["max_error_rate"], "max_error_rate", 0.0, 1.0) if "max_error_rate" in cfg else None,
        max_p95_ratio=_num(cfg["max_p95_ratio"], "max_p95_ratio", 0.0) if "max_p95_ratio" in cfg else None,
        on_inconclusive=on, judge_trust=trust, judge_calibration=ledger,
        alpha=_num(cfg.get("alpha", 0.05), "alpha", 0.0001, 0.5),
        n_boot=int(_num(cfg.get("n_boot", 2000), "n_boot", 100.0)),
        min_pairs=int(_num(cfg.get("min_pairs", MIN_PAIRS_FOR_SIGNIFICANCE), "min_pairs", 2.0)),
        bonferroni=bool(cfg.get("bonferroni", False)), errors_as_zero=bool(cfg.get("errors_as_zero", True)))


def load_gate_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"gate config not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})") from None
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: the gate config must be a JSON object")
    return cfg


# --------------------------------------------------------------------------- the two things being compared
@dataclass
class Side:
    label: str
    scores: Dict[str, Dict[str, float]]        # metric -> example id -> value (unscored items are absent)
    failed: Set[str]                           # example ids with no usable output (error or missing)
    n_examples: int
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    outputs: Dict[str, str]
    meta: Dict[str, Any]


def _clip(s: Optional[str], n: int = 400) -> Optional[str]:
    return s if s is None or len(s) <= n else s[:n] + "..."


def side_from_experiment(store: ExperimentStore, experiment_id: str, label: str) -> Side:
    exp = store.get_experiment(experiment_id)
    variants = {v["label"]: v for v in exp["config"]["variants"]}
    if label not in variants:
        raise ConfigError(f"unknown variant {label!r}; this experiment has {list(variants)}")
    examples = store.get_examples(experiment_id)
    ok = {r["example_id"]: r for r in store.get_results(experiment_id, label) if r["status"] == "ok"}
    scores: Dict[str, Dict[str, float]] = {}
    for s in store.get_scores(experiment_id):
        if s["variant"] == label and s["value"] is not None:
            scores.setdefault(s["metric"], {})[s["example_id"]] = s["value"]
    latencies = [r["latency_ms"] for r in ok.values() if r["latency_ms"] is not None]
    corpus = store.get_corpus_info(experiment_id)
    return Side(
        label=label, scores=scores, failed={e.id for e in examples if e.id not in ok}, n_examples=len(examples),
        p50_ms=percentile(latencies, 50), p95_ms=percentile(latencies, 95),
        outputs={i: r["output"] for i, r in ok.items()},
        meta={"source": "experiment", "experiment_id": exp["id"], "experiment_name": exp["name"],
              "dataset_name": exp["dataset_name"], "dataset_hash": exp["dataset_hash"],
              "corpus_hash": corpus["content_hash"] if corpus else None, "model": variants[label]["model"]})


def export_snapshot(store: ExperimentStore, experiment_id: str, label: str, path: str) -> Dict[str, Any]:
    """Write a variant's per-example scores to a JSON file: a baseline that can be committed next to the code
    (like a golden file) so CI only has to run the candidate."""
    exp = store.get_experiment(experiment_id)
    if exp["status"] != "complete":
        raise ConfigError(f"experiment is {exp['status']}: finish it before taking a baseline snapshot")
    side = side_from_experiment(store, experiment_id, label)
    snap = {
        "format": SNAPSHOT_FORMAT, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "variant": label,
        "model": side.meta["model"], "experiment": {"id": exp["id"], "name": exp["name"]},
        "dataset": {"name": exp["dataset_name"], "hash": exp["dataset_hash"], "n_examples": side.n_examples},
        "corpus_hash": side.meta["corpus_hash"], "failed": sorted(side.failed),
        "latency_ms": {"p50": side.p50_ms, "p95": side.p95_ms},
        "scores": {m: dict(sorted(v.items())) for m, v in sorted(side.scores.items())},
        "outputs": {i: _clip(o, 300) for i, o in sorted(side.outputs.items())},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=1, ensure_ascii=False)
    return snap


def side_from_snapshot(path: str) -> Side:
    try:
        with open(path, encoding="utf-8") as fh:
            snap = json.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"baseline snapshot not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON ({exc.msg})") from None
    if not isinstance(snap, dict) or snap.get("format") != SNAPSHOT_FORMAT:
        raise ConfigError(f"{path}: not a baseline snapshot (expected format {SNAPSHOT_FORMAT}); "
                          "create one with `baseline-export`")
    lat = snap.get("latency_ms") or {}
    return Side(
        label=f"{snap['variant']} (snapshot {snap['created'][:10]})", scores=snap["scores"],
        failed=set(snap.get("failed") or []), n_examples=snap["dataset"]["n_examples"],
        p50_ms=lat.get("p50"), p95_ms=lat.get("p95"), outputs=snap.get("outputs") or {},
        meta={"source": "snapshot", "experiment_id": snap["experiment"]["id"], "dataset_name": snap["dataset"]["name"],
              "dataset_hash": snap["dataset"]["hash"], "corpus_hash": snap.get("corpus_hash"), "model": snap.get("model")})


def check_compatible(baseline: Side, candidate: Side, allow_dataset_mismatch: bool = False) -> List[str]:
    """Refuse to compare runs of different datasets: the numbers would not mean anything."""
    warnings: List[str] = []
    bh, ch = baseline.meta.get("dataset_hash"), candidate.meta.get("dataset_hash")
    if bh and ch and bh != ch:
        msg = (f"the baseline was recorded on a different dataset ({str(bh)[:10]} vs {str(ch)[:10]})")
        if not allow_dataset_mismatch:
            raise ConfigError(msg + ": regenerate the baseline, or pass --allow-dataset-mismatch to compare on the "
                                    "examples they share")
        warnings.append(msg + "; comparing only the examples both contain")
    bc, cc = baseline.meta.get("corpus_hash"), candidate.meta.get("corpus_hash")
    if bc != cc and (bc or cc):
        warnings.append("the document corpus differs from the baseline's; retrieval changes are part of this comparison")
    return warnings


# --------------------------------------------------------------------------- judge trust
def judge_trust(store: ExperimentStore, experiment_id: str, rules: GateRules) -> Dict[str, Dict[str, Any]]:
    """Is each gated judge metric reliable enough to gate on? Local human labels win; else the calibration
    ledger in the gate config (so the evidence is reviewed in the same pull request as the gate)."""
    need = rules.judge_trust
    if not need:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for rule in rules.metrics:
        if not rule.name.startswith("judge:"):
            continue
        kappa = labels = source = None
        try:
            cal = calibration_payload(store, experiment_id, rule.name, n_boot=200)
            if cal["headline"]:
                s = cal[cal["headline"]]
                kappa, labels, source = s["binary_kappa"], s["n"], "local human labels"
        except ValueError:
            pass
        if source is None and rule.name in rules.judge_calibration:
            led = rules.judge_calibration[rule.name]
            kappa, labels, source = led.get("kappa"), int(led.get("labels") or 0), "judge_calibration in the gate config"
        if source is None:
            out[rule.name] = {"trusted": False, "source": None, "kappa": None, "labels": 0,
                              "reason": f"no calibration on record for {rule.name}: label answers and run "
                                        "`calibrate <id> --record <gate.json>`"}
            continue
        problems = []
        if labels < need["min_labels"]:
            problems.append(f"only {labels} labels (need {int(need['min_labels'])})")
        if kappa is None:
            problems.append("kappa is undefined")
        elif kappa < need["min_kappa"]:
            problems.append(f"kappa {kappa:.2f} is below {need['min_kappa']:.2f}")
        out[rule.name] = {"trusted": not problems, "source": source, "kappa": kappa, "labels": labels,
                          "reason": f"{rule.name} is not trusted ({source}): " + "; ".join(problems) if problems else
                                    f"calibrated: kappa {kappa:.2f} on {labels} labels ({source})"}
    return out


def record_calibration(path: str, metric: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Write a judge's measured calibration into a gate config's ledger (creating the file if needed)."""
    stats = payload.get(payload.get("headline") or "") if payload.get("headline") else None
    if not stats:
        raise ConfigError("no human labels to record: run `label` first")
    entry = {"kappa": None if stats["binary_kappa"] is None else round(stats["binary_kappa"], 3),
             "labels": stats["n"], "agreement": round(stats["binary_agreement"], 3), "bias": round(stats["bias"], 3),
             "sample": payload["headline"], "recorded": time.strftime("%Y-%m-%d")}
    cfg = load_gate_config(path) if os.path.exists(path) else {}
    cfg.setdefault("judge_calibration", {})[metric] = entry
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return entry


# --------------------------------------------------------------------------- the decision
def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _effective(verdict: str, rules: GateRules) -> str:
    if verdict in ("inconclusive", "untrusted"):
        return "fail" if rules.on_inconclusive == "fail" else "warn"
    return verdict                                   # pass | fail | error


def _metric_check(rule: MetricRule, baseline: Side, candidate: Side, rules: GateRules, alpha: float,
                  trust: Mapping[str, Dict[str, Any]], questions: Mapping[str, str], top_k: int):
    check: Dict[str, Any] = {"kind": "metric", "metric": rule.name, "max_drop": rule.max_drop, "min": rule.min,
                             "baseline_mean": None, "candidate_mean": None, "delta": None, "ci_low": None,
                             "ci_high": None, "n_pairs": 0, "n_filled": 0, "enough_pairs": False,
                             "verdict": "pass", "reasons": [], "trust": None}

    def error(msg):
        check.update(verdict="error", effective="error", reasons=[msg])
        return check, []

    b_scores, c_scores = baseline.scores.get(rule.name) or {}, candidate.scores.get(rule.name) or {}
    if not b_scores:
        return error(f"the baseline has no scores for {rule.name!r} (was it evaluated?)")
    if not c_scores:
        return error(f"the candidate has no scores for {rule.name!r} (was it evaluated?)")
    ids = [i for i in sorted(b_scores) if i in c_scores or (rules.errors_as_zero and i in candidate.failed)]
    if not ids:
        return error("the baseline and the candidate share no scored examples")
    a, b = [b_scores[i] for i in ids], [c_scores.get(i, 0.0) for i in ids]
    pb = paired_bootstrap(a, b, rules.n_boot, alpha, seed_for(baseline.label, candidate.label, rule.name),
                          rules.min_pairs)
    cm = _mean(b)
    check.update(baseline_mean=_mean(a), candidate_mean=cm, delta=pb["delta"], ci_low=pb["ci_low"],
                 ci_high=pb["ci_high"], n_pairs=pb["n_paired"], enough_pairs=pb["enough_pairs"],
                 n_filled=sum(1 for i in ids if i not in c_scores))
    if check["n_filled"]:
        check["reasons"].append(f"{check['n_filled']} failed generations counted as 0")
    if rule.min is not None and cm < rule.min:
        check["verdict"] = "fail"
        check["reasons"].append(f"candidate mean {cm:.3f} is below the required minimum {rule.min}")
    regressed = pb["delta"] is not None and pb["delta"] < -rule.max_drop
    if regressed:
        if pb["significant"]:
            check["verdict"] = "fail"
            check["reasons"].append(f"dropped {-pb['delta']:.3f} (allowed {rule.max_drop}) and the drop is statistically significant")
        else:
            if check["verdict"] != "fail":
                check["verdict"] = "inconclusive"
            why = (f"only {pb['n_paired']} paired examples (need {rules.min_pairs})" if not pb["enough_pairs"]
                   else "the interval includes 0")
            res = "" if (pb["ci_low"] is None or not pb["enough_pairs"]) else (
                f"; this comparison can only resolve differences down to about +/-{(pb['ci_high'] - pb['ci_low']) / 2:.2f}: "
                "add examples or raise max_drop")
            check["reasons"].append(f"dropped {-pb['delta']:.3f} (allowed {rule.max_drop}) but this cannot be told from noise: {why}{res}")
    t = trust.get(rule.name)
    if t is not None:
        check["trust"] = t
        if not t["trusted"]:
            if check["verdict"] == "pass":
                check["verdict"] = "untrusted"
            check["reasons"].append(t["reason"])
    check["effective"] = _effective(check["verdict"], rules)
    worst = []
    if regressed:
        diffs = sorted((y - x, i) for i, x, y in zip(ids, a, b) if y < x)
        for d, i in diffs[:top_k]:
            c_out = candidate.outputs.get(i)
            if c_out is None:
                c_out = "(generation failed)" if i in candidate.failed else "(output not recorded)"
            b_out = baseline.outputs.get(i)
            worst.append({"example_id": i, "question": _clip(questions.get(i), 200), "baseline": b_scores[i],
                          "candidate": c_scores.get(i, 0.0),
                          "baseline_output": _clip(b_out, 300) if b_out is not None else "(output not recorded)",
                          "candidate_output": _clip(c_out, 300)})
    return check, worst


def run_gate(baseline: Side, candidate: Side, rules: GateRules, questions: Optional[Mapping[str, str]] = None,
             trust: Optional[Mapping[str, Dict[str, Any]]] = None, top_k: int = 5) -> Dict[str, Any]:
    questions, trust = questions or {}, trust or {}
    alpha = rules.alpha / len(rules.metrics) if rules.bonferroni else rules.alpha
    checks: List[Dict[str, Any]] = []
    regressions: Dict[str, List[Dict[str, Any]]] = {}
    for rule in rules.metrics:
        check, worst = _metric_check(rule, baseline, candidate, rules, alpha, trust, questions, top_k)
        checks.append(check)
        if worst:
            regressions[rule.name] = worst
    if rules.max_error_rate is not None:
        rate = len(candidate.failed) / candidate.n_examples if candidate.n_examples else 0.0
        bad = rate > rules.max_error_rate
        checks.append({"kind": "error_rate", "metric": "generation errors", "baseline_mean": len(baseline.failed) / max(1, baseline.n_examples),
                       "candidate_mean": rate, "limit": rules.max_error_rate, "verdict": "fail" if bad else "pass",
                       "effective": "fail" if bad else "pass",
                       "reasons": [f"{len(candidate.failed)} of {candidate.n_examples} generations failed "
                                   f"(limit {rules.max_error_rate:.0%})"] if bad else []})
    if rules.max_p95_ratio is not None:
        c = {"kind": "latency", "metric": "p95 latency (ms)", "baseline_mean": baseline.p95_ms,
             "candidate_mean": candidate.p95_ms, "limit": rules.max_p95_ratio, "reasons": []}
        if not baseline.p95_ms or candidate.p95_ms is None:
            c.update(verdict="error", effective="error", reasons=["latency is not available for both sides"])
        else:
            ratio = candidate.p95_ms / baseline.p95_ms
            bad = ratio > rules.max_p95_ratio
            c.update(ratio=ratio, verdict="fail" if bad else "pass", effective="fail" if bad else "pass",
                     reasons=[f"p95 latency is {ratio:.2f}x the baseline's (limit {rules.max_p95_ratio}x)"] if bad else [])
        checks.append(c)
    effs = [c["effective"] for c in checks]
    status = "error" if "error" in effs else "fail" if "fail" in effs else "warn" if "warn" in effs else "pass"
    return {"status": status, "exit_code": EXIT_CODES[status],
            "baseline": {"label": baseline.label, "n_examples": baseline.n_examples, "failed": len(baseline.failed),
                         "meta": baseline.meta},
            "candidate": {"label": candidate.label, "n_examples": candidate.n_examples, "failed": len(candidate.failed),
                          "meta": candidate.meta},
            "checks": checks, "regressions": regressions, "warnings": [],
            "settings": {"alpha": alpha, "n_boot": rules.n_boot, "min_pairs": rules.min_pairs,
                         "on_inconclusive": rules.on_inconclusive, "errors_as_zero": rules.errors_as_zero},
            "summary": {k: effs.count(k) for k in ("pass", "warn", "fail", "error")}}


def gate_experiment(store: ExperimentStore, experiment_id: str, rules: GateRules, *,
                    baseline_label: Optional[str] = None, candidates: Optional[List[str]] = None,
                    baseline_file: Optional[str] = None, allow_dataset_mismatch: bool = False,
                    top_k: int = 5) -> Dict[str, Any]:
    exp = store.get_experiment(experiment_id)
    if exp["status"] != "complete":
        raise ConfigError(f"experiment is {exp['status']}: finish it first "
                          f"(python -m llmobs_eval resume {experiment_id}); a gate on partial results proves nothing")
    labels = [v["label"] for v in exp["config"]["variants"]]
    questions = {e.id: e.input for e in store.get_examples(experiment_id)}
    if baseline_file:
        base = side_from_snapshot(baseline_file)
        wanted = candidates or rules.candidates or labels
    else:
        bl = baseline_label or rules.baseline or labels[0]
        base = side_from_experiment(store, experiment_id, bl)
        wanted = candidates or rules.candidates or [l for l in labels if l != bl]
    if not wanted:
        raise ConfigError("there is no candidate variant to gate")
    trust = judge_trust(store, experiment_id, rules)
    reports = []
    for label in wanted:
        cand = side_from_experiment(store, experiment_id, label)
        notes = check_compatible(base, cand, allow_dataset_mismatch)
        report = run_gate(base, cand, rules, questions, trust, top_k)
        report["warnings"] = notes
        reports.append(report)
    statuses = [r["status"] for r in reports]
    status = "error" if "error" in statuses else "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    return {"status": status, "exit_code": EXIT_CODES[status],
            "experiment": {"id": exp["id"], "name": exp["name"], "dataset": exp["dataset_name"]}, "reports": reports}


# --------------------------------------------------------------------------- rendering
def _fmt(x: Optional[float], spec: str = ".3f") -> str:
    return "-" if x is None else format(x, spec)


def _row(c: Dict[str, Any]) -> List[str]:
    if c["kind"] == "metric":
        delta = "-" if c["delta"] is None else (f"{c['delta']:+.3f}" + ("" if c["ci_low"] is None else f" [{c['ci_low']:+.3f},{c['ci_high']:+.3f}]"))
        allowed = f"-{c['max_drop']:g}" + ("" if c["min"] is None else f", min {c['min']:g}")
        return [c["metric"], _fmt(c["baseline_mean"]), _fmt(c["candidate_mean"]), delta, allowed, c["effective"].upper()
                + ("" if c["verdict"] in ("pass", "fail", "error") else f" ({c['verdict']})")]
    if c["kind"] == "error_rate":
        return [c["metric"], _fmt(c["baseline_mean"], ".1%"), _fmt(c["candidate_mean"], ".1%"), "-",
                f"max {c['limit']:.0%}", c["effective"].upper()]
    return [c["metric"], _fmt(c["baseline_mean"], ".0f"), _fmt(c["candidate_mean"], ".0f"),
            "-" if c.get("ratio") is None else f"x{c['ratio']:.2f}", f"max x{c['limit']:g}", c["effective"].upper()]


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode()


def format_gate(result: Mapping[str, Any]) -> str:
    out: List[str] = []
    for rep in result["reports"]:
        s = rep["settings"]
        out += [f"GATE {rep['status'].upper()}   baseline = {rep['baseline']['label']}   candidate = {rep['candidate']['label']}   "
                f"({rep['candidate']['n_examples']} examples; {(1 - s['alpha']):.1%} paired bootstrap intervals; "
                f"inconclusive -> {s['on_inconclusive']})", ""]
        head = ["check", "baseline", "candidate", "delta [CI]", "allowed", "result"]
        body = [_row(c) for c in rep["checks"]]
        widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
        line = lambda cells: "  ".join(x.ljust(w) for x, w in zip(cells, widths))
        out += [line(head), line(["-" * w for w in widths])] + [line(b) for b in body]
        for c in rep["checks"]:
            for r in c["reasons"]:
                out.append(f"  - {c['metric']}: {_ascii(r)}")
        for metric, items in rep["regressions"].items():
            out += ["", f"worst regressions on {metric}:"]
            for it in items:
                out.append(f"  {it['example_id']:<10} {it['baseline']:.2f} -> {it['candidate']:.2f}  {_ascii(it['question'] or '')[:70]}")
                out.append(f"      baseline : {_ascii((it['baseline_output'] or '')[:100]).replace(chr(10), ' ')}")
                out.append(f"      candidate: {_ascii((it['candidate_output'] or '')[:100]).replace(chr(10), ' ')}")
        for w in rep["warnings"]:
            out.append(f"warning: {_ascii(w)}")
        sm = rep["summary"]
        out += ["", f"result: {rep['status'].upper()}  ({sm['fail']} failed, {sm['error']} errors, {sm['warn']} warnings, {sm['pass']} passed)", ""]
    return "\n".join(out).rstrip() + "\n"


def gate_markdown(result: Mapping[str, Any]) -> str:
    """A summary for a pull request or $GITHUB_STEP_SUMMARY."""
    out = [f"### LLM eval gate: {result['status'].upper()}", ""]
    for rep in result["reports"]:
        out += [f"**baseline** `{rep['baseline']['label']}` vs **candidate** `{rep['candidate']['label']}` "
                f"({rep['candidate']['n_examples']} examples) - {rep['status'].upper()}", "",
                "| check | baseline | candidate | delta [CI] | allowed | result |", "|---|---|---|---|---|---|"]
        out += ["| " + " | ".join(_row(c)) + " |" for c in rep["checks"]]
        notes = [f"- **{c['metric']}**: {r}" for c in rep["checks"] for r in c["reasons"]] + [f"- {w}" for w in rep["warnings"]]
        out += [""] + notes + [""] if notes else [""]
        for metric, items in rep["regressions"].items():
            out += [f"<details><summary>worst regressions on {metric}</summary>", ""]
            for it in items:
                out.append(f"- `{it['example_id']}` {it['baseline']:.2f} -> {it['candidate']:.2f}: {it['question']}")
            out += ["", "</details>", ""]
    return "\n".join(out)
