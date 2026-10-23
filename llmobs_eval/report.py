"""Per-variant summary: errors, latency percentiles, tokens, cost. (Quality metrics come in step 3.)"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .store import ExperimentStore


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Nearest-rank percentile; None for an empty sequence."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def summarize(store: ExperimentStore, experiment_id: str) -> List[Dict[str, Any]]:
    cfg = store.get_experiment(experiment_id)["config"]
    results = store.get_results(experiment_id)
    rows = []
    for v in cfg["variants"]:
        mine = [r for r in results if r["variant"] == v["label"]]
        ok = [r for r in mine if r["status"] == "ok"]
        lat = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
        costs = [r["cost_usd"] for r in ok if r["cost_usd"] is not None]
        rows.append({
            "variant": v["label"], "model": v["model"], "n": len(mine), "ok": len(ok),
            "errors": len(mine) - len(ok),
            "error_rate": (len(mine) - len(ok)) / len(mine) if mine else 0.0,
            "retried": sum(1 for r in mine if r["attempts"] > 1),
            "p50_ms": percentile(lat, 50), "p95_ms": percentile(lat, 95),
            "input_tokens": sum(r["input_tokens"] or 0 for r in ok),
            "output_tokens": sum(r["output_tokens"] or 0 for r in ok),
            "cost_usd": sum(costs) if costs else None,
        })
    return rows


def format_table(rows: List[Dict[str, Any]]) -> str:
    def f(v, spec=".0f", dash="-"):
        return dash if v is None else format(v, spec)
    head = ["variant", "n", "err", "retried", "p50 ms", "p95 ms", "in tok", "out tok", "cost USD"]
    body = [[r["variant"], str(r["n"]), f"{r['errors']} ({r['error_rate']:.0%})", str(r["retried"]),
             f(r["p50_ms"]), f(r["p95_ms"]), str(r["input_tokens"]), str(r["output_tokens"]),
             f(r["cost_usd"], ".4f")] for r in rows]
    widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))
    return "\n".join([line(head), line(["-" * w for w in widths])] + [line(b) for b in body])
