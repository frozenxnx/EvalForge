"""A compact description of an experiment for the UI header and navigation."""
from __future__ import annotations

from typing import Any, Dict

from .compare import _rank
from .store import ExperimentStore


def experiment_overview(store: ExperimentStore, experiment_id: str) -> Dict[str, Any]:
    exp = store.get_experiment(experiment_id)
    cfg = exp["config"]
    examples = store.get_examples(experiment_id)
    results = store.get_results(experiment_id)
    scores = store.get_scores(experiment_id)
    hidden = store.hidden_metrics(experiment_id)
    corpus = store.get_corpus_info(experiment_id)
    per_variant: Dict[str, Dict[str, int]] = {}
    for r in results:
        c = per_variant.setdefault(r["variant"], {"ok": 0, "error": 0})
        c["ok" if r["status"] == "ok" else "error"] += 1
    names = sorted({s["metric"] for s in scores}, key=_rank)
    categories = sorted({str(e.metadata["category"]) for e in examples if "category" in e.metadata})
    return {
        "id": exp["id"], "name": exp["name"], "status": exp["status"], "created_at": exp["created_at"],
        "dataset": {"name": exp["dataset_name"], "hash": exp["dataset_hash"], "n_examples": len(examples)},
        "corpus": corpus,
        "variants": [{"label": v["label"], "model": v["model"], "provider": v["provider"],
                      "retriever": (v.get("retriever") or {}).get("type"),
                      "ok": per_variant.get(v["label"], {}).get("ok", 0),
                      "errors": per_variant.get(v["label"], {}).get("error", 0)} for v in cfg["variants"]],
        "metrics": [{"name": n, "hidden": n in hidden} for n in names],
        "has_judge": any(n.startswith("judge:") for n in names),
        "has_retrieval": any(v.get("retriever") for v in cfg["variants"]),
        "n_labels": len({(l["variant"], l["example_id"]) for l in store.get_labels(experiment_id)}),
        "categories": categories, "total_calls": len(results),
    }
