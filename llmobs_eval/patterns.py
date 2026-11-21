"""Failure patterns: where do the failures concentrate?

A flat list of failing examples hides structure: "all the date-math questions fail" is one finding, not nine.
Examples are grouped two ways, and each group is tested against chance:

  by category  a metadata key of the dataset (default `category`); exact and interpretable
  by cluster   questions that look alike (tf-idf cosine k-means, no model needed; or embeddings with `embed`)

For a group with n examples of which f fail, in a population of N examples with F failures, the p-value is the
exact hypergeometric upper tail P(X >= f): the chance a random group of that size fails at least that much.
It is Bonferroni-corrected over the groups of the same kind that are big enough to test. Clusters are hints
(with a few dozen examples text similarity is rough); categories are the dataset author's own labels.

Two views: "failures" (everything that scores below the threshold) and "regressions" (examples the baseline
passes and the candidate fails: the failures a code change introduced).
"""
from __future__ import annotations

import math
import random
import re
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .compare import _rank, default_metric
from .providers import ConfigError, Provider, build_provider
from .resilience import call_with_retries
from .retrieval import tokenize
from .stats import seed_for
from .store import ExperimentStore

MIN_GROUP = 3                       # groups smaller than this are listed but never tested
MAX_CLUSTERS = 12
EMBED_BATCH = 32

_REFUSAL = re.compile(
    r"\b(i do not know|i don't know|i dont know|do not know|don't know|cannot (?:answer|find|determine|say)|"
    r"can't (?:answer|find|determine|say)|unable to|no information|not (?:stated|mentioned|provided|specified|sure)|"
    r"unknown|documents do not say)\b", re.I)


def is_refusal(text: Optional[str]) -> bool:
    """A cheap detector for answers that decline rather than answer ("I don't know", "not mentioned", ...)."""
    return bool(text) and bool(_REFUSAL.search(text.replace("\u2019", "'")))


def hypergeom_upper_tail(k: int, population: int, successes: int, draws: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(population, successes, draws), computed exactly with integers."""
    if k <= 0:
        return 1.0
    hi = min(draws, successes)
    if k > hi:
        return 0.0
    total = math.comb(population, draws)
    return sum(math.comb(successes, i) * math.comb(population - successes, draws - i) for i in range(k, hi + 1)) / total


# --------------------------------------------------------------------------- vectors and clustering
def _normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    return v if n == 0 else [x / n for x in v]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _term_stats(token_lists: List[List[str]]) -> Tuple[Dict[str, float], Counter]:
    n = len(token_lists)
    df: Counter = Counter(t for toks in token_lists for t in set(toks))
    return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}, df


def tfidf_vectors(token_lists: List[List[str]]) -> List[List[float]]:
    """L2-normalised tf-idf over terms that occur in at least two questions (a word seen once cannot group anything)."""
    idf, df = _term_stats(token_lists)
    vocab = sorted(t for t, c in df.items() if c >= 2)
    index = {t: i for i, t in enumerate(vocab)}
    out = []
    for toks in token_lists:
        v = [0.0] * len(vocab)
        for t, c in Counter(toks).items():
            if t in index:
                v[index[t]] = (1 + math.log(c)) * idf[t]
        out.append(_normalize(v))
    return out


def kmeans(vectors: List[List[float]], k: int, seed: int, iterations: int = 30) -> List[int]:
    """Spherical k-means (cosine) with k-means++ seeding. Zero vectors get cluster -1."""
    live = [i for i, v in enumerate(vectors) if any(v)]
    assign = [-1] * len(vectors)
    if len(live) < 2 or k < 2:
        for i in live:
            assign[i] = 0
        return assign
    k = min(k, len(live))
    rng = random.Random(seed)
    centers = [vectors[live[rng.randrange(len(live))]]]
    while len(centers) < k:
        d2 = [min(1 - _dot(vectors[i], c) for c in centers) ** 2 for i in live]
        total = sum(d2)
        if total <= 1e-12:
            break
        r, acc = rng.random() * total, 0.0
        for i, w in zip(live, d2):
            acc += w
            if acc >= r:
                centers.append(vectors[i])
                break
    current: Dict[int, int] = {}
    for step in range(iterations):
        new = {i: max(range(len(centers)), key=lambda c: _dot(vectors[i], centers[c])) for i in live}
        if new == current:
            break
        current = new
        for c in range(len(centers)):
            members = [vectors[i] for i in live if current[i] == c]
            if members:
                centers[c] = _normalize([sum(col) for col in zip(*members)])
    for i, c in current.items():
        assign[i] = c
    return assign


def default_k(n: int) -> int:
    return max(2, min(MAX_CLUSTERS, round(math.sqrt(n / 2)), n // 3))


def _embed(prov: Provider, model: str, texts: List[str], max_retries: int, sleep: Callable[[float], None]) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        chunk = texts[i:i + EMBED_BATCH]
        out.extend(call_with_retries(lambda: prov.embed(model, chunk), max_retries, sleep=sleep)[0])
    return [_normalize([float(x) for x in v]) for v in out]


# --------------------------------------------------------------------------- the analysis
def failure_patterns(
    store: ExperimentStore, experiment_id: str, variant: str, *, baseline: Optional[str] = None,
    metric: Optional[str] = None, threshold: float = 0.5, group_by: str = "category", cluster: bool = True,
    k: Optional[int] = None, embed: Optional[Mapping[str, Any]] = None,
    extra_providers: Optional[Mapping[str, Mapping[str, Any]]] = None,
    providers: Optional[Mapping[str, Provider]] = None, min_size: int = MIN_GROUP, alpha: float = 0.05,
    top_examples: int = 12, sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    exp = store.get_experiment(experiment_id)
    cfg = exp["config"]
    labels = [v["label"] for v in cfg["variants"]]
    if variant not in labels:
        raise ConfigError(f"unknown variant {variant!r}; this experiment has {labels}")
    if baseline is not None and (baseline not in labels or baseline == variant):
        raise ConfigError(f"baseline must be a different variant of this experiment (have {labels})")
    examples = store.get_examples(experiment_id)
    all_scores = store.get_scores(experiment_id)
    hidden = store.hidden_metrics(experiment_id)
    available = sorted({s["metric"] for s in all_scores if s["variant"] == variant and s["value"] is not None
                        and (s["metric"] not in hidden or s["metric"] == metric)}, key=_rank)
    if not available:
        raise ConfigError(f"variant {variant!r} has no scores: run `evaluate` first")
    if metric is None:
        metric = "contains" if "contains" in available else default_metric(available)
    elif metric not in available:
        raise ConfigError(f"variant {variant!r} has no scores for {metric!r}; available: {available}")

    results = {r["example_id"]: r for r in store.get_results(experiment_id, variant)}
    mine = {s["example_id"]: s["value"] for s in all_scores
            if s["variant"] == variant and s["metric"] == metric and s["value"] is not None}
    base = {s["example_id"]: s["value"] for s in all_scores
            if baseline and s["variant"] == baseline and s["metric"] == metric and s["value"] is not None}
    population, failing, value = [], set(), {}
    for ex in examples:
        ok = ex.id in results and results[ex.id]["status"] == "ok"
        v = mine.get(ex.id)
        if v is None and not ok:
            v = 0.0                                   # a generation that failed outright counts as a failure
        if v is None:
            continue                                  # answered but not scored on this metric
        if baseline is not None and not (base.get(ex.id) is not None and base[ex.id] >= threshold):
            continue                                  # regressions view: only examples the baseline got right
        population.append(ex)
        value[ex.id] = v
        if v < threshold:
            failing.add(ex.id)

    N, F = len(population), len(failing)
    retrievals = store.get_retrievals(experiment_id)
    has_retriever = bool((next(v for v in cfg["variants"] if v["label"] == variant)).get("retriever"))
    by_id = {e.id: e for e in population}

    def missed(ex_id: str) -> bool:
        ex = by_id[ex_id]
        got = {d for d, _ in retrievals.get((variant, ex_id), [])}
        return bool(ex.relevant_doc_ids) and not (got & set(ex.relevant_doc_ids))

    def labeled(ex_id: str) -> bool:
        return bool(by_id[ex_id].relevant_doc_ids)

    def refused(ex_id: str) -> bool:
        return ex_id in results and results[ex_id]["status"] == "ok" and is_refusal(results[ex_id]["output"])

    def make_group(label: str, source: str, ids: List[str], terms: Optional[List[str]] = None) -> Dict[str, Any]:
        n, f = len(ids), sum(1 for i in ids if i in failing)
        overall = F / N if N else 0.0
        fails = [i for i in ids if i in failing]
        return {
            "label": label, "source": source, "terms": terms, "n": n, "failing": f, "rate": f / n if n else 0.0,
            "enrichment": (f / n) / overall if n and overall > 0 else None,
            "p_value": hypergeom_upper_tail(f, N, F, n) if n else 1.0, "p_adjusted": None, "significant": False,
            "refusals": sum(1 for i in fails if refused(i)),
            "no_relevant_retrieved": (sum(1 for i in fails if missed(i)) if has_retriever else None),
            "member_ids": ids[:200],
            "examples": [{"id": i, "question": by_id[i].input[:200], "score": value[i],
                          "output": ((results[i]["output"] or "")[:200] if i in results and results[i]["status"] == "ok"
                                     else "(generation failed)"),
                          "refused": refused(i), "retrieval_missed": (missed(i) if has_retriever and labeled(i) else None)}
                         for i in fails[:top_examples]]}

    def finish(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        eligible = [g for g in groups if g["n"] >= min_size]
        for g in eligible:
            g["p_adjusted"] = min(1.0, g["p_value"] * len(eligible))
            g["significant"] = bool(g["p_adjusted"] < alpha and g["failing"] >= 2
                                    and g["enrichment"] is not None and g["enrichment"] > 1)
        groups.sort(key=lambda g: (not g["significant"], g["p_value"], -g["failing"]))
        return groups

    warnings: List[str] = []
    message: Optional[str] = None
    categories: List[Dict[str, Any]] = []
    clusters: List[Dict[str, Any]] = []
    cluster_mode: Optional[str] = None
    if N == 0:
        message = "no examples to analyse (nothing was scored, or the baseline passed none of them)"
    elif F == 0:
        message = "no failures to explain: every example passes"
    elif F == N:
        message = "every example fails, so there is no pattern to find: look at the failures themselves"
    else:
        buckets: Dict[str, List[str]] = {}
        for ex in population:
            buckets.setdefault(str(ex.metadata.get(group_by, "(none)")), []).append(ex.id)
        if set(buckets) != {"(none)"}:
            categories = finish([make_group(name, "category", ids) for name, ids in buckets.items()])
        else:
            warnings.append(f"the dataset has no metadata key {group_by!r}, so there is no grouping by category")
        if cluster:
            if N < 6:
                warnings.append("too few examples to cluster")
            else:
                texts = [e.input for e in population]
                tokens = [tokenize(t) for t in texts]
                if embed:
                    spec_name = embed["provider"]
                    specs = {**cfg["providers"], **(extra_providers or {})}
                    if spec_name not in specs:
                        raise ConfigError(f"provider {spec_name!r} is not in this experiment's config "
                                          f"(available: {', '.join(specs)}; add more with --extra-providers)")
                    prov = (providers or {}).get(spec_name) or build_provider(spec_name, specs[spec_name])
                    vectors = _embed(prov, embed["model"], texts, cfg["max_retries"], sleep)
                    cluster_mode = "embedding"
                else:
                    vectors = tfidf_vectors(tokens)
                    cluster_mode = "tfidf"
                assign = kmeans(vectors, k or default_k(N), seed_for(experiment_id, variant, metric, str(N)))
                idf, df = _term_stats(tokens)
                groups = []
                for c in sorted({a for a in assign if a >= 0}):
                    idx = [i for i, a in enumerate(assign) if a == c]
                    weight: Counter = Counter()
                    for i in idx:
                        for t in set(tokens[i]):
                            if df[t] >= 2:
                                weight[t] += idf[t]
                    terms = [t for t, _ in weight.most_common(3)]
                    groups.append(make_group(", ".join(terms) or f"cluster {c + 1}", "cluster",
                                             [population[i].id for i in idx], terms))
                clusters = finish(groups)
                if not clusters:
                    warnings.append("no usable clusters: the questions share too little vocabulary")
                warnings.append("clusters are hints: with a few dozen examples text similarity is rough, "
                                "and a cluster is only a pattern if its p-value says so")
    return {
        "experiment": {"id": exp["id"], "name": exp["name"]}, "variant": variant, "baseline": baseline, "metric": metric,
        "view": "regressions" if baseline else "failures", "threshold": threshold, "population": N, "failures": F,
        "failure_rate": F / N if N else None, "refusals": sum(1 for i in failing if refused(i)),
        "no_relevant_retrieved": (sum(1 for i in failing if missed(i)) if has_retriever else None),
        "group_by": group_by, "categories": categories, "clusters": clusters, "cluster_mode": cluster_mode,
        "available_metrics": available, "warnings": warnings, "message": message}


# --------------------------------------------------------------------------- terminal rendering (ASCII)
def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode()


def _p(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def format_patterns(p: Mapping[str, Any]) -> str:
    what = (f"new failures: the baseline {p['baseline']!r} passes, {p['variant']!r} fails" if p["baseline"]
            else "all failures")
    out = [f"FAILURE PATTERNS: {p['variant']} on {p['metric']}   (fails when the score is below {p['threshold']:g}; {what})", ""]
    if p["population"]:
        line = f"{p['population']} examples, {p['failures']} fail ({p['failure_rate']:.0%})."
        if p["failures"]:
            if p["refusals"]:
                line += f" {p['refusals']} of the failing answers are refusals (\"I don't know\" and the like)."
            if p["no_relevant_retrieved"] is not None:
                line += f" In {p['no_relevant_retrieved']} the retriever found none of the labeled relevant documents."
        out.append(line)
    if p["message"]:
        out.append(p["message"])
    for title, groups in ((f"by category (metadata key {p['group_by']!r})", p["categories"]),
                          ("by similar questions (%s clusters)" % (p["cluster_mode"] or "tf-idf"), p["clusters"])):
        if not groups:
            continue
        out += ["", title + ":"]
        show_ref = any(g["refusals"] for g in groups)
        head = (["group", "n", "fail", "rate", "vs all", "p (adj)"] + (["refusals"] if show_ref else [])
                + (["retr. missed"] if p["no_relevant_retrieved"] is not None else []))
        body = [[_ascii(g["label"])[:36], str(g["n"]), str(g["failing"]), f"{g['rate']:.0%}",
                 "-" if g["enrichment"] is None else f"x{g['enrichment']:.1f}",
                 ("-" if g["p_adjusted"] is None else _p(g["p_adjusted"])) + (" *" if g["significant"] else "")]
                + ([str(g["refusals"])] if show_ref else [])
                + ([str(g["no_relevant_retrieved"])] if p["no_relevant_retrieved"] is not None else [])
                for g in groups]
        widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
        fmt = lambda cells: "  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths))
        out += [fmt(head), fmt(["-" * w for w in widths])] + [fmt(b) for b in body]
        for g in groups:
            if g["significant"]:
                out += ["", f"  * {_ascii(g['label'])}: {g['failing']} of {g['n']} fail (chance: p {'< 0.001' if g['p_adjusted'] < 0.001 else '= %.3f' % g['p_adjusted']} after correction)"]
                for ex in g["examples"][:5]:
                    out.append(f"      {ex['id']:<10} score {ex['score']:.2f}  {_ascii(ex['question'])[:70]}")
                    out.append(f"                 answer: {_ascii(ex['output'])[:80]}".replace("\n", " "))
    if p["categories"] or p["clusters"]:
        out += ["", "* = more failures than chance explains (exact hypergeometric test, Bonferroni-corrected over the groups of that kind)."]
    out += [""] + [f"note: {w}" for w in p["warnings"]]
    return "\n".join(out).rstrip() + "\n"
