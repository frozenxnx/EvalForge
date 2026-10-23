"""Bootstrap statistics. Pure Python, seeded, so reports are reproducible."""
from __future__ import annotations

import math
import random
import zlib
from typing import Any, Dict, List, Optional, Sequence

# Below this many paired examples a result is never flagged significant: with a handful of
# binary outcomes the bootstrap interval can collapse to a point and look falsely certain.
MIN_PAIRS_FOR_SIGNIFICANCE = 10


def seed_for(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode())


def _boot_means(values: Sequence[float], n_boot: int, rng: random.Random) -> List[float]:
    n = len(values)
    means = [sum(rng.choices(values, k=n)) / n for _ in range(n_boot)]
    means.sort()
    return means


def _interval(means: List[float], alpha: float) -> tuple:
    n = len(means)
    return means[int(alpha / 2 * n)], means[min(n - 1, int((1 - alpha / 2) * n))]


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, Any]:
    """Percentile-bootstrap CI for the mean. One sample gets no interval (it would be meaningless)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "ci_low": None, "ci_high": None}
    lo, hi = _interval(_boot_means(list(values), n_boot, random.Random(seed)), alpha)
    return {"n": n, "mean": mean, "ci_low": lo, "ci_high": hi}


def paired_bootstrap(baseline: Sequence[float], candidate: Sequence[float], n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = 0,
                     min_pairs: int = MIN_PAIRS_FOR_SIGNIFICANCE) -> Dict[str, Any]:
    """Paired comparison on the same examples: bootstrap the mean of (candidate - baseline).

    Resampling whole examples keeps the pairing, so example difficulty cancels out; this is why
    paired tests detect real regressions that two unpaired means would drown in noise.
    """
    assert len(baseline) == len(candidate)
    diffs = [c - b for b, c in zip(baseline, candidate)]
    n = len(diffs)
    out: Dict[str, Any] = {"n_paired": n, "delta": None, "ci_low": None, "ci_high": None,
                           "p_value": None, "significant": False, "direction": "flat",
                           "enough_pairs": n >= min_pairs}
    if n == 0:
        return out
    out["delta"] = sum(diffs) / n
    if n == 1:
        return out
    means = _boot_means(diffs, n_boot, random.Random(seed))
    lo, hi = _interval(means, alpha)
    le = sum(1 for m in means if m <= 0)
    ge = sum(1 for m in means if m >= 0)
    out.update(ci_low=lo, ci_high=hi, p_value=min(1.0, 2 * (min(le, ge) + 1) / (n_boot + 1)))
    if out["enough_pairs"] and (lo > 0 or hi < 0):
        out["significant"] = True
        out["direction"] = "up" if out["delta"] > 0 else "down"
    return out


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a proportion. Unlike a bootstrap it stays honest at 0% and 100%
    (20/20 correct is about [0.84, 1.00], not [1.00, 1.00])."""
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mean_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
            seed: int = 0) -> Dict[str, Any]:
    """CI for a metric's mean, choosing the method that is valid for the data.

    Pass/fail metrics (every value 0 or 1) and constant metrics inside [0, 1] get a Wilson interval,
    because a percentile bootstrap collapses to a single point when there is no variation in the
    sample. Everything else uses the bootstrap. `method` says which was used.
    """
    out = bootstrap_ci(values, n_boot, alpha, seed)
    out["method"] = "bootstrap"
    n = len(values)
    if n >= 2:
        in_unit = all(0.0 <= v <= 1.0 for v in values)
        binary = all(v in (0.0, 1.0) for v in values)
        if in_unit and (binary or max(values) == min(values)):
            out["ci_low"], out["ci_high"] = wilson_ci(out["mean"], n, z=1.959964)
            out["method"] = "wilson"
    return out
