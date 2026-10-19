"""How many queries a monitoring window needs before the paired bootstrap can
tell a real nDCG@10 drop from noise (#4).

There are no real failures yet, so damage is simulated on the clean per-query
scores. Two knobs:

- knockout: each query drops to 0 with probability drop / mean, so the expected
  window drop is `drop`. With ~1 relevant doc per query this is what losing the
  relevant doc looks like.
- churn: a fraction of queries get another query's clean score instead of their
  own. That moves single queries up and down without changing the mean, which
  real failures do and pure knockout doesn't. With churn = 0 every per-query
  difference is >= 0 and the bootstrap looks better than it will in practice.
"""

from __future__ import annotations

import numpy as np

from silent_rag.eval.ground_truth import paired_bootstrap


def simulate_damage(clean: np.ndarray, drop: float, rng: np.random.Generator, churn: float = 0.0) -> np.ndarray:
    clean = np.asarray(clean, dtype=float)
    damaged = clean.copy()
    if churn > 0:
        moved = rng.random(len(clean)) < churn
        damaged[moved] = clean[rng.integers(0, len(clean), size=moved.sum())]
    mean = clean.mean()
    if drop < 0:
        raise ValueError("drop must be >= 0")
    if drop > 0 and mean > 0:
        # a window whose mean is below `drop` just loses everything it has
        damaged[rng.random(len(clean)) < min(1.0, drop / mean)] = 0.0
    return damaged


def detection_rate(
    pool: np.ndarray,
    window_size: int,
    drop: float,
    trials: int = 200,
    alpha: float = 0.05,
    churn: float = 0.0,
    seed: int = 0,
) -> float:
    """Fraction of sampled windows where the paired bootstrap calls the drop
    significant. At drop = 0 this is the false-alarm rate."""
    pool = np.asarray(pool, dtype=float)
    if window_size > len(pool):
        raise ValueError(f"window of {window_size} doesn't fit in a pool of {len(pool)} queries")
    rng = np.random.default_rng(seed)
    flagged = 0
    for trial in range(trials):
        clean = pool[rng.choice(len(pool), window_size, replace=False)]
        damaged = simulate_damage(clean, drop, rng, churn)
        _, p = paired_bootstrap(clean, damaged, seed=trial)
        flagged += p < alpha
    return flagged / trials


def min_detectable_drop(rates: dict[float, float], power: float = 0.8) -> float:
    """Smallest drop on the grid that gets flagged in at least `power` of the
    windows. NaN if none of them do."""
    hits = [d for d, r in sorted(rates.items()) if d > 0 and r >= power]
    return hits[0] if hits else float("nan")
