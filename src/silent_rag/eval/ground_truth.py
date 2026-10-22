"""Ground truth: how much damage did a condition actually do. This is the only
place qrels get used."""

from __future__ import annotations

import numpy as np


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = sum(rels.get(d, 0) / np.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def per_query_ndcg(topk_ids, query_ids, qrels, k: int = 10) -> np.ndarray:
    return np.array([ndcg_at_k(ranked, qrels[q], k) for ranked, q in zip(topk_ids, query_ids)])


def bootstrap_ci(scores, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Mean of per-query scores and a percentile bootstrap CI.

    Resamples queries with replacement n_boot times and takes the alpha/2 and
    1 - alpha/2 quantiles of the resampled means. Returns (mean, lo, hi).
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores must be a non-empty 1-d array")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if int(n_boot) != n_boot or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scores), size=(int(n_boot), len(scores)))
    boots = scores[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(scores.mean()), float(lo), float(hi)


def paired_bootstrap(clean: np.ndarray, damaged: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Mean drop (clean - damaged) and a one-sided p-value for "no drop".

    Both arrays are per-query scores over the same queries in the same order.
    """
    rng = np.random.default_rng(seed)
    diff = clean - damaged
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    p = float((boots <= 0).mean())
    return float(diff.mean()), p
