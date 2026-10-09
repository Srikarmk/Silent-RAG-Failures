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
