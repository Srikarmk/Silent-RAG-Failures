"""Per-query score features shared by the score-based detectors (#18).

PSI, KS and the score-based QPP detectors all work off the same few numbers per
query. Computing them here once means every detector sees exactly the same
values, so differences between detectors are differences in the test, not in
how the features were computed.

Features, from each query's retrieved scores (sorted high to low):
- top1: best score
- gap: top-1 minus top-2
- mean10: mean of the top 10
- std10: std of the top 10 (population, ddof=0)

Only the first 10 columns are used even if topk_scores is deeper, so the
features don't change if we start logging more scores later.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import pandas as pd

from silent_rag.detect.base import Detector, register
from silent_rag.types import Observation

DEPTH = 10
FEATURES = ["top1", "gap", "mean10", "std10"]


def score_features(obs: Observation) -> pd.DataFrame:
    """One row per query, indexed by query id, columns FEATURES."""
    scores = np.asarray(obs.topk_scores, dtype=float)
    if scores.ndim != 2 or len(scores) != len(obs.query_ids):
        raise ValueError(
            f"topk_scores has shape {scores.shape}, expected ({len(obs.query_ids)}, k) for "
            f"{len(obs.query_ids)} queries"
        )
    if scores.shape[1] < DEPTH:
        raise ValueError(f"score features need the top {DEPTH} scores, topk_scores only has {scores.shape[1]}")
    top = scores[:, :DEPTH]
    if not np.isfinite(top).all():
        raise ValueError("topk_scores has non-finite values in the top 10")
    if (np.diff(top, axis=1) > 1e-6).any():
        raise ValueError("topk_scores has to be sorted high to low per query")
    return pd.DataFrame(
        {
            "top1": top[:, 0],
            "gap": top[:, 0] - top[:, 1],
            "mean10": top.mean(axis=1),
            "std10": top.std(axis=1),
        },
        index=pd.Index(list(obs.query_ids), name="query_id"),
    )[FEATURES]


class FeatureDrift(Detector):
    """Compares one score feature's distribution in a window to the reference window.

    Subclasses set `compare(reference_values, window_values)`, and
    `register_per_feature` turns them into one registered detector per feature.
    """

    feature: str

    def values(self, obs: Observation) -> np.ndarray:
        """One value per query. Overridden by detectors that need more than the #18 features."""
        return score_features(obs)[self.feature].to_numpy()

    def fit(self, reference: Observation) -> None:
        self.reference_ = self.values(reference)

    def score(self, window: Observation) -> float:
        if not hasattr(self, "reference_"):
            raise RuntimeError(f"{self.name} needs fit(reference) before it can score")
        return float(self.compare(self.reference_, self.values(window)))

    @abstractmethod
    def compare(self, reference: np.ndarray, current: np.ndarray) -> float: ...

    def statistic(self, reference: np.ndarray, current: np.ndarray) -> float:
        """The detector's statistic as a function of the two samples and nothing else.

        The permutation test calls this on reshuffled halves, so it must not use
        anything fit() stored. On the fitted reference it has to equal compare().
        Detectors whose compare() already depends only on its arguments inherit this.
        """
        return self.compare(reference, current)


def permutation_p_value(statistic, reference: np.ndarray, current: np.ndarray,
                        permutations: int, rng: np.random.Generator) -> float:
    """Upper-tail permutation p-value of statistic(reference, current) (#73).

    Pools the two samples, reshuffles, splits back into the original sizes and
    recomputes. p = (1 + number of reshuffles at least as large as the observed
    statistic) / (permutations + 1), so it is never 0 and its smallest value is
    1 / (permutations + 1).

    The null is that the reference and window values are exchangeable: one pool,
    split at random. A small p says the window's values differ from the
    reference's. It says nothing about whether retrieval got worse.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    observed = float(statistic(reference, current))
    pooled = np.concatenate([reference, current])
    n = len(reference)
    # a reshuffle that reproduces the observed split can differ from it by float noise
    floor = observed - 1e-12 * max(1.0, abs(observed))
    hits = 0
    for _ in range(permutations):
        shuffled = rng.permutation(pooled)
        hits += float(statistic(shuffled[:n], shuffled[n:])) >= floor
    return (1 + hits) / (permutations + 1)


class Permutation:
    """Mixin that turns a FeatureDrift detector into its permutation test (#73).

    score() is 1 - p, so higher still means "more likely something changed" and an
    alarm at level alpha is score > 1 - alpha. `permutations` sets the resolution of
    p; the default 999 gives steps of 0.001.

    Use it for thresholding, not for ranking: 1 - p saturates at
    1 - 1 / (permutations + 1) for any clear shift, so AUROC should keep coming from
    the plain statistic. The same reshuffles are used for every window (fixed `seed`),
    so results are reproducible.
    """

    def __init__(self, permutations: int = 999, seed: int = 0, **kwargs):
        if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
            raise ValueError(f"permutations has to be a positive integer, got {permutations!r}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed has to be a non-negative integer, got {seed!r}")
        super().__init__(**kwargs)
        self.permutations = permutations
        self.seed = seed

    def p_value(self, window: Observation) -> float:
        if not hasattr(self, "reference_"):
            raise RuntimeError(f"{self.name} needs fit(reference) before it can score")
        rng = np.random.default_rng(self.seed)
        return permutation_p_value(self.statistic, self.reference_, self.values(window), self.permutations, rng)

    def score(self, window: Observation) -> float:
        return 1.0 - self.p_value(window)


def register_permutation(base: type[FeatureDrift]) -> type[FeatureDrift]:
    """Registers `<base.name>_perm`: the same feature and statistic, scored by 1 - p."""
    attrs = {"name": f"{base.name}_perm", "__module__": base.__module__, "__doc__": Permutation.__doc__}
    return register(type(f"{base.__name__}_perm", (Permutation, base), attrs))


def register_per_feature(base: type[FeatureDrift], prefix: str) -> dict[str, type[FeatureDrift]]:
    """One registered subclass of `base` per feature, named `<prefix>_<feature>`."""
    made = {}
    for feature in FEATURES:
        attrs = {"name": f"{prefix}_{feature}", "feature": feature, "__module__": base.__module__}
        made[feature] = register(type(f"{base.__name__}_{feature}", (base,), attrs))
    return made
