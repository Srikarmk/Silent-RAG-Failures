"""Score-based query performance predictors (#22).

A QPP estimates per query, without labels, how well retrieval went. Each
detector here averages one predictor over the window and scores the window by
how far that average fell below the clean reference window:

    score = reference mean - window mean

Positive means predicted quality went down. Negative means it went up, which is
not evidence of damage, so unlike KS and PSI these are one-sided.

- qpp_max: the top-1 score. The simplest score-based predictor: a query whose
  best match is a weak match probably didn't find what it needed.
- qpp_nqc: Normalized Query Commitment (Shtok et al., "Predicting Query
  Performance by Query-Drift Estimation", ACM TOIS 2012), the std of the top-k
  scores. A spread-out top-k means a few docs clearly beat the rest; a flat one
  means retrieval couldn't tell them apart. The original divides by the
  query's score against the whole corpus, a language-model quantity with no
  equivalent for a dense cosine index, so it is left out here, as is usual when
  NQC is applied to dense retrievers. The paper uses the top 100; `depth: 100`
  does that with `score_depth: 100` in the run config, and the default
  `depth: 10` uses the #18 `std10` feature.
- qpp_gap: top-1 minus top-2. How clearly the best doc beats the runner-up,
  the same idea as the margin of a classifier's top prediction.

What they see: damage that makes retrieval look less confident. What they can't
see: damage that leaves scores alone (wrong docs with the same scores) or that
makes them look more confident, which a model swap can do.
"""

from __future__ import annotations

import numpy as np

from silent_rag.detect.base import register
from silent_rag.detect.features import DEPTH, FeatureDrift
from silent_rag.types import Observation


class QPP(FeatureDrift):
    def compare(self, reference: np.ndarray, current: np.ndarray) -> float:
        return reference.mean() - current.mean()


@register
class MaxScore(QPP):
    name = "qpp_max"
    feature = "top1"


@register
class GapQPP(QPP):
    name = "qpp_gap"
    feature = "gap"


@register
class NQC(QPP):
    name = "qpp_nqc"
    feature = "std10"

    def __init__(self, depth: int = DEPTH):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < DEPTH:
            raise ValueError(f"depth has to be an integer >= {DEPTH}, got {depth!r}")
        self.depth = depth

    def values(self, obs: Observation) -> np.ndarray:
        if self.depth == DEPTH:
            return super().values(obs)
        if obs.deep_scores is None or obs.deep_scores.shape[1] < self.depth:
            have = "none" if obs.deep_scores is None else obs.deep_scores.shape[1]
            raise ValueError(
                f"qpp_nqc with depth {self.depth} needs score_depth >= {self.depth} in the config, got {have}"
            )
        top = np.asarray(obs.deep_scores, dtype=float)[:, : self.depth]
        if not np.isfinite(top).all():
            raise ValueError("deep_scores has non-finite values")
        return top.std(axis=1)
