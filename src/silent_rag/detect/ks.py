"""KS detector on the top-1 retrieval score (#8).

Fits the distribution of each query's best score on the clean reference window,
then scores a window by the two-sample KS statistic against it. 0 means the two
windows' top-1 scores look the same, 1 means they don't overlap at all.

The score is the statistic, not the p-value: higher has to mean "more likely
broken", and with windows of 100 the p-value is near 0 for any visible shift,
so it stops ranking windows against each other.

What it sees: anything that shifts how confident retrieval looks, e.g. a mix of
doc vectors from two models (F1). What it can't see: damage that changes which
docs come back without changing their scores. It also fires on query drift that
is harmless, since it measures "scores look different", not harm.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from silent_rag.detect.base import Detector, register
from silent_rag.detect.features import score_features
from silent_rag.types import Observation


@register
class KSTop1(Detector):
    name = "ks_top1"
    feature = "top1"

    def fit(self, reference: Observation) -> None:
        self.reference_ = score_features(reference)[self.feature].to_numpy()

    def score(self, window: Observation) -> float:
        if not hasattr(self, "reference_"):
            raise RuntimeError(f"{self.name} needs fit(reference) before it can score")
        current = score_features(window)[self.feature].to_numpy()
        # only the statistic is used, so skip the exact p-value (it warns on ties)
        return float(ks_2samp(self.reference_, current, method="asymp").statistic)
