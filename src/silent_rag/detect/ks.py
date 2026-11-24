"""KS detectors on the score features (#8, #19).

One detector per feature from #18: ks_top1, ks_gap, ks_mean10, ks_std10. Each
fits the feature's distribution on the clean reference window, then scores a
window by the two-sample KS statistic against it. 0 means the two windows look
the same, 1 means they don't overlap at all.

The score is the statistic, not the p-value: higher has to mean "more likely
broken", and with windows of 100 the p-value is near 0 for any visible shift,
so it stops ranking windows against each other.

What they see: anything that shifts how confident retrieval looks, e.g. a mix
of doc vectors from two models (F1). What they can't see: damage that changes
which docs come back without changing their scores. They also fire on query
drift that is harmless, since they measure "scores look different", not harm.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from silent_rag.detect.features import FeatureDrift, register_per_feature, register_permutation


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """The two-sample KS statistic: the largest gap between the two empirical CDFs.
    Same value as scipy's, without its p-value, which the permutation test (#73)
    would otherwise pay for on every reshuffle."""
    reference, current = np.sort(reference), np.sort(current)
    points = np.concatenate([reference, current])
    cdf_reference = np.searchsorted(reference, points, side="right") / len(reference)
    cdf_current = np.searchsorted(current, points, side="right") / len(current)
    return float(np.abs(cdf_reference - cdf_current).max())


class KS(FeatureDrift):
    def compare(self, reference: np.ndarray, current: np.ndarray) -> float:
        # only the statistic is used, so skip the exact p-value (it warns on ties)
        return ks_2samp(reference, current, method="asymp").statistic

    def statistic(self, reference: np.ndarray, current: np.ndarray) -> float:
        return ks_statistic(reference, current)


KS_DETECTORS = register_per_feature(KS, "ks")
KSTop1 = KS_DETECTORS["top1"]
# ks_top1_perm, ...: the same statistic with a per-window permutation p-value (#73)
KS_PERMUTATION = {feature: register_permutation(cls) for feature, cls in KS_DETECTORS.items()}
