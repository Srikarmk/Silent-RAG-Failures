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

from silent_rag.detect.features import FeatureDrift, register_per_feature


class KS(FeatureDrift):
    def compare(self, reference: np.ndarray, current: np.ndarray) -> float:
        # only the statistic is used, so skip the exact p-value (it warns on ties)
        return ks_2samp(reference, current, method="asymp").statistic


KS_DETECTORS = register_per_feature(KS, "ks")
KSTop1 = KS_DETECTORS["top1"]
