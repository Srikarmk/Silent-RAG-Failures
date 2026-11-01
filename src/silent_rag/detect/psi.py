"""PSI detectors on the score features (#19).

One detector per feature from #18: psi_top1, psi_gap, psi_mean10, psi_std10.

Standard PSI, so the 0.25 rule of thumb can be tested as is in #31:

    PSI = sum over bins of (window% - reference%) * ln(window% / reference%)

Bins are quantiles of the reference window (deciles by default), fit once in
fit() and reused for every window. Quantile rather than fixed-width bins:
- with ~100 reference queries, fixed-width bins leave the tails nearly empty,
  and PSI there is driven by the empty-bin floor instead of the data
- the features live on very different scales (top1 ~0.6, gap ~0.01), and
  quantiles need no per-feature bin width
- deciles of the reference are what PSI is usually computed on, which is what
  the 0.25 rule of thumb assumes
The outer bins are open-ended, so window values outside the reference range
land in the first or last bin instead of being dropped. Tied quantiles are
merged, so a feature with many identical values gets fewer bins. Proportions
are floored at `eps` so an empty bin doesn't give log(0).

PSI is biased upward with small samples. With 100 reference and 100 window
queries and 10 bins, PSI between two samples of the same distribution averages
~0.19 and is above 0.25 in ~22% of windows. That doesn't affect AUROC, but it
does mean 0.25 is close to the noise level at our window size (#31).
"""

from __future__ import annotations

import numpy as np

from silent_rag.detect.features import FeatureDrift, register_per_feature
from silent_rag.types import Observation


class PSI(FeatureDrift):
    def __init__(self, bins: int = 10, eps: float = 1e-4):
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError(f"bins has to be an integer >= 2, got {bins!r}")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not 0 < eps < 0.1:
            raise ValueError(f"eps has to be a number in (0, 0.1), got {eps!r}")
        self.bins = bins
        self.eps = float(eps)

    def fit(self, reference: Observation) -> None:
        super().fit(reference)
        self.edges_ = np.unique(np.quantile(self.reference_, np.linspace(0, 1, self.bins + 1)[1:-1]))
        self.expected_ = self.proportions(self.reference_)

    def proportions(self, values: np.ndarray) -> np.ndarray:
        counts = np.bincount(np.searchsorted(self.edges_, values, side="right"), minlength=len(self.edges_) + 1)
        return np.maximum(counts / len(values), self.eps)

    def compare(self, reference: np.ndarray, current: np.ndarray) -> float:
        actual = self.proportions(current)
        return float(((actual - self.expected_) * np.log(actual / self.expected_)).sum())


PSI_DETECTORS = register_per_feature(PSI, "psi")
