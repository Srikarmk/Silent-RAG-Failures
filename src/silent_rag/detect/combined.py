"""combined_scores (#75): the cheap score detectors combined with weights fit on
NFCorpus results by silent_rag.eval.combine.

Label-free at run time: it fits and scores each part on the same windows the part
would see alone, then applies the saved standardization and logistic weights. The
score is the logit, so higher still means more likely broken. Cost is the sum of
the parts, about 10 ms a window.

The weights are learned from labelled dev windows, so a run that evaluates this
detector on data it was trained on (same dataset and seeds) measures fit, not
detection. check_config refuses weights trained on a held-out dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from silent_rag.detect.base import REGISTRY, Detector, register
from silent_rag.types import Observation

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = "configs/combined_weights.json"
HELD_OUT = {"scifact", "fiqa"}


def load_weights(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        weights = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"can't read combined_scores weights from {p}: {e}") from None
    n = len(weights.get("detectors", []))
    if n == 0 or any(len(weights.get(k, [])) != n for k in ("mean", "scale", "coef")) or "intercept" not in weights:
        raise ValueError(f"{p} isn't a weights file from silent_rag.eval.combine")
    return weights


@register
class CombinedScores(Detector):
    name = "combined_scores"

    def __init__(self, weights: str = DEFAULT_WEIGHTS):
        self.weights_path = weights

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        w = load_weights(params.get("weights", DEFAULT_WEIGHTS))
        unknown = sorted(set(w["detectors"]) - set(REGISTRY))
        if unknown:
            raise ValueError(f"weights use detectors that aren't registered: {unknown}")
        leaked = sorted(set(w.get("trained_on", {}).get("datasets", [])) & HELD_OUT)
        if leaked:
            raise ValueError(f"weights were trained on held-out data {leaked}; train on NFCorpus only")

    def _parts(self):
        if not hasattr(self, "parts_"):
            w = load_weights(self.weights_path)
            self.parts_ = [REGISTRY[d]() for d in w["detectors"]]
            self.mean_, self.scale_ = np.array(w["mean"]), np.array(w["scale"])
            self.coef_, self.intercept_ = np.array(w["coef"]), float(w["intercept"])
        return self.parts_

    def attach(self, retriever) -> None:
        for part in self._parts():
            part.attach(retriever)

    def fit(self, reference: Observation) -> None:
        for part in self._parts():
            part.fit(reference)

    def score(self, window: Observation) -> float:
        x = np.array([part.score(window) for part in self._parts()])
        return float(((x - self.mean_) / self.scale_) @ self.coef_ + self.intercept_)
