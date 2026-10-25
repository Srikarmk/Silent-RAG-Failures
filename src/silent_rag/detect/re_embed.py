"""Re-embedding check (#25): re-embed a sample of indexed docs with the model the
system is supposed to be running and compare to the stored vectors.

What it sees: stored vectors that the expected encoder and settings would not
produce, i.e. docs embedded by another model or config (F1, a partial or stale
re-index). What it can't see, by construction:
- damage where every stored vector is correct for its text: stale docs that were
  embedded properly (F2), a re-chunk embedded with the same model (F3)
- anything on the query side (F4, F5 query prefix / normalization)
It also fires on harmless re-indexes with a different model, since it measures
"not what we expect", not harm. Negative controls that touch the doc model will
false-alarm.

metric="cosine" is direction only. metric="l2" (relative L2 error) also sees
norm changes, e.g. doc vectors stored without normalization, which break
inner-product ranking while leaving cosine at ~0.

Per-row distances below `floor` count as 0. Re-encoding the same text in a
different batch moves vectors by ~1e-7 in cosine (~5e-4 in relative L2), so
without a floor a clean index scores float noise and any threshold calibrated on
clean windows sits inside that noise. With it, a clean index scores exactly 0;
use a threshold above 0.

Costs one encode of `sample` docs per check, so sample size is the cost/accuracy knob.
"""

from __future__ import annotations

import zlib

import numpy as np

from silent_rag.detect.base import Detector, register
from silent_rag.types import Observation


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return 1.0 - (a * b).sum(axis=1)


def relative_l2(stored: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    return np.linalg.norm(stored - fresh, axis=1) / np.maximum(np.linalg.norm(fresh, axis=1), 1e-12)


METRICS = {"cosine": cosine_distance, "l2": relative_l2}
# well above batch/device float noise, well below any real model or config change (~0.1+)
FLOORS = {"cosine": 1e-5, "l2": 5e-3}


@register
class ReEmbed(Detector):
    name = "re_embed"

    def __init__(self, sample: int = 200, metric: str = "cosine", seed: int = 0, floor: float | None = None):
        if isinstance(sample, bool) or not isinstance(sample, int) or sample < 1:
            raise ValueError(f"sample has to be a positive integer, got {sample!r}")
        if metric not in METRICS:
            raise ValueError(f"metric has to be one of {sorted(METRICS)}, got {metric!r}")
        floor = FLOORS[metric] if floor is None else floor
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not 0 <= floor < 1:
            raise ValueError(f"floor has to be a number in [0, 1), got {floor!r}")
        self.sample = sample
        self.metric = metric
        self.floor = float(floor)
        self.seed = seed
        self.retriever = None

    def attach(self, retriever) -> None:
        self.retriever = retriever

    def rows(self, window: Observation) -> np.ndarray:
        """Pick rows by a hash of (window, doc id) rather than by position, so the same
        docs are sampled under every condition even when a failure adds rows (F2)."""
        key = zlib.crc32("\n".join(window.query_ids).encode()) ^ self.seed
        ranks = np.fromiter((zlib.crc32(d.encode(), key) for d in window.doc_ids), dtype=np.uint32,
                            count=len(window.doc_ids))
        return np.argsort(ranks, kind="stable")[: self.sample]

    def distances(self, window: Observation) -> np.ndarray:
        if self.retriever is None:
            raise RuntimeError("re_embed needs attach(retriever) before it can score")
        if window.doc_texts is None or window.doc_embeddings is None or window.doc_ids is None:
            raise ValueError("re_embed needs doc_ids, doc_texts and doc_embeddings in the Observation")
        n = len(window.doc_embeddings)
        if len(window.doc_texts) != n or len(window.doc_ids) != n:
            raise ValueError(
                f"index has {n} rows but {len(window.doc_texts)} doc_texts and {len(window.doc_ids)} doc_ids, "
                "they have to be row-aligned"
            )
        rows = self.rows(window)
        fresh = self.retriever.encode([window.doc_texts[i] for i in rows])
        d = METRICS[self.metric](np.asarray(window.doc_embeddings[rows], dtype=np.float32), fresh)
        return np.where(d < self.floor, 0.0, d)

    def fit(self, reference: Observation) -> None:
        # nothing to learn, a clean index scores ~0. Kept so we can report the noise floor.
        self.reference_score_ = float(self.distances(reference).mean())

    def score(self, window: Observation) -> float:
        return float(self.distances(window).mean())
