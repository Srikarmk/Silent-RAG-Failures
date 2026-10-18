"""Re-embedding check (#25): re-embed a sample of indexed docs with the model the
system is supposed to be running and compare to the stored vectors. If they
disagree, the index is out of sync with the model (F1, F3).

Score = mean cosine distance between stored and fresh vectors. Costs one
encode of `sample` docs per check, so sample size is the cost/accuracy knob.
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


@register
class ReEmbed(Detector):
    name = "re_embed"

    def __init__(self, sample: int = 200, seed: int = 0):
        self.sample = sample
        self.seed = seed
        self.retriever = None

    def attach(self, retriever) -> None:
        self.retriever = retriever

    def distances(self, window: Observation) -> np.ndarray:
        if self.retriever is None:
            raise RuntimeError("re_embed needs attach(retriever) before it can score")
        if window.doc_texts is None or window.doc_embeddings is None:
            raise ValueError("re_embed needs doc_texts and doc_embeddings in the Observation")
        n = len(window.doc_texts)
        # same window -> same sample, so repeated checks are comparable
        key = zlib.crc32("\n".join(window.query_ids).encode())
        rows = np.random.default_rng([self.seed, key]).choice(n, min(self.sample, n), replace=False)
        fresh = self.retriever.encode([window.doc_texts[i] for i in rows])
        return cosine_distance(np.asarray(window.doc_embeddings[rows], dtype=np.float32), fresh)

    def fit(self, reference: Observation) -> None:
        # nothing to learn, a clean index scores ~0. Kept so we can report the noise floor.
        self.reference_score_ = float(self.distances(reference).mean())

    def score(self, window: Observation) -> float:
        return float(self.distances(window).mean())
