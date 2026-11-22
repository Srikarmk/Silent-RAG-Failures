"""MMD between the reference window and a scored window, on embedding vectors.

Two views, one row per question in both:
- mmd_query: the question vectors. Cannot respond to F1, which replaces stored
  document vectors and leaves the questions unchanged.
- mmd_retrieved_docs: per question, the mean of its retrieved documents' stored index
  vectors. Questions stay the independent unit: a question's documents are correlated.

Kernel: RBF exp(-|x-y|^2 / (2 sigma^2)), sigma = median pairwise distance of the
pooled sample (median heuristic). Vectors are not re-normalised. The score is the
unbiased MMD^2 (kernel diagonals left out), so it can be slightly negative when the
two windows match. The *_perm variants score 1 - permutation p-value.
"""

from __future__ import annotations

import numpy as np

from silent_rag.detect.base import register
from silent_rag.detect.domain_classifier import _doc_rows, _query_rows
from silent_rag.detect.features import FeatureDrift, register_permutation
from silent_rag.types import Observation


def mmd2(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased MMD^2 between two samples, rows = observations.

    RBF kernel with sigma = median pairwise distance of the pooled sample (if every
    distance is 0, the kernel is all ones and MMD^2 is 0). Near 0 when x and y come
    from the same distribution, and can be slightly negative. Needs >= 2 rows each.
    """
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        raise ValueError(f"MMD needs at least 2 rows per sample, got {n} and {m}")
    z = np.vstack([x, y]).astype(float)
    sq = (z * z).sum(1)
    d = np.maximum(sq[:, None] + sq[None, :] - 2.0 * z @ z.T, 0.0)
    sigma2 = np.median(d[np.triu_indices(n + m, k=1)])
    k = np.exp(-d / (2.0 * sigma2)) if sigma2 > 0 else np.ones_like(d)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    xx = (kxx.sum() - n) / (n * (n - 1))
    yy = (kyy.sum() - m) / (m * (m - 1))
    return float(xx + yy - 2.0 * kxy.mean())


class MMD(FeatureDrift):
    """MMD^2 between the reference window and a window, one row per question.
    `view` picks the vectors: "query" or "retrieved_docs"."""

    feature = "embeddings"
    view = "query"

    def values(self, obs: Observation) -> np.ndarray:
        """(n_questions, dim): the question vectors, or per question the mean of its
        retrieved documents' stored index vectors."""
        if self.view == "query":
            return _query_rows(obs)[0]
        doc_vectors, doc_ids = _doc_rows(obs)
        lookup = dict(zip(doc_ids, doc_vectors))
        if any(len(hits) == 0 for hits in obs.topk_ids):
            raise ValueError("a question in this window retrieved no documents")
        return np.stack([np.mean([lookup[d] for d in hits], axis=0) for hits in obs.topk_ids])

    def compare(self, reference: np.ndarray, current: np.ndarray) -> float:
        return mmd2(reference, current)


@register
class MMDQuery(MMD):
    """MMD^2 on question vectors. Blind to F1, which leaves questions unchanged."""

    name = "mmd_query"
    view = "query"


@register
class MMDRetrievedDocs(MMD):
    """MMD^2 on the per-question mean of the retrieved documents' stored vectors."""

    name = "mmd_retrieved_docs"
    view = "retrieved_docs"


MMDQueryPerm = register_permutation(MMDQuery)
MMDRetrievedDocsPerm = register_permutation(MMDRetrievedDocs)
