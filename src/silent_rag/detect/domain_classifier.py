"""Label-free domain classifiers for query and retrieved-document drift (#21).

The detector score is mean held-out AUROC. A score near 0.5 means the two
windows are hard to distinguish; a high score means they differ.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from silent_rag.detect.base import Detector, register
from silent_rag.types import Observation


def _query_rows(observation: Observation) -> tuple[np.ndarray, list[str]]:
    vectors = np.asarray(observation.query_embeddings, dtype=float)
    return vectors, list(observation.query_ids)


def _doc_rows(observation: Observation) -> tuple[np.ndarray, list[str]]:
    """One vector per retrieved parent document, never one per query hit."""
    if observation.doc_embeddings is None or observation.doc_ids is None:
        raise ValueError("retrieved-doc classifier requires doc_embeddings and doc_ids")

    vectors = np.asarray(observation.doc_embeddings, dtype=float)
    ids = list(observation.doc_ids)
    if vectors.ndim != 2 or len(vectors) != len(ids):
        raise ValueError("doc_ids and doc_embeddings must align")

    parents = getattr(observation, "parent_ids", None)
    if parents is None:
        parents = ids
    elif len(parents) != len(ids):
        raise ValueError("parent_ids must align with doc_embeddings")

    # Chunked documents use the mean of their chunk embeddings as a stable
    # document-level representation. This is NOT the best-scoring chunk.
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for parent, vector in zip(parents, vectors):
        if parent not in sums:
            sums[parent] = np.array(vector, copy=True)
            counts[parent] = 1
        else:
            sums[parent] += vector
            counts[parent] += 1

    selected = list(dict.fromkeys(doc for row in observation.topk_ids for doc in row))
    unknown = [doc for doc in selected if doc not in sums]
    if unknown:
        raise ValueError(
            "retrieved doc ids do not map to index entries; "
            "chunked results need Observation.parent_ids"
        )
    if not selected:
        raise ValueError("no retrieved documents in this window")
    return np.stack([sums[doc] / counts[doc] for doc in selected]), selected


def _validate(vectors: np.ndarray, ids: list[str], label: str) -> None:
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] == 0:
        raise ValueError(f"{label}: embeddings must be a 2-D array aligned with ids")
    if len(vectors) == 0 or not np.isfinite(vectors).all():
        raise ValueError(f"{label}: embeddings must be nonempty and finite")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label}: duplicate identities within the same window")


def _cv_auc(
    reference: np.ndarray,
    current: np.ndarray,
    reference_ids: list[str],
    current_ids: list[str],
    folds: int,
) -> float:
    _validate(reference, reference_ids, "reference")
    _validate(current, current_ids, "current")
    if reference.shape[1] != current.shape[1]:
        raise ValueError("reference and current embedding dimensions differ")

    n_splits = min(folds, len(reference_ids), len(current_ids))
    if n_splits < 2:
        raise ValueError("cross-validation requires at least two samples per domain")

    x = np.vstack([reference, current])
    y = np.r_[np.zeros(len(reference), dtype=int), np.ones(len(current), dtype=int)]
    groups = np.asarray(reference_ids + current_ids, dtype=str)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs = []
    for train_idx, test_idx in splitter.split(x, y, groups=groups):
        if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
            raise ValueError("not enough independent identities in each CV fold")
        # Scaling stays inside each fold: no preprocessing leakage.
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
        )
        model.fit(x[train_idx], y[train_idx])
        probabilities = model.predict_proba(x[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probabilities))
    return float(np.mean(aucs))


class _DomainClassifier(Detector):
    folds = 3
    view = "query"

    def _rows(self, observation: Observation) -> tuple[np.ndarray, list[str]]:
        return _query_rows(observation) if self.view == "query" else _doc_rows(observation)

    def fit(self, reference: Observation) -> None:
        vectors, ids = self._rows(reference)
        _validate(vectors, ids, "reference")
        self._reference = np.array(vectors, copy=True)
        self._reference_ids = ids.copy()

    def score(self, window: Observation) -> float:
        if not hasattr(self, "_reference"):
            raise RuntimeError("fit(reference) must be called before score(window)")
        vectors, ids = self._rows(window)
        return _cv_auc(self._reference, vectors, self._reference_ids, ids, self.folds)


@register
class QueryDomainClassifier(_DomainClassifier):
    name = "domain_classifier_query"
    view = "query"


@register
class RetrievedDocDomainClassifier(_DomainClassifier):
    name = "domain_classifier_retrieved_docs"
    view = "retrieved_docs"
