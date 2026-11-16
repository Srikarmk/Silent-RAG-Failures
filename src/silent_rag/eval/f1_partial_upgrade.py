"""F1-specific measurements: how different the replacement model's document vectors and
rankings are from the baseline model's. Kept small; another failure gets its own module."""

from __future__ import annotations

import numpy as np

from silent_rag.detect.re_embed import cosine_distance


def mean_cosine_baseline_vs_replacement_document_vectors(baseline_vectors, replacement_vectors) -> float:
    """Mean over documents of the cosine between each document's baseline and replacement vector.

    Rows are documents, in the same order in both arrays. The cosine is 1 - re_embed.cosine_distance,
    which normalises each row itself, so vectors need not be unit length.
    """
    baseline, replacement = np.asarray(baseline_vectors, dtype=float), np.asarray(replacement_vectors, dtype=float)
    if baseline.ndim != 2 or baseline.shape != replacement.shape:
        raise ValueError(f"need two (n_documents, dim) arrays of the same shape, got {baseline.shape} and {replacement.shape}")
    return float((1.0 - cosine_distance(baseline, replacement)).mean())


def topk_documents_shared_per_question(baseline_ids, switched_ids) -> np.ndarray:
    """Number of documents in both top-k lists, per question: len(set(a) & set(b)).
    Order within a list does not matter. Both arguments list one top-k per question, same order."""
    if len(baseline_ids) != len(switched_ids):
        raise ValueError(f"{len(baseline_ids)} baseline lists but {len(switched_ids)} switched lists")
    return np.array([len(set(a) & set(b)) for a, b in zip(baseline_ids, switched_ids)], dtype=int)
