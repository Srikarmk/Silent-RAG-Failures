"""Shared data types. Everything else in the package passes these around."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset:
    name: str
    corpus: dict[str, str]  # doc_id -> "title text"
    queries: dict[str, str]  # query_id -> text
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}


@dataclass
class RetrievalState:
    """Everything an injector is allowed to change.

    The clean pipeline produces one of these, and each injector returns a
    modified copy. Severity is always "fraction affected" (0.1 to 0.9).
    """

    doc_ids: list[str]
    doc_texts: list[str]
    doc_embeddings: np.ndarray  # (n_docs, dim), float32
    query_ids: list[str]
    query_texts: list[str]
    config: dict = field(default_factory=dict)  # model, query prefix, normalize, chunking...


@dataclass
class Observation:
    """What a production system could log for one window of queries.

    There is deliberately no qrels field. Detectors only ever see this, so
    they can't use labels even by accident.
    """

    query_ids: list[str]
    query_embeddings: np.ndarray  # (n_queries, dim)
    topk_ids: list[list[str]]  # retrieved doc ids per query
    topk_scores: np.ndarray  # (n_queries, k), similarity scores
    doc_embeddings: np.ndarray | None = None  # for detectors that look at the index itself
