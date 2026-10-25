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

    Rows of doc_embeddings are index entries. Without chunking they are whole
    docs and parent_ids is None. With chunking, doc_ids are chunk ids and
    parent_ids[i] is the doc that row i came from; search ranks docs by their
    best chunk.

    query_embeddings are computed once by the retriever. An injector that
    changes how queries are encoded (F5) edits them directly. One that changes
    query_texts (F4) sets query_embeddings to None so the runner re-embeds.
    Query positions never change, the runner lines them up with the clean run.
    """

    doc_ids: list[str]
    doc_texts: list[str]
    doc_embeddings: np.ndarray  # (n_rows, dim), float32
    query_ids: list[str]
    query_texts: list[str]
    query_embeddings: np.ndarray | None = None  # (n_queries, dim)
    parent_ids: list[str] | None = None  # doc id per row when chunked
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
    query_texts: list[str] | None = None  # for the LLM judge and the re-embedding check
    # the index itself, row-aligned, for detectors that look at it
    doc_ids: list[str] | None = None
    doc_texts: list[str] | None = None
    doc_embeddings: np.ndarray | None = None
    parent_ids: list[str] | None = None  # doc id per row when chunked; topk_ids are always parent docs
