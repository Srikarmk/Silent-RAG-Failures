"""Negative controls (#9, #15): changes a real system goes through that move what
the detectors see without making retrieval worse. They run through the runner like
F1 to F5, with `harmful = False`, so every alarm they raise is a false alarm.

- H_mixed_truncation: a `severity` fraction of index rows re-embedded with the
  clean model at another max_seq_length (e.g. 512 instead of 256). Someone bumps
  the length setting and re-embeds part of the corpus. Doc side only.
- H_full_swap: documents and queries both moved to another model, so the system
  is consistent, just different. A finished model upgrade. All or nothing, so it
  only runs at severity 1.0.
- H_offtopic_docs: documents from another corpus added to the index, embedded
  with the clean model. The harmless look-alike of F2's distractors (#72).
- H_query_topic_shift: traffic oversamples one topic the corpus can answer. The
  harmless look-alike of F4's out-of-corpus queries (#72).

Measured on NFCorpus in #9: neither drops nDCG@10 (both full swaps improve it),
and each moves at least one detector. Harmless has to stay measured, not assumed:
the runner still records `harmful` for every window, and a control that comes out
harmful is reported as a failed control.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from silent_rag import data
from silent_rag.inject.base import Injector, register
from silent_rag.inject.f1_partial_upgrade import check_alt_model, like
from silent_rag.pipeline import Retriever
from silent_rag.types import RetrievalState


class _Attached(Injector):
    """Shared checks: apply() needs attach(), on the same state."""

    def _check_state(self, state: RetrievalState) -> None:
        if not hasattr(self, "attached_doc_ids"):
            raise RuntimeError(f"{self.name}: call attach() before apply()")
        if list(state.doc_ids) != self.attached_doc_ids:
            raise ValueError(f"{self.name}: state's doc_ids don't match the state passed to attach()")


def _same_dim(name: str, what: str, vecs: np.ndarray, clean: np.ndarray) -> None:
    # the detectors are fit on clean vectors, so a control has to keep the dimension
    if vecs.shape[1] != clean.shape[1]:
        raise ValueError(f"{name}: {what} gives {vecs.shape[1]}-dim vectors but the index is {clean.shape[1]}-dim")


@register
class MixedTruncation(_Attached):
    name = "H_mixed_truncation"
    harmful = False
    retriever_cls = Retriever  # tests swap in one with a fake encoder

    def __init__(self, max_seq_length: int):
        self.max_seq_length = max_seq_length

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        n = params["max_seq_length"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"max_seq_length must be a positive integer, got {n!r}")

    def attach(self, retriever, state: RetrievalState) -> None:
        if self.max_seq_length == retriever.max_seq_length:
            raise ValueError(
                f"{self.name}: max_seq_length {self.max_seq_length} is the clean setting, so nothing would change"
            )
        alt = like(self.retriever_cls, retriever, max_seq_length=self.max_seq_length)
        vecs, _ = alt.embed_texts("h_mixed_truncation", state.doc_ids, state.doc_texts)
        _same_dim(self.name, f"max_seq_length {self.max_seq_length}", vecs, state.doc_embeddings)
        self.alt_embeddings = vecs
        self.attached_doc_ids = list(state.doc_ids)

    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState:
        self._check_state(state)
        # same permutation-and-cut as F1: higher severities re-embed a superset of rows
        rows = np.random.default_rng(seed).permutation(len(state.doc_ids))[: round(severity * len(state.doc_ids))]
        emb = state.doc_embeddings.copy()
        emb[rows] = self.alt_embeddings[rows]
        return dataclasses.replace(state, doc_embeddings=emb)


@register
class OffTopicDocs(_Attached):
    """Off-topic documents added to the index, embedded properly with the clean model.

    `severity` is the number added as a fraction of the corpus (0.5 on NFCorpus adds
    1,816 docs). The index grows and the doc-side distribution moves, the way F2's
    distractors do, but nobody asks about these documents, so they shouldn't outrank
    relevant ones. Whether they do is measured, not assumed: a distractor source that
    lowers nDCG@10 is a failed control. Each seed takes one permutation of the source,
    cut at the severity, so higher severities add a superset of documents.
    """

    name = "H_offtopic_docs"
    harmful = False
    load = staticmethod(data.load_distractors)  # tests swap in a fake corpus

    def __init__(self, source: str = "arguana"):
        self.source = source

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        source = params.get("source", "arguana")
        if source not in data.DISTRACTOR_SOURCES:
            raise ValueError(f"source must be one of {sorted(data.DISTRACTOR_SOURCES)}, got {source!r}")

    def attach(self, retriever, state: RetrievalState) -> None:
        corpus = self.load(self.source)
        ids = sorted(corpus)
        # prefixed so an added doc can never collide with, or be scored as, a doc of the dataset
        self.added_ids = [f"{self.source}:{d}" for d in ids]
        self.added_texts = [corpus[d] for d in ids]
        self.added_embeddings, _ = retriever.embed_texts(f"h_offtopic_{self.source}", self.added_ids, self.added_texts)
        _same_dim(self.name, self.source, self.added_embeddings, state.doc_embeddings)
        self.attached_doc_ids = list(state.doc_ids)

    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState:
        self._check_state(state)
        n = round(severity * len(state.doc_ids))
        if n > len(self.added_ids):
            raise ValueError(
                f"{self.name}: severity {severity} adds {n} docs but {self.source} only has {len(self.added_ids)}"
            )
        rows = np.random.default_rng(seed).permutation(len(self.added_ids))[:n]
        ids = [self.added_ids[i] for i in rows]
        return dataclasses.replace(
            state,
            doc_ids=list(state.doc_ids) + ids,
            doc_texts=list(state.doc_texts) + [self.added_texts[i] for i in rows],
            doc_embeddings=np.vstack([state.doc_embeddings, self.added_embeddings[rows]]),
            # chunked index: each added doc is one row and its own parent
            parent_ids=None if state.parent_ids is None else list(state.parent_ids) + ids,
        )


@register
class QueryTopicShift(_Attached):
    """Traffic moves toward one topic the corpus can answer (#72), the harmless look-alike
    of F4's shift to topics outside the corpus.

    Topics are k-means clusters of the clean query vectors, fit on the query pool only, so
    the reference window is never asked. `topic` 0 is the largest cluster. At severity s each
    pool position is, with probability s, asked one of the topic's queries instead, so the
    window oversamples the topic and repeats some of its questions, as real traffic does.
    Each position's draw and replacement are fixed per seed, so higher severities move a
    superset of positions.

    The index doesn't change, so every question asked gets exactly its clean result: the
    nDCG@10 drop is 0 by construction. What it tests is whether a detector false-alarms on a
    change in what users ask about.
    """

    name = "H_query_topic_shift"
    harmful = False

    def __init__(self, n_topics: int = 5, topic: int = 0):
        self.n_topics = n_topics
        self.topic = topic

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        n, topic = params.get("n_topics", 5), params.get("topic", 0)
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError(f"n_topics must be an integer >= 2, got {n!r}")
        if isinstance(topic, bool) or not isinstance(topic, int) or not 0 <= topic < n:
            raise ValueError(f"topic must be an integer in [0, n_topics), got {topic!r}")

    def attach(self, retriever, state: RetrievalState) -> None:
        from sklearn.cluster import KMeans

        if self.query_pool is None:
            raise RuntimeError(f"{self.name}: needs query_pool, which the runner sets before attach()")
        if state.query_embeddings is None:
            raise ValueError(f"{self.name}: needs the clean query_embeddings")
        position = {q: i for i, q in enumerate(state.query_ids)}
        self.pool_positions = np.array([position[q] for q in self.query_pool])
        if len(self.pool_positions) < self.n_topics:
            raise ValueError(f"{self.name}: {len(self.pool_positions)} pool queries is too few for {self.n_topics} topics")
        vecs = np.asarray(state.query_embeddings, dtype=np.float64)[self.pool_positions]
        labels = KMeans(n_clusters=self.n_topics, n_init=10, random_state=0).fit_predict(vecs)
        # topics by size, largest first; ties broken by label so the order is deterministic
        by_size = sorted(range(self.n_topics), key=lambda c: (-(labels == c).sum(), c))
        self.topic_positions = self.pool_positions[labels == by_size[self.topic]]
        self.attached_doc_ids = list(state.doc_ids)

    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState:
        self._check_state(state)
        rng = np.random.default_rng(seed)
        draws = rng.random(len(self.pool_positions))
        picks = self.topic_positions[rng.integers(0, len(self.topic_positions), len(self.pool_positions))]
        source = np.arange(len(state.query_ids))
        moved = draws < severity
        source[self.pool_positions[moved]] = picks[moved]
        return dataclasses.replace(
            state,
            query_ids=[state.query_ids[j] for j in source],
            query_texts=[state.query_texts[j] for j in source],
            query_embeddings=np.asarray(state.query_embeddings)[source],
        )


@register
class FullSwap(_Attached):
    name = "H_full_swap"
    harmful = False
    severities = (1.0,)  # a half-swapped system is F1, not a control
    retriever_cls = Retriever

    def __init__(self, alt_model: str, alt_revision: str):
        self.alt_model = alt_model
        self.alt_revision = alt_revision

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        check_alt_model(params, run_cfg, cls.name)

    def attach(self, retriever, state: RetrievalState) -> None:
        alt = like(self.retriever_cls, retriever, self.alt_model, revision=self.alt_revision)
        # same cache name and key as F1, so a run with both encodes the alt corpus once
        docs, _ = alt.embed_texts("f1_alt", state.doc_ids, state.doc_texts)
        _same_dim(self.name, self.alt_model, docs, state.doc_embeddings)
        self.doc_embeddings = docs
        self.query_embeddings = alt.embed_queries(state.query_texts)
        self.attached_doc_ids = list(state.doc_ids)

    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState:
        self._check_state(state)
        if severity != 1.0:
            raise ValueError(f"{self.name} is all or nothing and only runs at severity 1.0, got {severity}")
        return dataclasses.replace(state, doc_embeddings=self.doc_embeddings, query_embeddings=self.query_embeddings)
