"""Negative controls (#9, #15): changes a real system goes through that move what
the detectors see without making retrieval worse. They run through the runner like
F1 to F5, with `harmful = False`, so every alarm they raise is a false alarm.

- H_mixed_truncation: a `severity` fraction of index rows re-embedded with the
  clean model at another max_seq_length (e.g. 512 instead of 256). Someone bumps
  the length setting and re-embeds part of the corpus. Doc side only.
- H_full_swap: documents and queries both moved to another model, so the system
  is consistent, just different. A finished model upgrade. All or nothing, so it
  only runs at severity 1.0.

Measured on NFCorpus in #9: neither drops nDCG@10 (both full swaps improve it),
and each moves at least one detector. Harmless has to stay measured, not assumed:
the runner still records `harmful` for every window, and a control that comes out
harmful is reported as a failed control.
"""

from __future__ import annotations

import dataclasses

import numpy as np

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
