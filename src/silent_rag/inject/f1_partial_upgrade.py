"""F1 partial embedding upgrade (#8): a `severity` fraction of index rows is
re-embedded with a second model while queries stay on the clean one.

The replacement model has no default in code: `alt_model` and `alt_revision`
come from the config, since they decide how much damage F1 does.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from silent_rag.inject.base import Injector
from silent_rag.pipeline import Retriever
from silent_rag.types import RetrievalState


class PartialUpgrade(Injector):
    name = "F1_partial_upgrade"
    retriever_cls = Retriever  # tests swap in one with a fake encoder

    def __init__(self, alt_model: str, alt_revision: str):
        self.alt_model = alt_model
        self.alt_revision = alt_revision

    def attach(self, retriever, state: RetrievalState) -> None:
        # same settings as the clean retriever, only the model changes; not its encoder, that's the clean model
        alt = self.retriever_cls(
            self.alt_model,
            normalize=retriever.normalize,
            query_prefix=retriever.query_prefix,
            max_seq_length=retriever.max_seq_length,
            revision=self.alt_revision,
            batch_size=retriever.batch_size,
            device=retriever.device,
        )
        vecs, _ = alt.embed_texts("f1_alt", state.doc_ids, state.doc_texts)
        if vecs.shape[1] != state.doc_embeddings.shape[1]:
            raise ValueError(
                f"{self.alt_model} gives {vecs.shape[1]}-dim vectors but the index is "
                f"{state.doc_embeddings.shape[1]}-dim, F1 needs a model with the same dimension"
            )
        self.alt_embeddings = vecs
        self.attached_doc_ids = list(state.doc_ids)

    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState:
        if not hasattr(self, "alt_embeddings"):
            raise RuntimeError(f"{self.name}: call attach() before apply(), it builds the alt model's vectors")
        n = len(state.doc_ids)
        if list(state.doc_ids) != self.attached_doc_ids:
            msg = f"{self.name}: state's doc_ids don't match the state passed to attach()"
            if n != len(self.attached_doc_ids):
                msg += f", state has {n} rows but attach() was given {len(self.attached_doc_ids)}"
            raise ValueError(msg)
        # one permutation per seed, cut at the severity: higher severities upgrade a superset of rows
        rows = np.random.default_rng(seed).permutation(n)[: round(severity * n)]
        emb = state.doc_embeddings.copy()
        emb[rows] = self.alt_embeddings[rows]
        return dataclasses.replace(state, doc_embeddings=emb)
