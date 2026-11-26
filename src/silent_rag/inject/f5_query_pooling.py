"""F5 configuration mismatch (#14): a `severity` fraction of questions is encoded with
CLS pooling instead of the model's mean pooling, while the index stays as built.

The realistic cause: a query service that encodes with plain transformers code instead
of sentence-transformers. Model cards warn that outside the library you must apply the
model's own pooling (e.g.
https://huggingface.co/pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb), and
taking the [CLS] token's vector is the common default in such code. The vectors keep the
model's Normalize step, so only the pooling differs.

Severity is the fraction of questions. On NFCorpus the whole-test-set nDCG@10 drop is
small (0.001 at 0.1, 0.011 at 0.5, 0.025 at 1.0): below the drop of about 0.06 that a
window of 100 questions detects 80% of the time (README, window size), so few windows
will count as harmful even at full severity.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from silent_rag.inject.base import Injector, register
from silent_rag.inject.f1_partial_upgrade import like
from silent_rag.pipeline import Retriever
from silent_rag.types import RetrievalState


@register
class QueryClsPooling(Injector):
    name = "F5_query_cls_pooling"
    retriever_cls = Retriever  # tests swap in one with a fake encoder

    @staticmethod
    def cls_encoder(encoder):
        """The clean encoder's modules with each Pooling swapped for CLS pooling; the Transformer
        and Normalize modules are shared, not copied. Tests swap this."""
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Pooling

        modules = [Pooling(m.embedding_dimension, pooling_mode="cls") if isinstance(m, Pooling) else m for m in encoder]
        return SentenceTransformer(modules=modules, device=encoder.device)

    def attach(self, retriever, state: RetrievalState) -> None:
        alt = like(self.retriever_cls, retriever, encoder=self.cls_encoder(retriever.encoder))
        self.query_vectors = alt.embed_queries(state.query_texts)
        self.attached_query_ids = list(state.query_ids)

    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState:
        if not hasattr(self, "query_vectors"):
            raise RuntimeError(f"{self.name}: call attach() before apply(), it builds the CLS-pooled question vectors")
        n = len(state.query_ids)
        if list(state.query_ids) != self.attached_query_ids:
            msg = f"{self.name}: state's query_ids don't match the state passed to attach()"
            if n != len(self.attached_query_ids):
                msg += f", state has {n} questions but attach() was given {len(self.attached_query_ids)}"
            raise ValueError(msg)
        # F1's selection over question positions: one permutation per seed, cut at the severity
        rows = np.random.default_rng(seed).permutation(n)[: round(severity * n)]
        q = np.array(state.query_embeddings, copy=True)
        q[rows] = self.query_vectors[rows]
        return dataclasses.replace(state, query_embeddings=q)
