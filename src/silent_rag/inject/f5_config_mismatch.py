"""F5 configuration mismatch (#14): a `severity` fraction of index rows is encoded
without the Normalize layer, while the index is still searched by inner product and the
questions stay normalised.

The realistic cause: documents added through a second pipeline that leaves Normalize
out, such as hand-written transformers code (model + mean pooling) or an ONNX export of
the Transformer and Pooling modules only. FAISS's inner product equals cosine similarity
only on unit vectors, so those rows are silently ranked by cosine times vector length.

The rows keep their direction and change only their length. The dose curve falls on
NFCorpus (whole-test-set nDCG@10 drop 0.200 at severity 0.1, 0.015 at 1.0): a few long
vectors crowd into every top 10. At 1.0 every row is long, and the ranking is cosine
weighted by length.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from silent_rag.inject.base import Injector, register
from silent_rag.inject.f1_partial_upgrade import like
from silent_rag.pipeline import Retriever
from silent_rag.types import RetrievalState


@register
class IndexWithoutNormalize(Injector):
    name = "F5_index_no_normalize"
    retriever_cls = Retriever  # tests swap in one with a fake encoder

    @staticmethod
    def without_normalize(encoder):
        """The clean encoder's modules minus Normalize (shared, not copied); tests swap this."""
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Normalize

        return SentenceTransformer(modules=[m for m in encoder if not isinstance(m, Normalize)], device=encoder.device)

    def attach(self, retriever, state: RetrievalState) -> None:
        alt = like(self.retriever_cls, retriever, normalize=False, encoder=self.without_normalize(retriever.encoder))
        # its own cache name: the key doesn't hold the module list, so it must never share the baseline's
        vecs, _ = alt.embed_texts("f5_nonorm", state.doc_ids, state.doc_texts)
        if np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4):
            raise ValueError(f"{self.name}: the vectors are unit length without Normalize too, so nothing would change")
        self.alt_embeddings = vecs
        self.attached_doc_ids = list(state.doc_ids)

    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState:
        if not hasattr(self, "alt_embeddings"):
            raise RuntimeError(f"{self.name}: call attach() before apply(), it builds the un-normalised vectors")
        n = len(state.doc_ids)
        if list(state.doc_ids) != self.attached_doc_ids:
            msg = f"{self.name}: state's doc_ids don't match the state passed to attach()"
            if n != len(self.attached_doc_ids):
                msg += f", state has {n} rows but attach() was given {len(self.attached_doc_ids)}"
            raise ValueError(msg)
        # F1's selection: one permutation per seed, cut at the severity, so higher severities change a superset
        rows = np.random.default_rng(seed).permutation(n)[: round(severity * n)]
        emb = state.doc_embeddings.copy()
        emb[rows] = self.alt_embeddings[rows]
        return dataclasses.replace(state, doc_embeddings=emb)
