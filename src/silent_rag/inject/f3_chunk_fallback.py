"""F3 chunking change on part of the corpus (#12): in a chunked index, a `severity`
fraction of docs falls back to one vector per doc, the text truncated at the model's
max_seq_length. The pipeline bug it stands for: a new ingestion path (or a config
change partway through) stops chunking, so the index holds two kinds of rows.

Measured before building it (MLDR English, 20k-doc subsample, 800 test queries,
multi-qa-MiniLM, 256-token chunks): nDCG@10 is 0.671 chunked and 0.361 truncated, and
the fallback costs 0.037 / 0.121 / 0.197 / 0.325 at severities 0.1 / 0.3 / 0.5 / 0.9.
The fallen-back docs almost vanish from the top 10 (1% of the slots at severity 0.1),
since one vector competes against other docs' best of ~12 chunks.

On NFCorpus the same change is harmless: its docs are short and the relevance sits in
the title and first sentences, so every chunking or truncation variant there moved
nDCG@10 by less than 0.03 (measured in #12). Use it on a chunked long-document index.

Every row is the clean model's vector for its own text, so re_embed can't see it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from silent_rag.inject.base import Injector, register
from silent_rag.types import RetrievalState


@register
class ChunkFallback(Injector):
    name = "F3_chunk_fallback"

    def attach(self, retriever, state: RetrievalState) -> None:
        if state.parent_ids is None:
            raise ValueError(f"{self.name} needs a chunked clean index (retriever chunk_size); this one has one row per doc")
        # each doc's rows, in index order; the chunks join back into the doc's text exactly
        self.docs: list[str] = list(dict.fromkeys(state.parent_ids))
        rows: dict[str, list[int]] = {}
        for i, parent in enumerate(state.parent_ids):
            rows.setdefault(parent, []).append(i)
        for d, r in rows.items():
            if r != list(range(r[0], r[0] + len(r))):
                raise ValueError(f"{self.name}: chunks of doc {d!r} aren't contiguous in the index")
        self.rows = [rows[d] for d in self.docs]
        texts = ["".join(state.doc_texts[i] for i in r) for r in self.rows]
        # one vector per doc, truncated at max_seq_length like the unchunked pipeline
        self.single_ids = [f"{d}#all" for d in self.docs]
        self.single_texts = texts
        self.single_embeddings, _ = retriever.embed_texts("f3_single", self.single_ids, texts)
        if self.single_embeddings.shape[1] != state.doc_embeddings.shape[1]:
            raise ValueError(f"{self.name}: single-vector embeddings don't match the index dimension")
        self.attached_doc_ids = list(state.doc_ids)

    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState:
        if not hasattr(self, "attached_doc_ids"):
            raise RuntimeError(f"{self.name}: call attach() before apply()")
        if list(state.doc_ids) != self.attached_doc_ids:
            raise ValueError(f"{self.name}: state's doc_ids don't match the state passed to attach()")
        n = len(self.docs)
        # one permutation per seed, cut at the severity: higher severities change a superset of docs
        fallback = set(np.random.default_rng(seed).permutation(n)[: round(severity * n)].tolist())
        ids, texts, parents, take = [], [], [], []  # take: row of the clean index, or -1 - doc position for a single vector
        for j, (d, r) in enumerate(zip(self.docs, self.rows)):
            if j in fallback:
                ids.append(self.single_ids[j]); texts.append(self.single_texts[j]); parents.append(d); take.append(-1 - j)
            else:
                ids += [state.doc_ids[i] for i in r]; texts += [state.doc_texts[i] for i in r]
                parents += [d] * len(r); take += r
        take = np.array(take)
        emb = np.empty((len(take), state.doc_embeddings.shape[1]), dtype=state.doc_embeddings.dtype)
        kept = take >= 0
        emb[kept] = state.doc_embeddings[take[kept]]
        emb[~kept] = self.single_embeddings[-1 - take[~kept]]
        return dataclasses.replace(state, doc_ids=ids, doc_texts=texts, doc_embeddings=emb, parent_ids=parents)
