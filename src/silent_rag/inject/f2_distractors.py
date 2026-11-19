"""F2, distractor half (#11): documents from another corpus added to the index.

Found while building the negative controls (#72): ArguAna debate arguments added to
NFCorpus, embedded with the clean model, lower nDCG@10 by 0.004 / 0.010 / 0.013 at
severities 0.1 / 0.5 / 0.9 (one seed, 20 windows each), significant in 80-100% of
windows, because some arguments are about health and diet and outrank relevant
documents. Small, real, and invisible to re_embed, since every row is correct for its
text. The stale-document half of F2 (HoH) is separate.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from silent_rag import data
from silent_rag.inject.base import Injector, register
from silent_rag.types import RetrievalState


@register
class Distractors(Injector):
    """Distractor documents from another corpus added to the index, embedded properly
    with the clean model.

    `severity` is the number added as a fraction of the corpus (0.5 on NFCorpus adds
    1,816 docs). Nothing in the index is mismatched, so `re_embed` can't see it; the
    damage is that some distractors outrank relevant documents. Each seed takes one
    permutation of the source, cut at the severity, so higher severities add a superset.
    """

    name = "F2_distractors"
    load = staticmethod(data.load_distractors)  # tests swap in a fake corpus

    def __init__(self, source: str = "arguana"):
        self.source = source

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        source = params.get("source", "arguana")
        if source not in data.DISTRACTOR_SOURCES:
            raise ValueError(f"source must be one of {sorted(data.DISTRACTOR_SOURCES)}, got {source!r}")

    @classmethod
    def check_dataset(cls, params: dict, severities: list[float], dataset) -> None:
        # severity is a share of the target corpus, so a big corpus can need more docs than
        # the source has (ArguAna: ~8.7k, so FiQA's ~57k docs fail above ~0.15)
        source = params.get("source", "arguana")
        available = len(cls.load(source))
        need = round(max(severities) * len(dataset.corpus))
        if need > available:
            most = available / len(dataset.corpus)
            raise ValueError(f"severity {max(severities)} adds {need} docs but {source} has {available}; "
                             f"on this dataset the largest severity that fits is {most:.3f}")

    def attach(self, retriever, state: RetrievalState) -> None:
        corpus = self.load(self.source)
        ids = sorted(corpus)
        # prefixed so an added doc can never collide with, or be scored as, a doc of the dataset
        self.added_ids = [f"{self.source}:{d}" for d in ids]
        self.added_texts = [corpus[d] for d in ids]
        self.added_embeddings, _ = retriever.embed_texts(f"f2_distractors_{self.source}", self.added_ids, self.added_texts)
        if self.added_embeddings.shape[1] != state.doc_embeddings.shape[1]:
            raise ValueError(f"{self.name}: {self.source} gives {self.added_embeddings.shape[1]}-dim vectors "
                             f"but the index is {state.doc_embeddings.shape[1]}-dim")
        self.attached_doc_ids = list(state.doc_ids)

    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState:
        if not hasattr(self, "attached_doc_ids"):
            raise RuntimeError(f"{self.name}: call attach() before apply()")
        if list(state.doc_ids) != self.attached_doc_ids:
            raise ValueError(f"{self.name}: state's doc_ids don't match the state passed to attach()")
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
