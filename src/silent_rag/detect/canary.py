"""Canary queries (#24): a fixed set of probe queries whose clean results are saved,
rerun on every check to see how far the results moved.

The canaries are `n` queries from the dataset's training split, so they never appear
in a window the runner scores (the windows are test queries; the splits don't overlap
on NFCorpus, SciFact or FiQA). fit() embeds them once with the clean retriever and
saves their clean top-k on the reference window's index. score() searches the window's
index with the same canary vectors and returns

    1 - mean over canaries of |clean top-k & current top-k| / k

so 0 means every canary got back the same documents and 1 means none did. Order inside
the top k doesn't count: overlap@k, not a rank correlation, so a reshuffle among the
same ten documents is not an alarm.

On a clean index every canary gets exactly its saved results and the score is 0: the
same vectors searched in the same index. So, like re_embed, the threshold is 0 and any
change that moves a canary's results alarms, harmful or not.

What it sees: anything on the document side that changes which documents come back,
F1 (part of the index on another model), F2 (added documents that outrank old ones),
F3 (re-chunking). What it can't see, by construction:
- changes to the users' queries (F4, the topic shift): the canaries are fixed
- the query encoder going wrong (F5): the canaries are embedded once with the clean
  retriever, since the runner gives detectors the index but not the system's encoder.
  A real deployment would push its canaries through the live encoder and would see F5
- harm that leaves results unchanged for the canaries, e.g. damage confined to topics
  the sample doesn't cover; a bigger `n` covers more of the corpus

Cost: one encode of `n` queries per dataset and seed, then one exact search of `n`
queries per check.

Canary set size: n = 100 (#24). On NFCorpus (F1 with two partners, F2 distractors and
the four negative controls, 10 seeds) n = 25, 50, 100 and 200 alarm on exactly the same
windows at threshold 0: every harmful F1 and F2 window, every window of every
document-side control, no topic-shift window. Size only matters for damage confined to
a small part of the corpus: with a fraction p of documents damaged, the chance that no
canary's top 10 touches one is roughly (1 - p) ** (10 * n), about 8% at n = 25 and
practically 0 at n = 100 for p = 0.01. n = 100 costs ~8 ms a check (n = 200 ~16 ms).
"""

from __future__ import annotations

import numpy as np

from silent_rag import data
from silent_rag.detect.base import Detector, register
from silent_rag.types import Observation, RetrievalState


@register
class Canary(Detector):
    name = "canary"
    load = staticmethod(data.load)  # tests swap in a fake dataset
    _queries: dict[str, dict[str, str]] = {}  # training queries per dataset, loaded once per process

    def __init__(self, n: int = 100, k: int = 10, seed: int = 0):
        for label, v in (("n", n), ("k", k)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{label} has to be a positive integer, got {v!r}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed has to be a non-negative integer, got {seed!r}")
        self.n, self.k, self.seed = n, k, seed
        self.retriever = None

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        k = run_cfg.get("k", 10)
        if params.get("k", 10) > k:
            raise ValueError(f"k {params['k']} is deeper than the k={k} the run retrieves")

    def attach(self, retriever) -> None:
        self.retriever = retriever

    def canaries(self, exclude) -> tuple[list[str], list[str]]:
        """`n` training queries of the dataset, the same ones for every seed of a run."""
        if self.dataset is None:
            raise RuntimeError("canary needs the runner to set .dataset before fit()")
        if self.dataset not in Canary._queries:
            Canary._queries[self.dataset] = self.load(self.dataset, split="train").queries
        pool = {q: t for q, t in Canary._queries[self.dataset].items() if q not in set(exclude)}
        if len(pool) < self.n:
            raise ValueError(f"canary: {self.dataset} has {len(pool)} training queries, fewer than n={self.n}")
        ids = sorted(pool)
        picked = sorted(np.random.default_rng(self.seed).choice(len(ids), self.n, replace=False))
        return [ids[i] for i in picked], [pool[ids[i]] for i in picked]

    def results(self, window: Observation) -> list[set[str]]:
        """Each canary's top-k document ids on the window's index."""
        if window.doc_ids is None or window.doc_embeddings is None:
            raise ValueError("canary needs doc_ids and doc_embeddings in the Observation")
        state = RetrievalState(
            doc_ids=list(window.doc_ids),
            doc_texts=list(window.doc_texts) if window.doc_texts is not None else [""] * len(window.doc_ids),
            doc_embeddings=window.doc_embeddings,
            query_ids=self.ids_,
            query_texts=self.texts_,
            query_embeddings=self.vectors_,
            parent_ids=None if window.parent_ids is None else list(window.parent_ids),
        )
        ids, _, _ = self.retriever.search(state, k=self.k, query_embeddings=self.vectors_)
        return [set(row) for row in ids]

    def fit(self, reference: Observation) -> None:
        if self.retriever is None:
            raise RuntimeError("canary needs attach(retriever) before fit()")
        # never a query the runner scores; on BEIR the splits are disjoint, so this is a backstop
        self.ids_, self.texts_ = self.canaries(exclude=reference.query_ids)
        self.vectors_ = self.retriever.embed_queries(self.texts_)
        self.clean_ = self.results(reference)

    def score(self, window: Observation) -> float:
        if not hasattr(self, "clean_"):
            raise RuntimeError("canary needs fit(reference) before it can score")
        if set(self.ids_) & set(window.query_ids):
            raise RuntimeError("a canary query is in the window; canaries must stay out of scored traffic")
        now = self.results(window)
        return float(1.0 - np.mean([len(c & m) / self.k for c, m in zip(self.clean_, now)]))
