"""F1 partial embedding upgrade (#8): a `severity` fraction of index rows is
re-embedded with a second model while queries stay on the clean one.

The replacement model has no default in code: `alt_model` and `alt_revision`
come from the config, since they decide how much damage F1 does.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from silent_rag.inject.base import Injector, register
from silent_rag.inject.model_check import check_replacement_model
from silent_rag.pipeline import Retriever, is_commit_hash
from silent_rag.types import RetrievalState

# documents for these must start with "passage: " (e5 model card); F1 encodes them bare, which is a second kind of damage
PREFIX_MODELS = ("intfloat/e5-", "intfloat/multilingual-e5-")


@register
class PartialUpgrade(Injector):
    name = "F1_partial_upgrade"
    retriever_cls = Retriever  # tests swap in one with a fake encoder
    model_checker = staticmethod(check_replacement_model)  # reads the local HF cache; tests swap in a fake

    def __init__(self, alt_model: str, alt_revision: str):
        self.alt_model = alt_model
        self.alt_revision = alt_revision

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        # binding already guaranteed both keys are present
        alt_model, alt_revision = params["alt_model"], params["alt_revision"]
        if not isinstance(alt_model, str) or not alt_model.strip():
            raise ValueError(f"alt_model must be a non-empty model name, got {alt_model!r}")
        # lowercase only: the revision goes into the cache key, so one commit gets one spelling
        if not is_commit_hash(alt_revision):
            raise ValueError(f"alt_revision must be a 40-character lowercase commit hash, got {alt_revision!r}")
        # any revision: the clean revision may be unpinned, and a new revision of one model often has the same weights
        if alt_model.strip().casefold() == run_cfg["model"].strip().casefold():
            raise ValueError(f"alt_model {alt_model!r} is the clean model; F1 needs a different model, whatever the revision")
        if alt_model.strip().casefold().startswith(PREFIX_MODELS):
            raise ValueError(f"alt_model {alt_model!r} needs a document prefix (e.g. 'passage: '), which F1 doesn't add")
        # last, after the checks that need no files: Normalize, dimension and prefixes from the local cache
        cls.model_checker(alt_model, alt_revision, run_cfg["model"], run_cfg["revision"])

    def build_replacement_model_retriever(self, baseline_retriever, query_prefix: str | None = None):
        """A retriever for the replacement model with the baseline retriever's settings.

        Copies normalize, max_seq_length, batch_size and device; uses alt_model and
        alt_revision; never copies the encoder (that is the baseline model). query_prefix=None
        inherits the baseline's prefix; a string replaces it, and "" clears it. F1 itself only
        encodes documents, so the prefix matters only to callers that encode questions with
        the replacement model (e.g. bge needs its own query instruction).
        """
        return self.retriever_cls(
            self.alt_model,
            normalize=baseline_retriever.normalize,
            query_prefix=baseline_retriever.query_prefix if query_prefix is None else query_prefix,
            max_seq_length=baseline_retriever.max_seq_length,
            revision=self.alt_revision,
            batch_size=baseline_retriever.batch_size,
            device=baseline_retriever.device,
        )

    def attach(self, retriever, state: RetrievalState) -> None:
        alt = self.build_replacement_model_retriever(retriever)
        vecs, _ = alt.embed_texts("f1_alt", state.doc_ids, state.doc_texts)
        if vecs.shape[1] != state.doc_embeddings.shape[1]:
            raise ValueError(
                f"{self.alt_model} gives {vecs.shape[1]}-dim vectors but the index is "
                f"{state.doc_embeddings.shape[1]}-dim, F1 needs a model with the same dimension"
            )
        self.alt_embeddings = vecs
        self.attached_doc_ids = list(state.doc_ids)

    @staticmethod
    def replaced_document_row_indices(n_documents: int, severity: float, seed: int) -> np.ndarray:
        """Index rows whose vectors apply() replaces with the replacement model's, in permutation order.

        One permutation per seed, cut at round(severity * n_documents): within a seed, a higher
        severity replaces a superset of rows (the lower severity's rows are a prefix).
        """
        return np.random.default_rng(seed).permutation(n_documents)[: round(severity * n_documents)]

    def modified_document_row_indices(self, n_documents: int, severity: float, seed: int) -> np.ndarray:
        """Rows apply(state, severity, seed) changes, for per-question tables."""
        return self.replaced_document_row_indices(n_documents, severity, seed)

    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState:
        if not hasattr(self, "alt_embeddings"):
            raise RuntimeError(f"{self.name}: call attach() before apply(), it builds the alt model's vectors")
        n = len(state.doc_ids)
        if list(state.doc_ids) != self.attached_doc_ids:
            msg = f"{self.name}: state's doc_ids don't match the state passed to attach()"
            if n != len(self.attached_doc_ids):
                msg += f", state has {n} rows but attach() was given {len(self.attached_doc_ids)}"
            raise ValueError(msg)
        rows = self.replaced_document_row_indices(n, severity, seed)
        emb = state.doc_embeddings.copy()
        emb[rows] = self.alt_embeddings[rows]
        return dataclasses.replace(state, doc_embeddings=emb)
