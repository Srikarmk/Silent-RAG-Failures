"""F1 partial embedding upgrade (#8): a `severity` fraction of index rows is
re-embedded with a second model while queries stay on the clean one.

The replacement model has no default in code: `alt_model` and `alt_revision`
come from the config, since they decide how much damage F1 does.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from silent_rag.inject.base import Injector, register
from silent_rag.pipeline import Retriever, is_commit_hash
from silent_rag.types import RetrievalState

# documents for these must start with "passage: " (e5 model card); F1 encodes them bare, which is a second kind of damage
PREFIX_MODELS = ("intfloat/e5-", "intfloat/multilingual-e5-")


def check_alt_model(params: dict, run_cfg: dict, who: str) -> None:
    """Checks for an injector that encodes with a second model given as alt_model/alt_revision.
    `who` names the injector in the messages."""
    # binding already guaranteed both keys are present
    alt_model, alt_revision = params["alt_model"], params["alt_revision"]
    if not isinstance(alt_model, str) or not alt_model.strip():
        raise ValueError(f"alt_model must be a non-empty model name, got {alt_model!r}")
    # lowercase only: the revision goes into the cache key, so one commit gets one spelling
    if not is_commit_hash(alt_revision):
        raise ValueError(f"alt_revision must be a 40-character lowercase commit hash, got {alt_revision!r}")
    # any revision: the clean revision may be unpinned, and a new revision of one model often has the same weights
    if alt_model.strip().casefold() == run_cfg["model"].strip().casefold():
        raise ValueError(f"alt_model {alt_model!r} is the clean model; {who} needs a different model, whatever the revision")
    if alt_model.strip().casefold().startswith(PREFIX_MODELS):
        raise ValueError(f"alt_model {alt_model!r} needs a document prefix (e.g. 'passage: '), which {who} doesn't add")


def like(retriever_cls, retriever, model: str | None = None, **changes):
    """A retriever with the clean retriever's settings except `model` and `changes`. A new
    instance, not the clean one's encoder, so a changed model or length really is used."""
    settings = dict(
        normalize=retriever.normalize,
        query_prefix=retriever.query_prefix,
        max_seq_length=retriever.max_seq_length,
        revision=retriever.revision,
        batch_size=retriever.batch_size,
        device=retriever.device,
    )
    return retriever_cls(model or retriever.model, **{**settings, **changes})


@register
class PartialUpgrade(Injector):
    name = "F1_partial_upgrade"
    retriever_cls = Retriever  # tests swap in one with a fake encoder

    def __init__(self, alt_model: str, alt_revision: str):
        self.alt_model = alt_model
        self.alt_revision = alt_revision

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        check_alt_model(params, run_cfg, "F1")

    def attach(self, retriever, state: RetrievalState) -> None:
        # same settings as the clean retriever, only the model changes
        alt = like(self.retriever_cls, retriever, self.alt_model, revision=self.alt_revision)
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
