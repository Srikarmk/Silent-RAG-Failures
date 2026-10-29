import dataclasses
import inspect

import numpy as np
import pytest

from silent_rag import pipeline
from silent_rag.inject.f1_partial_upgrade import PartialUpgrade
from silent_rag.pipeline import Retriever
from test_pipeline import HashEncoder, toy_dataset

ALT = {"alt_model": "alt/model", "alt_revision": "abc123"}


@pytest.mark.parametrize("given", [{}, {"alt_model": ALT["alt_model"]}, {"alt_revision": ALT["alt_revision"]}])
def test_alt_model_and_revision_are_required(given):
    with pytest.raises(TypeError, match="missing a required argument"):
        inspect.signature(PartialUpgrade).bind(**given)
    inspect.signature(PartialUpgrade).bind(**ALT)


class SmallEncoder:
    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True):
        return np.ones((len(texts), 32), dtype=np.float32)


def attached(monkeypatch, encoder, seen=None, **clean_settings):
    class Alt(Retriever):
        def __init__(self, model, **kw):
            if seen is not None:
                seen.append((model, kw))
            super().__init__(model, encoder=encoder, **kw)

    monkeypatch.setattr(PartialUpgrade, "retriever_cls", Alt)
    clean = Retriever("clean/model", encoder=HashEncoder(), **clean_settings)
    state = clean.embed_corpus(toy_dataset())
    inj = PartialUpgrade(**ALT)
    inj.attach(clean, state)
    return inj, state


def test_alt_retriever_keeps_the_clean_settings_except_model_and_revision(monkeypatch):
    seen = []
    clean = dict(normalize=False, query_prefix="query: ", max_seq_length=128, revision="cleanrev", batch_size=8)
    inj, state = attached(monkeypatch, HashEncoder(), seen, **clean)
    assert seen == [(ALT["alt_model"], {**clean, "revision": ALT["alt_revision"], "device": "cpu"})]
    assert inj.alt_embeddings.shape == state.doc_embeddings.shape


def test_alt_corpus_is_encoded_once_then_loaded_from_the_cache(monkeypatch):
    first, second = HashEncoder(), HashEncoder()
    a, _ = attached(monkeypatch, first)
    b, _ = attached(monkeypatch, second)
    assert (first.calls, second.calls) == (1, 0)
    np.testing.assert_array_equal(a.alt_embeddings, b.alt_embeddings)
    assert len(list(pipeline.CACHE_DIR.glob("f1_alt-*.npy"))) == 1


def test_alt_model_with_another_dimension_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="alt/model gives 32-dim vectors but the index is 64-dim"):
        attached(monkeypatch, SmallEncoder())


class ShiftedEncoder(HashEncoder):
    """HashEncoder with its dims rolled by one: same size, different vector for every doc."""

    def encode(self, texts, **kw):
        return np.roll(super().encode(texts, **kw), 1, axis=1)


def upgraded_rows(damaged, state):
    return np.flatnonzero((damaged.doc_embeddings != state.doc_embeddings).any(axis=1))


@pytest.mark.parametrize("severity, count", [(0.1, 4), (0.5, 20), (0.9, 36), (1.0, 40)])
def test_severity_fraction_of_rows_get_the_alt_vectors(monkeypatch, severity, count):
    inj, state = attached(monkeypatch, ShiftedEncoder())
    assert not (inj.alt_embeddings == state.doc_embeddings).all(axis=1).any()
    damaged = inj.apply(state, severity, seed=0)
    rows = upgraded_rows(damaged, state)
    assert len(rows) == count
    np.testing.assert_array_equal(damaged.doc_embeddings[rows], inj.alt_embeddings[rows])


def test_rows_follow_the_seed_and_grow_with_severity(monkeypatch):
    inj, state = attached(monkeypatch, ShiftedEncoder())

    def rows(severity, seed):
        return set(upgraded_rows(inj.apply(state, severity, seed), state))

    assert rows(0.5, 0) == rows(0.5, 0)
    assert rows(0.5, 0) != rows(0.5, 1)
    assert rows(0.5, 0) < rows(0.9, 0)


def test_input_state_is_left_unchanged(monkeypatch):
    inj, state = attached(monkeypatch, ShiftedEncoder())
    before = state.doc_embeddings.copy()
    state.doc_embeddings.setflags(write=False)  # as the runner does
    inj.apply(state, 0.9, seed=0)
    np.testing.assert_array_equal(state.doc_embeddings, before)


def test_apply_before_attach_is_an_error():
    state = Retriever("clean/model", encoder=HashEncoder()).embed_corpus(toy_dataset())
    with pytest.raises(RuntimeError, match=r"call attach\(\) before apply\(\)"):
        PartialUpgrade(**ALT).apply(state, 0.5, seed=0)


def test_apply_on_another_state_than_attach_is_an_error(monkeypatch):
    inj, _ = attached(monkeypatch, ShiftedEncoder())
    other = Retriever("clean/model", encoder=HashEncoder()).embed_corpus(toy_dataset(30))
    with pytest.raises(ValueError, match="state has 30 rows but attach\\(\\) was given 40"):
        inj.apply(other, 0.5, seed=0)


def test_apply_on_same_count_but_other_doc_ids_is_an_error(monkeypatch):
    inj, state = attached(monkeypatch, ShiftedEncoder())
    reordered = dataclasses.replace(state, doc_ids=state.doc_ids[::-1])
    with pytest.raises(ValueError, match="doc_ids don't match the state passed to attach") as err:
        inj.apply(reordered, 0.5, seed=0)
    assert "rows" not in str(err.value)
