import dataclasses

import numpy as np
import pytest

from silent_rag.inject.base import REGISTRY
from silent_rag.inject.f5_query_pooling import QueryClsPooling
from silent_rag.pipeline import Retriever
from silent_rag.run import load_plugins
from test_inject_f1 import ShiftedEncoder
from test_pipeline import HashEncoder, toy_dataset


def attached(monkeypatch, n=40):
    """The injector attached to a clean state; its CLS-pooling encoder is ShiftedEncoder (unit length, different)."""
    monkeypatch.setattr(QueryClsPooling, "cls_encoder", staticmethod(lambda clean_encoder: ShiftedEncoder()))
    clean = Retriever("clean/model", encoder=HashEncoder())
    state = clean.embed_corpus(toy_dataset(n))
    inj = QueryClsPooling()
    inj.attach(clean, state)
    return inj, state


def changed_rows(damaged, state):
    return np.flatnonzero((damaged.query_embeddings != state.query_embeddings).any(axis=1))


def test_registered_as_harmful():
    load_plugins()
    assert REGISTRY["F5_query_cls_pooling"] is QueryClsPooling
    assert QueryClsPooling.harmful


@pytest.mark.parametrize("severity, count", [(0.1, 4), (0.5, 20), (1.0, 40)])
def test_only_the_selected_rows_change(monkeypatch, severity, count):
    inj, state = attached(monkeypatch)
    damaged = inj.apply(state, severity, seed=0)
    rows = changed_rows(damaged, state)
    assert sorted(rows) == sorted(np.random.default_rng(0).permutation(40)[:count])
    np.testing.assert_array_equal(damaged.query_embeddings[rows], inj.query_vectors[rows])


def test_changed_rows_differ_from_the_clean_vectors_and_are_unit_length(monkeypatch):
    inj, state = attached(monkeypatch)
    damaged = inj.apply(state, 1.0, seed=0)
    assert not (damaged.query_embeddings == state.query_embeddings).all(axis=1).any()
    np.testing.assert_allclose(np.linalg.norm(damaged.query_embeddings, axis=1), 1.0, atol=1e-6)


def test_apply_before_attach_is_an_error():
    state = Retriever("clean/model", encoder=HashEncoder()).embed_corpus(toy_dataset())
    with pytest.raises(RuntimeError, match=r"call attach\(\) before apply\(\)"):
        QueryClsPooling().apply(state, 0.5, seed=0)


def test_apply_on_a_state_with_other_question_count_is_an_error(monkeypatch):
    inj, _ = attached(monkeypatch)
    other = Retriever("clean/model", encoder=HashEncoder()).embed_corpus(toy_dataset(30))
    with pytest.raises(ValueError, match=r"state has 30 questions but attach\(\) was given 40"):
        inj.apply(other, 0.5, seed=0)


def test_apply_on_same_count_but_other_question_ids_is_an_error(monkeypatch):
    inj, state = attached(monkeypatch)
    reordered = dataclasses.replace(state, query_ids=state.query_ids[::-1])
    with pytest.raises(ValueError, match="query_ids don't match the state passed to attach") as err:
        inj.apply(reordered, 0.5, seed=0)
    assert "questions but" not in str(err.value)
