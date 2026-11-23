import dataclasses

import numpy as np
import pytest

from silent_rag import pipeline
from silent_rag.inject.base import REGISTRY
from silent_rag.inject.f5_config_mismatch import IndexWithoutNormalize
from silent_rag.pipeline import Retriever
from silent_rag.run import load_plugins
from test_pipeline import HashEncoder, toy_dataset


def attached(monkeypatch, encoder, seen=None, **clean_settings):
    """An attached injector whose un-normalised encoder is `encoder` (HashEncoder honours normalize)."""

    class Alt(Retriever):
        def __init__(self, model, **kw):
            if seen is not None:
                seen.append((model, {k: v for k, v in kw.items() if k != "encoder"}, kw.get("encoder")))
            super().__init__(model, **kw)

    monkeypatch.setattr(IndexWithoutNormalize, "retriever_cls", Alt)
    monkeypatch.setattr(IndexWithoutNormalize, "without_normalize", staticmethod(lambda clean_encoder: encoder))
    clean = Retriever("clean/model", encoder=HashEncoder(), **clean_settings)
    state = clean.embed_corpus(toy_dataset())
    inj = IndexWithoutNormalize()
    inj.attach(clean, state)
    return inj, state


def test_registered_under_its_config_name():
    load_plugins()
    assert REGISTRY["F5_index_no_normalize"] is IndexWithoutNormalize
    assert IndexWithoutNormalize.harmful


def test_attach_keeps_the_clean_settings_except_normalize_and_the_encoder(monkeypatch):
    seen, encoder = [], HashEncoder()
    clean = dict(query_prefix="query: ", max_seq_length=128, revision="cleanrev", batch_size=8)
    inj, state = attached(monkeypatch, encoder, seen, **clean)
    assert seen == [("clean/model", {**clean, "normalize": False, "device": "cpu"}, encoder)]
    assert inj.alt_embeddings.shape == state.doc_embeddings.shape


def test_vectors_are_encoded_once_then_loaded_from_their_own_cache_file(monkeypatch):
    first, second = HashEncoder(), HashEncoder()
    a, _ = attached(monkeypatch, first)
    b, _ = attached(monkeypatch, second)
    assert (first.calls, second.calls) == (1, 0)
    np.testing.assert_array_equal(a.alt_embeddings, b.alt_embeddings)
    assert len(list(pipeline.CACHE_DIR.glob("f5_nonorm-*.npy"))) == 1


def changed_rows(damaged, state):
    return np.flatnonzero((damaged.doc_embeddings != state.doc_embeddings).any(axis=1))


@pytest.mark.parametrize("severity, count", [(0.1, 4), (0.5, 20), (0.9, 36), (1.0, 40)])
def test_severity_fraction_of_rows_lose_normalisation_and_keep_their_direction(monkeypatch, severity, count):
    inj, state = attached(monkeypatch, HashEncoder())
    damaged = inj.apply(state, severity, seed=0)
    rows = changed_rows(damaged, state)
    assert len(rows) == count
    np.testing.assert_array_equal(damaged.doc_embeddings[rows], inj.alt_embeddings[rows])
    norms = np.linalg.norm(damaged.doc_embeddings[rows], axis=1)
    assert not np.isclose(norms, 1.0).any()
    # only the length changes: normalised again, every row is the clean index's row for that document
    np.testing.assert_allclose(damaged.doc_embeddings[rows] / norms[:, None], state.doc_embeddings[rows], atol=1e-6)


def test_baseline_cache_is_left_untouched(monkeypatch):
    monkeypatch.setattr(IndexWithoutNormalize, "without_normalize", staticmethod(lambda clean_encoder: HashEncoder()))
    clean = Retriever("clean/model", encoder=HashEncoder())
    state = clean.embed_corpus(toy_dataset())
    before = {p.name: p.read_bytes() for p in pipeline.CACHE_DIR.glob("*.npy")}
    IndexWithoutNormalize().attach(clean, state)
    after = {p.name: p.read_bytes() for p in pipeline.CACHE_DIR.glob("*.npy")}
    assert len(before) == 1 and all(after[name] == data for name, data in before.items())
    new = set(after) - set(before)
    assert len(new) == 1 and new.pop().startswith("f5_nonorm-")


def test_rows_are_nested_across_severities_and_follow_the_seed(monkeypatch):
    inj, state = attached(monkeypatch, HashEncoder())
    low, high = (set(changed_rows(inj.apply(state, s, seed=0), state)) for s in (0.2, 0.6))
    assert low < high
    other_seed = set(changed_rows(inj.apply(state, 0.6, seed=1), state))
    assert other_seed != high and len(other_seed) == len(high)
    np.testing.assert_array_equal(inj.apply(state, 0.6, seed=0).doc_embeddings,
                                  inj.apply(state, 0.6, seed=0).doc_embeddings)


def test_input_is_unchanged_and_other_fields_pass_through(monkeypatch):
    inj, state = attached(monkeypatch, HashEncoder())
    before = state.doc_embeddings.copy()
    damaged = inj.apply(state, 0.9, seed=0)
    np.testing.assert_array_equal(state.doc_embeddings, before)
    for field in ("doc_ids", "doc_texts", "query_ids", "query_texts", "query_embeddings", "parent_ids", "config"):
        assert getattr(damaged, field) is getattr(state, field), field


class AlwaysNormalised(HashEncoder):
    """An encoder whose vectors are unit length whatever it is asked: dropping Normalize changes nothing."""

    def encode(self, texts, **kw):
        return super().encode(texts, **{**kw, "normalize_embeddings": True})


def test_a_setup_that_is_already_unnormalised_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="unit length without Normalize too, so nothing would change"):
        attached(monkeypatch, AlwaysNormalised(), normalize=False)


def test_without_normalize_drops_the_normalize_module_and_shares_the_rest():
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Normalize, Pooling

    pooling = Pooling(4)
    dropped = IndexWithoutNormalize.without_normalize(SentenceTransformer(modules=[pooling, Normalize()], device="cpu"))
    assert list(dropped) == [pooling]
    assert list(dropped)[0] is pooling


def test_apply_before_attach_is_an_error():
    state = Retriever("clean/model", encoder=HashEncoder()).embed_corpus(toy_dataset())
    with pytest.raises(RuntimeError, match=r"call attach\(\) before apply\(\)"):
        IndexWithoutNormalize().apply(state, 0.5, seed=0)


def test_apply_on_another_state_than_attach_is_an_error(monkeypatch):
    inj, _ = attached(monkeypatch, HashEncoder())
    other = Retriever("clean/model", encoder=HashEncoder()).embed_corpus(toy_dataset(30))
    with pytest.raises(ValueError, match="state has 30 rows but attach\\(\\) was given 40"):
        inj.apply(other, 0.5, seed=0)


def test_apply_on_same_count_but_other_doc_ids_is_an_error(monkeypatch):
    inj, state = attached(monkeypatch, HashEncoder())
    reordered = dataclasses.replace(state, doc_ids=state.doc_ids[::-1])
    with pytest.raises(ValueError, match="doc_ids don't match the state passed to attach") as err:
        inj.apply(reordered, 0.5, seed=0)
    assert "rows" not in str(err.value)
