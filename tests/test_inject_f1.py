import dataclasses
import inspect

import numpy as np
import pytest

from silent_rag import pipeline
from silent_rag import run as runner
from silent_rag.inject.base import REGISTRY
from silent_rag.inject.f1_partial_upgrade import PartialUpgrade
from silent_rag.inject.model_check import ModelNotCachedError
from silent_rag.pipeline import Retriever
from silent_rag.run import load_plugins
from test_pipeline import HashEncoder, toy_dataset

ALT = {"alt_model": "alt/model", "alt_revision": "abc123"}


@pytest.fixture(autouse=True)
def fake_model_checker(monkeypatch):
    """check_config's cache lookup replaced by a recorder that accepts, so these tests stay offline.
    Set `.error` to make it raise."""
    def fake(*args):
        fake.calls.append(args)
        if fake.error is not None:
            raise fake.error

    fake.calls, fake.error = [], None
    monkeypatch.setattr(PartialUpgrade, "model_checker", staticmethod(fake))
    return fake


def test_registered_under_its_config_name():
    load_plugins()
    assert REGISTRY["F1_partial_upgrade"] is PartialUpgrade


@pytest.mark.parametrize("given", [{}, {"alt_model": ALT["alt_model"]}, {"alt_revision": ALT["alt_revision"]}])
def test_alt_model_and_revision_are_required(given):
    with pytest.raises(TypeError, match="missing a required argument"):
        inspect.signature(PartialUpgrade).bind(**given)
    inspect.signature(PartialUpgrade).bind(**ALT)


CLEAN = {"model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "revision": "b207367332321f8e44f96e224ef15bc607f4dbf0"}
PINNED = [
    {"alt_model": "sentence-transformers/all-MiniLM-L6-v2", "alt_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"},
    {"alt_model": "BAAI/bge-small-en-v1.5", "alt_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"},
]
HASH = PINNED[0]["alt_revision"]


@pytest.mark.parametrize("params", PINNED, ids=["minilm", "bge"])
def test_check_config_accepts_the_pinned_dev_models(params):
    assert PartialUpgrade.check_config(params, CLEAN) is None


@pytest.mark.parametrize("alt_model", ["", " ", None, 123])
def test_check_config_rejects_a_blank_or_non_string_alt_model(alt_model):
    with pytest.raises(ValueError, match="alt_model must be a non-empty model name"):
        PartialUpgrade.check_config({**PINNED[0], "alt_model": alt_model}, CLEAN)


@pytest.mark.parametrize("alt_revision", [None, "abc123", "main", HASH[:-1], HASH + "0", HASH + "\n", HASH.upper(),
                                          "g" + HASH[1:]],
                         ids=["none", "short", "branch", "39", "41", "newline", "upper", "not-hex"])
def test_check_config_rejects_a_revision_that_is_not_a_lowercase_commit_hash(alt_revision):
    with pytest.raises(ValueError, match="alt_revision must be a 40-character lowercase commit hash"):
        PartialUpgrade.check_config({**PINNED[0], "alt_revision": alt_revision}, CLEAN)


@pytest.mark.parametrize("alt_model, alt_revision, clean_revision", [
    (CLEAN["model"], CLEAN["revision"], CLEAN["revision"]),
    ("  Sentence-Transformers/Multi-QA-MiniLM-L6-cos-v1 ", CLEAN["revision"], CLEAN["revision"]),
    (CLEAN["model"], HASH, CLEAN["revision"]),
    (CLEAN["model"], HASH, None),
], ids=["exact", "case-and-spaces", "other-revision", "clean-unpinned"])
def test_check_config_rejects_the_clean_model_whatever_the_revision(alt_model, alt_revision, clean_revision):
    with pytest.raises(ValueError, match="is the clean model"):
        PartialUpgrade.check_config({"alt_model": alt_model, "alt_revision": alt_revision},
                                    {**CLEAN, "revision": clean_revision})


@pytest.mark.parametrize("alt_model", ["intfloat/e5-small-v2", "intfloat/multilingual-e5-base", " INTFLOAT/E5-small-v2 "],
                         ids=["e5", "multilingual-e5", "case-and-spaces"])
def test_check_config_rejects_models_that_need_a_document_prefix(alt_model, fake_model_checker):
    with pytest.raises(ValueError, match="needs a document prefix"):
        PartialUpgrade.check_config({"alt_model": alt_model, "alt_revision": HASH}, CLEAN)
    assert fake_model_checker.calls == []  # the name check rejects e5 before any cache lookup


@pytest.mark.parametrize("clean_revision", [CLEAN["revision"], None], ids=["pinned", "unpinned"])
def test_check_config_passes_the_replacement_and_the_runs_baseline_to_the_checker(fake_model_checker, clean_revision):
    PartialUpgrade.check_config(PINNED[1], {**CLEAN, "revision": clean_revision})
    assert fake_model_checker.calls == [(PINNED[1]["alt_model"], PINNED[1]["alt_revision"], CLEAN["model"], clean_revision)]


@pytest.mark.parametrize("error", [ValueError("no Normalize module"), ModelNotCachedError("README.md is not cached")],
                         ids=["rejected", "not-cached"])
def test_checker_errors_become_the_labelled_stop(fake_model_checker, error):
    load_plugins()
    fake_model_checker.error = error
    failures = {"F1_label": ("F1_partial_upgrade", [0.5], dict(PINNED[1]))}
    with pytest.raises(SystemExit, match=f"^F1_label: {error}$"):
        runner.check_injectors(failures, CLEAN)


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


CLEAN_SETTINGS = dict(normalize=False, query_prefix="query: ", max_seq_length=128, revision="cleanrev", batch_size=8)


def test_build_replacement_retriever_passes_the_same_settings_as_attach(monkeypatch):
    seen = []

    class Recording(Retriever):
        def __init__(self, model, **kw):
            seen.append((model, kw))
            super().__init__(model, **kw)

    monkeypatch.setattr(PartialUpgrade, "retriever_cls", Recording)
    clean = Retriever("clean/model", encoder=HashEncoder(), **CLEAN_SETTINGS)
    PartialUpgrade(**ALT).build_replacement_model_retriever(clean)
    # the same constructor call the settings test above sees through attach()
    assert seen == [(ALT["alt_model"], {**CLEAN_SETTINGS, "revision": ALT["alt_revision"], "device": "cpu"})]


@pytest.mark.parametrize("query_prefix, expected", [
    (None, "query: "),                      # inherits the baseline's prefix
    ("Represent this: ", "Represent this: "),  # a string replaces it
    ("", ""),                               # "" clears the inherited prefix
])
def test_build_replacement_retriever_query_prefix(query_prefix, expected):
    clean = Retriever("clean/model", encoder=HashEncoder(), **CLEAN_SETTINGS)
    alt = PartialUpgrade(**ALT).build_replacement_model_retriever(clean, query_prefix=query_prefix)
    assert alt.query_prefix == expected
    assert (alt.model, alt.revision) == (ALT["alt_model"], ALT["alt_revision"])
    assert (alt.normalize, alt.max_seq_length, alt.batch_size, alt.device) == (False, 128, 8, "cpu")


def test_build_replacement_retriever_does_not_copy_the_encoder():
    clean = Retriever("clean/model", encoder=HashEncoder(), **CLEAN_SETTINGS)
    alt = PartialUpgrade(**ALT).build_replacement_model_retriever(clean)
    assert clean._encoder is not None and alt._encoder is None  # loads the replacement model when first used


def test_attach_output_is_unchanged_by_the_refactor(monkeypatch):
    inj, state = attached(monkeypatch, ShiftedEncoder(), **CLEAN_SETTINGS)
    # attach() before this step built this retriever inline and encoded the corpus with it
    old = Retriever(ALT["alt_model"], encoder=ShiftedEncoder(), normalize=False, query_prefix="query: ",
                    max_seq_length=128, revision=ALT["alt_revision"], batch_size=8, device="cpu")
    np.testing.assert_array_equal(inj.alt_embeddings, old.encode(state.doc_texts))
    assert inj.attached_doc_ids == list(state.doc_ids)


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


@pytest.mark.parametrize("severity", [0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("seed", [0, 7])
def test_replaced_rows_are_the_rows_apply_changes(monkeypatch, severity, seed):
    inj, state = attached(monkeypatch, ShiftedEncoder())  # every alt vector differs, so changed rows are visible
    n = len(state.doc_ids)
    rows = PartialUpgrade.replaced_document_row_indices(n, severity, seed)
    np.testing.assert_array_equal(np.sort(rows), upgraded_rows(inj.apply(state, severity, seed), state))
    np.testing.assert_array_equal(inj.modified_document_row_indices(n, severity, seed), rows)


@pytest.mark.parametrize("severity, n, count", [(0.1, 40, 4), (0.5, 40, 20), (1.0, 40, 40), (0.25, 10, 2), (0.01, 40, 0)])
def test_replaced_row_count_is_round_severity_times_n(severity, n, count):
    # Python's round() rounds halves to even: 0.25 * 10 = 2.5 -> 2
    rows = PartialUpgrade.replaced_document_row_indices(n, severity, seed=0)
    assert len(rows) == count == round(severity * n)
    assert len(set(rows.tolist())) == count and ((0 <= rows) & (rows < n)).all()


def test_replaced_rows_are_nested_across_severities_within_a_seed():
    by_severity = [PartialUpgrade.replaced_document_row_indices(40, s, seed=3) for s in (0.1, 0.3, 0.5, 0.9, 1.0)]
    for lower, higher in zip(by_severity, by_severity[1:]):
        np.testing.assert_array_equal(higher[: len(lower)], lower)  # a prefix, so also a subset
    assert set(by_severity[-1].tolist()) == set(range(40))


@pytest.mark.parametrize("severity, seed", [(0.3, 0), (0.7, 5)])
def test_apply_output_is_unchanged_by_the_refactor(monkeypatch, severity, seed):
    inj, state = attached(monkeypatch, ShiftedEncoder())
    # apply() before this step selected rows inline with exactly this expression
    n = len(state.doc_ids)
    expected = state.doc_embeddings.copy()
    old_rows = np.random.default_rng(seed).permutation(n)[: round(severity * n)]
    expected[old_rows] = inj.alt_embeddings[old_rows]
    np.testing.assert_array_equal(inj.apply(state, severity, seed).doc_embeddings, expected)


def test_input_state_is_left_unchanged(monkeypatch):
    inj, state = attached(monkeypatch, ShiftedEncoder())
    before = state.doc_embeddings.copy()
    state.doc_embeddings.setflags(write=False)  # as the runner does
    inj.apply(state, 0.9, seed=0)
    np.testing.assert_array_equal(state.doc_embeddings, before)


def test_apply_passes_the_other_fields_through_as_the_same_objects(monkeypatch):
    inj, state = attached(monkeypatch, ShiftedEncoder())
    assert state.query_embeddings is not None
    damaged = inj.apply(state, 0.9, seed=0)
    for field in ("doc_ids", "doc_texts", "query_ids", "query_texts", "query_embeddings", "config"):
        assert getattr(damaged, field) is getattr(state, field), field


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
