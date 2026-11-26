import numpy as np
import pytest

from silent_rag.eval.f1_partial_upgrade import (
    mean_cosine_baseline_vs_replacement_document_vectors as mean_cosine,
    replacement_model_difference_table,
    topk_documents_shared_per_question as shared,
)
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.inject.f1_partial_upgrade import PartialUpgrade
from silent_rag.pipeline import Retriever
from test_pipeline import HashEncoder, ScrambleDocs, toy_dataset

E = np.eye(3)


def test_identical_vectors_give_cosine_1():
    assert mean_cosine(E, E) == pytest.approx(1.0)


def test_orthogonal_vectors_give_cosine_0():
    assert mean_cosine(E, np.roll(E, 1, axis=1)) == pytest.approx(0.0)


def test_opposite_vectors_give_cosine_minus_1():
    assert mean_cosine(E, -E) == pytest.approx(-1.0)


def test_vectors_that_are_not_unit_length_give_the_normalised_cosine():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(20, 8)), rng.normal(size=(20, 8))
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    scaled = a * rng.uniform(0.1, 10, size=(20, 1))  # each row stretched differently
    assert mean_cosine(scaled, b) == pytest.approx(mean_cosine(unit(a), unit(b)))
    assert mean_cosine(unit(a), unit(b)) == pytest.approx(float((unit(a) * unit(b)).sum(axis=1).mean()))


@pytest.mark.parametrize("b", [np.eye(3)[:2], np.ones(3)], ids=["fewer-rows", "1-d"])
def test_arrays_of_different_shape_are_an_error(b):
    with pytest.raises(ValueError, match="same shape"):
        mean_cosine(E, b)


def test_shared_counts_identical_disjoint_and_partial_lists():
    base = [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]
    switched = [["a", "b", "c"], ["x", "y", "z"], ["a", "x", "c"]]
    got = shared(base, switched)
    assert got.tolist() == [3, 0, 2] and got.dtype.kind == "i"


def test_shared_ignores_order_within_a_list():
    assert shared([["a", "b", "c"]], [["c", "a", "b"]]).tolist() == [3]


def test_lists_of_different_length_are_an_error():
    with pytest.raises(ValueError, match="2 baseline lists but 1 switched lists"):
        shared([["a"], ["b"]], [["a"]])


HASH_A, HASH_B = "a" * 40, "b" * 40
ANALYSIS = {"question_bootstrap": {"resamples": 200, "seed": 0},
            "injector_specific": {"F1_partial_upgrade": {"replacement_model_difference_table": {
                "query_prefixes": {"listed/model": "query: "}}}}}


@pytest.fixture
def replacement_retrievers(monkeypatch):
    """Replacement retrievers use the baseline's fake encoder (an identical model); records each query_prefix."""
    seen = []

    class Same(Retriever):
        def __init__(self, model, **kw):
            seen.append((model, kw["query_prefix"]))
            super().__init__(model, encoder=HashEncoder(), **kw)

    monkeypatch.setattr(PartialUpgrade, "retriever_cls", Same)
    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)
    return seen


def table(failures, baseline_prefix=""):
    run_config = {"datasets": ["nfcorpus"], "model": "clean/model", "failures": failures}
    baseline = lambda: Retriever("clean/model", encoder=HashEncoder(), query_prefix=baseline_prefix)  # noqa: E731
    return replacement_model_difference_table(run_config, ANALYSIS, loader=lambda name: toy_dataset(),
                                              make_retriever=baseline)


F1_ENTRY = {"injector": "F1_partial_upgrade", "severities": [0.5]}


def test_identical_replacement_matches_the_baseline(replacement_retrievers):
    row = table({"F1_same": {**F1_ENTRY, "alt_model": "listed/model", "alt_revision": HASH_A}}).iloc[0]
    assert row["mean_cosine"] == pytest.approx(1.0)
    assert (row["mean_topk_shared"], row["topk_shared_ci_low"], row["topk_shared_ci_high"]) == (10, 10, 10)
    assert row["switched_ndcg10"] == row["baseline_ndcg10"] and row["k"] == 10
    assert (row["replacement_model"], row["replacement_revision"]) == ("listed/model", HASH_A)


def test_one_row_per_f1_label_prefixes_listed_or_empty_and_other_labels_ignored(replacement_retrievers):
    got = table({
        "F1_listed": {**F1_ENTRY, "alt_model": "listed/model", "alt_revision": HASH_A},
        "F1_unlisted": {**F1_ENTRY, "alt_model": "unlisted/model", "alt_revision": HASH_B},
        ScrambleDocs.name: [0.5],
    }, baseline_prefix="base: ")
    assert got["label"].tolist() == ["F1_listed", "F1_unlisted"]
    # attach() inherits the baseline's "base: "; the switched retriever gets the listed prefix, or "" (not "base: ")
    assert [p for m, p in replacement_retrievers if m == "listed/model"] == ["base: ", "query: "]
    assert [p for m, p in replacement_retrievers if m == "unlisted/model"] == ["base: ", ""]


def test_held_out_datasets_need_final():
    with pytest.raises(SystemExit, match="held out"):
        replacement_model_difference_table({"datasets": ["scifact"], "failures": {}}, ANALYSIS)
