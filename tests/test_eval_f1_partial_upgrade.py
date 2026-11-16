import numpy as np
import pytest

from silent_rag.eval.f1_partial_upgrade import (
    mean_cosine_baseline_vs_replacement_document_vectors as mean_cosine,
    topk_documents_shared_per_question as shared,
)

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
