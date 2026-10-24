import numpy as np
import pytest

from silent_rag.detect.base import REGISTRY
from silent_rag.detect.domain_classifier import (
    QueryDomainClassifier,
    RetrievedDocDomainClassifier,
)
from silent_rag.types import Observation


def observation(queries, query_ids, docs=None, doc_ids=None, hits=None):
    queries = np.asarray(queries, dtype=np.float32)
    if docs is None:
        docs = queries
        doc_ids = query_ids
    if hits is None:
        hits = [[d] for d in doc_ids]
    return Observation(
        query_ids=list(query_ids),
        query_embeddings=queries,
        topk_ids=hits,
        topk_scores=np.ones((len(hits), 1), dtype=np.float32),
        doc_ids=list(doc_ids),
        doc_embeddings=np.asarray(docs, dtype=np.float32),
    )


def test_registered():
    assert REGISTRY["domain_classifier_query"] is QueryDomainClassifier
    assert REGISTRY["domain_classifier_retrieved_docs"] is RetrievedDocDomainClassifier


def test_query_detects_shift():
    rng = np.random.default_rng(2)
    ref = observation(rng.normal(size=(60, 8)), [f"ref{i}" for i in range(60)])
    cur = observation(rng.normal(size=(60, 8)) + 2.5, [f"cur{i}" for i in range(60)])
    detector = QueryDomainClassifier()
    detector.fit(ref)
    assert detector.score(cur) > 0.9


def test_query_clean_noise_near_chance():
    rng = np.random.default_rng(42)
    ref = observation(rng.normal(size=(120, 8)), [f"r{i}" for i in range(120)])
    cur = observation(rng.normal(size=(120, 8)), [f"c{i}" for i in range(120)])
    detector = QueryDomainClassifier()
    detector.fit(ref)
    assert 0.25 < detector.score(cur) < 0.75


def test_doc_classifier_detects_shift_and_deduplicates_hits():
    rng = np.random.default_rng(0)
    ref_ids = [f"r{i}" for i in range(50)]
    cur_ids = [f"c{i}" for i in range(50)]
    ref_x = rng.normal(size=(50, 8))
    cur_x = rng.normal(size=(50, 8)) + 2.5
    ref = observation(ref_x[:5], [f"q{i}" for i in range(5)], ref_x, ref_ids, [[d, d] for d in ref_ids])
    cur = observation(cur_x[:5], [f"p{i}" for i in range(5)], cur_x, cur_ids, [[d, d] for d in cur_ids])
    detector = RetrievedDocDomainClassifier()
    detector.fit(ref)
    assert detector.score(cur) > 0.9


def test_identical_docs_in_both_domains_stay_at_chance():
    rng = np.random.default_rng(33)
    ids = [f"d{i}" for i in range(40)]
    vectors = rng.normal(size=(40, 10))
    obs = observation(vectors[:5], [f"q{i}" for i in range(5)], vectors, ids)
    detector = RetrievedDocDomainClassifier()
    detector.fit(obs)
    assert detector.score(obs) == pytest.approx(0.5)


def test_fit_required():
    obs = observation(np.ones((8, 4)), [f"q{i}" for i in range(8)])
    with pytest.raises(RuntimeError, match="fit"):
        QueryDomainClassifier().score(obs)


def test_embedding_shape_rejected():
    ref = observation(np.ones((10, 4)), [f"r{i}" for i in range(10)])
    cur = observation(np.ones((10, 3)), [f"c{i}" for i in range(10)])
    det = QueryDomainClassifier()
    det.fit(ref)
    with pytest.raises(ValueError, match="dimensions"):
        det.score(cur)


def test_not_enough_samples():
    ref = observation(np.ones((1, 4)), ["r"])
    cur = observation(np.ones((1, 4)), ["c"])
    det = QueryDomainClassifier()
    det.fit(ref)
    with pytest.raises(ValueError, match="at least two"):
        det.score(cur)


def test_nonfinite_rejected():
    arr = np.ones((8, 4))
    arr[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        QueryDomainClassifier().fit(observation(arr, [f"r{i}" for i in range(8)]))


def test_duplicate_identity_in_window_rejected():
    with pytest.raises(ValueError, match="duplicate identities"):
        QueryDomainClassifier().fit(observation(np.ones((3, 4)), ["a", "a", "b"]))


def test_missing_retrieved_doc_mapping_rejected():
    ref = observation(np.ones((5, 4)), [f"q{i}" for i in range(5)], np.ones((5, 4)), [f"d{i}#0" for i in range(5)], [[f"d{i}"] for i in range(5)])
    with pytest.raises(ValueError, match="parent_ids"):
        RetrievedDocDomainClassifier().fit(ref)


def test_query_groups_dont_split_repeated_ids_across_folds():
    # A balanced pair of each identity in both windows must not be memorized.
    rng = np.random.default_rng(1)
    ids = [f"q{i}" for i in range(40)]
    vectors = rng.normal(size=(40, 20))
    obs = observation(vectors, ids)
    det = QueryDomainClassifier()
    det.fit(obs)
    assert det.score(obs) == pytest.approx(0.5)


def test_chunked_docs_aggregate_by_parent_id():
    ref = observation(
        np.ones((3, 2)), ["q1", "q2", "q3"],
        docs=[[1, 0], [3, 0], [0, 3], [0, 1], [2, 2], [3, 3]],
        doc_ids=["a#0", "a#1", "b#0", "b#1", "c#0", "c#1"],
        hits=[["a", "b", "c"]],
    )
    ref.parent_ids = ["a", "a", "b", "b", "c", "c"]
    from silent_rag.detect.domain_classifier import _doc_rows

    vectors, ids = _doc_rows(ref)
    assert ids == ["a", "b", "c"]
    np.testing.assert_allclose(vectors, [[2, 0], [0, 2], [2.5, 2.5]])


def test_runner_discovers_both_detector_names():
    from silent_rag.run import load_plugins

    load_plugins()
    assert "domain_classifier_query" in REGISTRY
    assert "domain_classifier_retrieved_docs" in REGISTRY

def test_runner_integration_both_domain_classifiers(tmp_path, monkeypatch):
    from silent_rag import pipeline, run as runner
    from silent_rag.inject.base import REGISTRY as INJECTORS
    from test_pipeline import HashEncoder, ScrambleDocs, toy_dataset

    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)

    cfg = {
        "datasets": ["nfcorpus"],
        "model": "fake",
        "failures": {ScrambleDocs.name: [0.5]},
        "detectors": [
            "domain_classifier_query",
            "domain_classifier_retrieved_docs",
        ],
        "window_size": 10,
        "trials": 3,
        "seed": 0,
    }

    df = runner.run(
        cfg,
        loader=lambda name: toy_dataset(),
        make_retriever=lambda: pipeline.Retriever(
            "fake", encoder=HashEncoder()
        ),
    )

    assert len(df) == 12
    assert set(df["detector"]) == set(cfg["detectors"])
    assert set(df["failure"]) == {"clean", ScrambleDocs.name}
    assert np.isfinite(df["score"]).all()
    assert df["score"].between(0, 1).all()