import time

import numpy as np

from silent_rag.detect.base import REGISTRY
from silent_rag.detect.mmd import MMDQuery, MMDQueryPerm, MMDRetrievedDocs, MMDRetrievedDocsPerm, mmd2
from silent_rag.types import Observation


def observation(queries, docs=None, hits=None, prefix="q"):
    queries = np.asarray(queries, dtype=np.float32)
    if docs is None:
        docs = queries
    doc_ids = [f"{prefix}d{i}" for i in range(len(docs))]
    if hits is None:
        hits = [[d] for d in doc_ids[: len(queries)]]
    return Observation(
        query_ids=[f"{prefix}{i}" for i in range(len(queries))],
        query_embeddings=queries,
        topk_ids=hits,
        topk_scores=np.ones((len(hits), 1), dtype=np.float32),
        doc_ids=doc_ids,
        doc_embeddings=np.asarray(docs, dtype=np.float32),
    )


def samples(shift=0.0, n=100, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)), rng.normal(size=(n, dim)) + shift


def test_same_distribution_near_zero_and_shift_clearly_larger():
    same, shifted = mmd2(*samples()), mmd2(*samples(shift=1.0))
    assert abs(same) < 0.01
    assert shifted > 0.1 and shifted > 10 * abs(same)


def test_query_fit_and_score():
    x, y = samples(shift=1.0, n=60, dim=8)
    _, same = samples(n=60, dim=8, seed=3)
    detector = MMDQuery()
    detector.fit(observation(x, prefix="r"))
    assert detector.score(observation(y, prefix="c")) > 10 * abs(detector.score(observation(same, prefix="c")))


def test_retrieved_docs_row_is_mean_of_stored_vectors_and_sees_doc_shift():
    docs = np.arange(12, dtype=float).reshape(4, 3)
    obs = observation(np.zeros((2, 3)), docs, hits=[["qd0", "qd1"], ["qd1", "qd2", "qd3"]])
    assert np.allclose(MMDRetrievedDocs().values(obs), [docs[[0, 1]].mean(0), docs[[1, 2, 3]].mean(0)])
    # F1-like: same questions, the stored doc vectors move
    rng = np.random.default_rng(4)
    queries, docs = rng.normal(size=(60, 8)), rng.normal(size=(120, 8))
    hits = [[f"qd{2 * i}", f"qd{2 * i + 1}"] for i in range(60)]
    detector = MMDRetrievedDocs()
    detector.fit(observation(queries, docs, hits))
    assert detector.score(observation(queries, docs + 1.5, hits)) > 0.1

def test_permutation_p_small_for_shift_not_for_no_shift():
    x, y = samples(shift=1.0, n=60, dim=8)
    _, same = samples(n=60, dim=8, seed=5)
    detector = MMDQueryPerm(permutations=199)
    detector.fit(observation(x, prefix="r"))
    assert detector.p_value(observation(y, prefix="c")) <= 0.01
    assert detector.p_value(observation(same, prefix="c")) > 0.05


def test_registered_names_resolve():
    assert REGISTRY["mmd_query"] is MMDQuery
    assert REGISTRY["mmd_retrieved_docs"] is MMDRetrievedDocs
    assert REGISTRY["mmd_query_perm"] is MMDQueryPerm
    assert REGISTRY["mmd_retrieved_docs_perm"] is MMDRetrievedDocsPerm


def test_one_window_scores_fast():
    x, y = samples(shift=0.1, dim=384)
    detector = MMDQuery()
    detector.fit(observation(x, prefix="r"))
    window = observation(y, prefix="c")
    times = []
    for _ in range(5):
        start = time.perf_counter()
        detector.score(window)
        times.append(time.perf_counter() - start)
    assert np.median(times) < 0.1
