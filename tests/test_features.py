import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.detect.features import FEATURES, score_features
from silent_rag.pipeline import Retriever
from silent_rag.types import Observation
from test_pipeline import HashEncoder, toy_dataset


def obs(scores, ids=None):
    scores = np.asarray(scores, dtype=np.float32)
    ids = ids or [f"q{i}" for i in range(len(scores))]
    return Observation(ids, np.zeros((len(scores), 4)), [[]] * len(scores), scores)


def test_features_by_hand():
    f = score_features(obs([np.linspace(0.9, 0.0, 10), np.full(10, 0.5)]))
    assert list(f.columns) == FEATURES
    assert list(f.index) == ["q0", "q1"]
    np.testing.assert_allclose(f.loc["q0"], [0.9, 0.1, 0.45, np.std(np.linspace(0.9, 0.0, 10))], atol=1e-6)
    np.testing.assert_allclose(f.loc["q1"], [0.5, 0.0, 0.5, 0.0], atol=1e-6)


def test_deeper_scores_dont_change_the_features():
    rows = np.sort(np.random.default_rng(0).random((5, 100)), axis=1)[:, ::-1]
    np.testing.assert_array_equal(score_features(obs(rows)), score_features(obs(rows[:, :10])))


def test_bad_scores_are_errors():
    with pytest.raises(ValueError, match="top 10"):
        score_features(obs(np.ones((3, 5))))
    with pytest.raises(ValueError, match="sorted"):
        score_features(obs([np.linspace(0, 1, 10)]))
    with pytest.raises(ValueError, match="non-finite"):
        score_features(obs([[np.nan] * 10]))
    with pytest.raises(ValueError, match="expected"):
        score_features(obs(np.ones((3, 10)), ids=["q0", "q1"]))


def test_works_on_what_the_runner_logs(tmp_path):
    r = Retriever("fake", encoder=HashEncoder())
    state = r.embed_corpus(toy_dataset(40), cache_dir=tmp_path)
    idx = np.arange(15)
    ids, scores, q = r.search(state, [state.query_texts[i] for i in idx], k=10)
    f = score_features(runner.observe(state, idx, ids, scores, q))
    assert list(f.index) == state.query_ids[:15]
    assert (f["gap"] >= 0).all() and (f["std10"] >= 0).all()
    assert (f["top1"] >= f["mean10"]).all()
