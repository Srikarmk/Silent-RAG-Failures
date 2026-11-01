import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.inject.base import REGISTRY as INJECTORS
from test_pipeline import MeanTop1, ScrambleDocs, toy_dataset
from test_seeds_and_chunks import ChunkHalf, fake_retriever


def spy_run(monkeypatch, injector=ScrambleDocs, **extra):
    """Run the toy benchmark and keep every Observation the detector sees (reference first)."""
    seen = []

    class Spy(MeanTop1):
        name = "test_depth_spy"

        def fit(self, reference):
            seen.append(reference)
            super().fit(reference)

        def score(self, window):
            seen.append(window)
            return super().score(window)

    monkeypatch.setitem(INJECTORS, injector.name, injector)
    monkeypatch.setitem(DETECTORS, Spy.name, Spy)
    cfg = {"datasets": ["nfcorpus"], "failures": {injector.name: [0.5]}, "detectors": [Spy.name],
           "window_size": 10, "trials": 2, **extra}
    df = runner.run(cfg, loader=lambda name: toy_dataset(), make_retriever=fake_retriever)
    return df, seen


def test_deep_scores_extend_the_top_k(monkeypatch):
    _, seen = spy_run(monkeypatch, score_depth=25)
    assert len(seen) == 1 + 2 * 2  # reference, then clean and damaged windows
    for obs in seen:
        assert obs.deep_scores.shape == (10, 25)
        assert (np.diff(obs.deep_scores, axis=1) <= 1e-6).all()
        np.testing.assert_allclose(obs.deep_scores[:, :10], obs.topk_scores, atol=1e-6)
        assert not obs.deep_scores.flags.writeable


def test_without_score_depth_nothing_changes(monkeypatch):
    plain, seen = spy_run(monkeypatch)
    assert all(obs.deep_scores is None for obs in seen)
    deep, _ = spy_run(monkeypatch, score_depth=25)
    cols = ["failure", "severity", "trial", "score", "ndcg_drop", "p_value", "harmful"]
    # only run_id differs, since score_depth is part of the config
    assert plain[cols].equals(deep[cols])


def test_chunked_deep_scores_are_per_parent_doc(monkeypatch):
    _, seen = spy_run(monkeypatch, injector=ChunkHalf, score_depth=25)
    chunked = seen[-1]
    assert chunked.parent_ids is not None and chunked.deep_scores.shape == (10, 25)
    np.testing.assert_allclose(chunked.deep_scores[:, :10], chunked.topk_scores, atol=1e-6)


@pytest.mark.parametrize("depth", [5, "100", True, 2.5])
def test_bad_score_depth_stops_before_anything_runs(monkeypatch, depth):
    with pytest.raises(SystemExit, match="score_depth must be an integer >= k"):
        spy_run(monkeypatch, score_depth=depth)


def test_score_depth_deeper_than_the_index(monkeypatch):
    n_docs = len(toy_dataset().corpus)
    with pytest.raises(SystemExit, match="deeper than the index"):
        spy_run(monkeypatch, score_depth=n_docs + 1)
