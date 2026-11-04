import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY
from silent_rag.detect.qpp import NQC, GapQPP, MaxScore
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.pipeline import Retriever
from silent_rag.types import Observation
from test_ks import NoisyDocs
from test_pipeline import HashEncoder, ScrambleDocs, toy_dataset


def obs(scores, deep=None):
    scores = np.asarray(scores, dtype=np.float32)
    ids = [f"q{i}" for i in range(len(scores))]
    return Observation(ids, np.zeros((len(scores), 4)), [[]] * len(scores), scores, deep_scores=deep)


def ranked(top1, step, n=10):
    """Rows that start at top1 and fall by `step` per rank."""
    return np.asarray(top1, dtype=float)[:, None] - np.arange(n) * np.asarray(step, dtype=float)[:, None]


def test_registered():
    assert REGISTRY["qpp_max"] is MaxScore and REGISTRY["qpp_gap"] is GapQPP and REGISTRY["qpp_nqc"] is NQC


def test_score_is_the_drop_in_the_window_mean():
    det = MaxScore()
    det.fit(obs(ranked([0.6, 0.8], [0.01, 0.01])))  # reference mean top1 0.7
    assert det.score(obs(ranked([0.5, 0.6], [0.01, 0.01]))) == pytest.approx(0.15, abs=1e-6)
    # predicted quality going up is negative, not an alarm
    assert det.score(obs(ranked([0.9, 0.9], [0.01, 0.01]))) == pytest.approx(-0.2, abs=1e-6)


def test_gap_and_nqc_by_hand():
    reference = obs(ranked([0.7, 0.7], [0.02, 0.02]))
    flatter = obs(ranked([0.7, 0.7], [0.01, 0.01]))
    gap, nqc = GapQPP(), NQC()
    for det in (gap, nqc):
        det.fit(reference)
    assert gap.score(flatter) == pytest.approx(0.01, abs=1e-6)
    # std of 0..9 times the step, so halving the step halves NQC
    assert nqc.score(flatter) == pytest.approx(0.01 * np.arange(10).std(), abs=1e-6)


def test_deep_nqc_reads_deep_scores():
    top10 = ranked([0.7, 0.7], [0.0, 0.0])  # flat top 10, std10 = 0
    deep = ranked([0.7, 0.7], [0.0, 0.01], n=100)
    det = NQC(depth=100)
    det.fit(obs(top10, deep=deep))
    np.testing.assert_allclose(det.reference_, [0.0, 0.01 * np.arange(100).std()], atol=1e-6)
    det50 = NQC(depth=50)
    det50.fit(obs(top10, deep=deep))
    np.testing.assert_allclose(det50.reference_, [0.0, 0.01 * np.arange(50).std()], atol=1e-6)


def test_deep_nqc_without_score_depth_says_so():
    det = NQC(depth=100)
    with pytest.raises(ValueError, match="score_depth >= 100"):
        det.fit(obs(ranked([0.7], [0.01])))
    with pytest.raises(ValueError, match="score_depth >= 100"):
        det.fit(obs(ranked([0.7], [0.01]), deep=ranked([0.7], [0.01], n=50)))


@pytest.mark.parametrize("depth", [5, 2.5, True, "100"])
def test_bad_depth_is_rejected(depth):
    with pytest.raises(ValueError):
        NQC(depth=depth)


def test_score_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="fit"):
        MaxScore().score(obs(ranked([0.7], [0.01])))


def run_toy(monkeypatch, failure, detectors, **extra):
    for inj in (ScrambleDocs, NoisyDocs):
        monkeypatch.setitem(INJECTORS, inj.name, inj)
    cfg = {"datasets": ["nfcorpus"], "failures": {failure: [0.5]}, "detectors": detectors,
           "window_size": 15, "trials": 4, "seed": 0, **extra}
    df = runner.run(
        cfg, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("fake", encoder=HashEncoder())
    )
    assert np.isfinite(df.score).all()
    return df.groupby(["detector", "failure"]).score.mean().unstack()


def test_runs_through_the_runner(monkeypatch):
    scores = run_toy(monkeypatch, NoisyDocs.name, ["qpp_max", "qpp_gap", "qpp_nqc"], score_depth=25)
    # random doc vectors make the best match a worse match
    assert scores.loc["qpp_max", NoisyDocs.name] > scores.loc["qpp_max", "clean"]
    scrambled = run_toy(monkeypatch, ScrambleDocs.name, ["qpp_max", "qpp_gap", "qpp_nqc"])
    assert (scrambled[ScrambleDocs.name] == scrambled["clean"]).all()


def test_deep_nqc_through_the_runner(monkeypatch):
    scores = run_toy(monkeypatch, NoisyDocs.name, [{"qpp_nqc": {"depth": 25}}], score_depth=25)
    assert scores.notna().all().all()
    with pytest.raises(ValueError, match="score_depth"):
        run_toy(monkeypatch, NoisyDocs.name, [{"qpp_nqc": {"depth": 25}}])


def test_clean_windows_center_on_zero(monkeypatch):
    # no damage: window means scatter around the reference mean, so scores do too
    scores = run_toy(monkeypatch, NoisyDocs.name, ["qpp_max", "qpp_gap", "qpp_nqc"])
    assert (scores["clean"].abs() < 0.1).all()
