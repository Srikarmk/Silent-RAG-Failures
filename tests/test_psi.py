import importlib

import numpy as np
import pytest

from silent_rag.detect import ks, psi
from silent_rag.detect.base import REGISTRY
from silent_rag.detect.features import FEATURES
from silent_rag.detect.psi import PSI_DETECTORS
from silent_rag.types import Observation
from test_ks import NoisyDocs, ScrambleDocs, run_toy

PSITop1 = PSI_DETECTORS["top1"]


def obs(top1, gap=None, seed=0):
    """Windows with the given top-1 scores and top-1 minus top-2 gaps."""
    top1 = np.asarray(top1, dtype=float)
    gap = np.full(len(top1), 0.01) if gap is None else np.asarray(gap, dtype=float)
    rest = (top1 - gap)[:, None] - np.sort(np.random.default_rng(seed).random((len(top1), 8)), axis=1) * 0.05
    scores = np.hstack([top1[:, None], (top1 - gap)[:, None], rest]).astype(np.float32)
    return Observation([f"q{i}" for i in range(len(top1))], np.zeros((len(top1), 4)), [[]] * len(top1), scores)


def test_one_detector_per_feature_for_ks_and_psi():
    for prefix in ("ks", "psi"):
        for feature in FEATURES:
            assert REGISTRY[f"{prefix}_{feature}"].feature == feature


def test_psi_by_hand():
    det = PSITop1(bins=2)
    det.fit(obs([0.1, 0.2, 0.3, 0.4]))  # one edge at the median, 0.25
    # 3 of 4 above the edge instead of 2: (0.25 - 0.5) ln(0.5) + (0.75 - 0.5) ln(1.5)
    assert det.score(obs([0.1, 0.3, 0.35, 0.4])) == pytest.approx(0.25 * np.log(3), rel=1e-6)
    assert det.score(obs([0.1, 0.2, 0.3, 0.4])) == 0.0


def test_psi_same_distribution_low_shift_high():
    rng = np.random.default_rng(0)
    det = PSITop1()
    det.fit(obs(rng.normal(0.6, 0.05, 1000)))
    assert det.score(obs(rng.normal(0.6, 0.05, 1000), seed=1)) < 0.05
    assert det.score(obs(rng.normal(0.55, 0.05, 1000), seed=2)) > 0.25


def test_psi_bins_come_from_the_reference_and_values_outside_it_count():
    det = PSITop1()
    det.fit(obs(np.linspace(0.5, 0.7, 100)))
    edges = det.edges_.copy()
    # everything above the reference range goes to the last bin, not nowhere
    above = det.score(obs(np.full(100, 0.9)))
    np.testing.assert_array_equal(det.edges_, edges)
    assert above == pytest.approx(det.score(obs(np.full(100, 0.71))))
    assert above > 2


def test_psi_with_ties_merges_bins():
    det = PSITop1()
    det.fit(obs(np.r_[np.full(80, 0.5), np.linspace(0.6, 0.7, 20)]))
    assert len(det.edges_) < 9
    assert det.score(obs(np.r_[np.full(80, 0.5), np.linspace(0.6, 0.7, 20)])) == 0.0


def test_each_detector_reads_its_own_feature():
    rng = np.random.default_rng(0)
    top1 = rng.normal(0.6, 0.05, 200)
    reference, wider_gaps = obs(top1, gap=rng.uniform(0, 0.02, 200)), obs(top1, gap=rng.uniform(0.03, 0.05, 200))
    for prefix in ("ks", "psi"):
        on_top1, on_gap = REGISTRY[f"{prefix}_top1"](), REGISTRY[f"{prefix}_gap"]()
        for det in (on_top1, on_gap):
            det.fit(reference)
        assert on_top1.score(wider_gaps) == pytest.approx(0.0, abs=1e-9)
        assert on_gap.score(wider_gaps) > 0.9


@pytest.mark.parametrize("kw", [{"bins": 1}, {"bins": 2.5}, {"bins": True}, {"eps": 0}, {"eps": 0.5}])
def test_bad_psi_parameters_are_rejected(kw):
    with pytest.raises(ValueError):
        PSITop1(**kw)


def test_score_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="fit"):
        PSITop1().score(obs(np.full(10, 0.5)))


def test_module_reload_does_not_break_the_registry():
    for module in (ks, psi, ks, psi):
        importlib.reload(module)


@pytest.mark.parametrize("name", [f"{p}_{f}" for p in ("ks", "psi") for f in FEATURES])
def test_runs_through_the_runner(monkeypatch, name):
    noisy = run_toy(monkeypatch, NoisyDocs.name, detector=name)
    assert noisy[NoisyDocs.name] > noisy["clean"]
    # scrambling moves vectors between doc ids: wrong docs come back, with the same scores
    scrambled = run_toy(monkeypatch, ScrambleDocs.name, detector=name)
    assert scrambled[ScrambleDocs.name] == scrambled["clean"]
