import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.detect.llm_judge import LLMJudge
from silent_rag.detect.two_stage import TwoStage
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.pipeline import Retriever
from silent_rag.run import load_plugins
from test_llm_judge import FakeClient
from silent_rag.detect.base import Detector
from test_pipeline import HashEncoder, ScrambleDocs, toy_dataset


class Spread(Detector):
    """A stage 1 whose score differs from window to window: the window's mean second score."""

    name = "test_spread"

    def fit(self, reference):
        pass

    def score(self, window):
        return float(window.topk_scores[:, 1].mean())




@pytest.fixture
def setup(monkeypatch):
    load_plugins()
    client = FakeClient()
    monkeypatch.setattr(LLMJudge, "client", client)
    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)
    monkeypatch.setitem(DETECTORS, Spread.name, Spread)
    return client


def run(threshold, trials=3):
    cfg = {"datasets": ["nfcorpus"], "window_size": 10, "trials": trials,
           "failures": {ScrambleDocs.name: [1.0]},
           "detectors": [Spread.name,
                         {"two_stage": {"stage1": Spread.name, "stage1_threshold": threshold,
                                        "model": "claude-haiku-5-5", "queries": 5, "depth": 2, "live": True}}]}
    df = runner.run(cfg, loader=lambda n: toy_dataset(), make_retriever=lambda: Retriever("fake", encoder=HashEncoder()))
    return df.pivot_table(index=["failure", "trial"], columns="detector", values=["score", "cost_s", "cost_usd"])


def test_nothing_flagged_means_the_judge_never_runs(setup):
    t = run(threshold=1e9)
    calls_for_reference = len(setup.calls)
    assert (t["score"]["two_stage"] == 0).all()
    assert (t["cost_usd"]["two_stage"] == 0).all()
    # only the reference window was judged (fit), never a scored window
    assert calls_for_reference <= 5 * 2


def test_everything_flagged_scores_like_the_judge_alone(setup):
    t = run(threshold=-1e9)
    assert (t["score"]["two_stage"] > 0).any()
    assert (t["cost_usd"]["two_stage"] > 0).all()
    assert (t["cost_s"]["two_stage"] >= t["cost_s"][Spread.name]).all()


def test_only_windows_above_the_threshold_are_judged(setup):
    stage1 = run(threshold=-1e9)["score"][Spread.name]
    cut = float(np.median(stage1))
    t = run(threshold=cut)
    flagged = t["score"][Spread.name] > cut
    assert flagged.any() and (~flagged).any()
    assert (t["score"]["two_stage"][~flagged] == 0).all()
    assert (t["cost_usd"]["two_stage"][~flagged] == 0).all()
    assert (t["cost_usd"]["two_stage"][flagged] > 0).all()


@pytest.mark.parametrize("params, message", [
    ({}, "stage1_threshold must be a finite number"),
    ({"stage1_threshold": float("nan")}, "finite number"),
    ({"stage1_threshold": True}, "finite number"),
    ({"stage1_threshold": 0.1, "stage1": "llm_judge"}, "cheap detector"),
    ({"stage1_threshold": 0.1, "stage1": "two_stage"}, "cheap detector"),
    ({"stage1_threshold": 0.1, "stage1": "nope"}, "isn't a registered detector"),
    ({"stage1_threshold": 0.1, "stage1": "ks_top1", "depth": 20}, "deeper than"),
])
def test_check_config_refuses(params, message):
    load_plugins()
    with pytest.raises(ValueError, match=message):
        TwoStage.check_config(params, {"k": 10})
