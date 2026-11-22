import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY
from silent_rag.detect.canary import Canary
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.pipeline import Retriever
from silent_rag.types import Dataset
from test_ks import NoisyDocs
from test_pipeline import HashEncoder, ScrambleDocs, toy_dataset


def train_split(name, split="train"):
    """Training queries on the toy corpus's topics, with ids that are never test ids."""
    assert split == "train"
    queries = {f"t{i}": f"topic{i} filler word{i}" for i in range(0, 40, 2)}
    return Dataset(name, {}, queries, {q: {} for q in queries})


@pytest.fixture(autouse=True)
def fake_training_split(monkeypatch):
    monkeypatch.setattr(Canary, "load", staticmethod(train_split))
    monkeypatch.setattr(Canary, "_queries", {})


def run_toy(monkeypatch, failure, severity=0.5, **params):
    for inj in (ScrambleDocs, NoisyDocs):
        monkeypatch.setitem(INJECTORS, inj.name, inj)
    cfg = {"datasets": ["nfcorpus"], "failures": {failure: [severity]},
           "detectors": [{"canary": {"n": 10, **params}}], "window_size": 15, "trials": 4, "seed": 0}
    df = runner.run(
        cfg, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("fake", encoder=HashEncoder())
    )
    assert df.score.between(0, 1).all()
    return df.groupby("failure").score.mean()


def test_registered():
    assert REGISTRY["canary"] is Canary


def test_clean_is_exactly_zero_and_changed_documents_are_not(monkeypatch):
    scores = run_toy(monkeypatch, NoisyDocs.name)
    assert scores["clean"] == 0.0
    assert scores[NoisyDocs.name] > 0.1


def test_sees_wrong_documents_with_the_same_scores(monkeypatch):
    # every score detector is blind to this (#8, #19, #22): the scores don't change, the documents do
    # clean is exactly 0, so any moved result is a detection
    scores = run_toy(monkeypatch, ScrambleDocs.name)
    assert scores["clean"] == 0.0 and scores[ScrambleDocs.name] > 0.0


def test_blind_to_a_change_in_what_users_ask(monkeypatch):
    # the canaries are fixed, so moving the users' traffic to one topic moves nothing they see
    scores = run_toy(monkeypatch, "H_query_topic_shift")
    assert scores["H_query_topic_shift"] == 0.0


def test_canaries_come_from_the_training_split_and_stay_the_same():
    det = Canary(n=5)
    det.dataset = "nfcorpus"
    ids, texts = det.canaries(exclude=[])
    assert len(ids) == 5 and all(q.startswith("t") for q in ids)
    assert det.canaries(exclude=[]) == (ids, texts)
    other = Canary(n=5, seed=1)
    other.dataset = "nfcorpus"
    assert other.canaries(exclude=[])[0] != ids
    # a canary that's also a scored query is left out
    assert ids[0] not in det.canaries(exclude=[ids[0]])[0]


def test_overlap_by_hand():
    r = Retriever("fake", encoder=HashEncoder())
    det = Canary(n=4, k=2)
    det.dataset = "nfcorpus"
    det.attach(r)
    det.ids_, det.texts_, det.vectors_ = ["a", "b"], ["x", "y"], np.zeros((2, 4))
    det.clean_ = [{"d1", "d2"}, {"d3", "d4"}]
    det.results = lambda window: [{"d1", "d9"}, {"d3", "d4"}]
    window = type("W", (), {"query_ids": ["q0"]})()
    # one canary kept half its results, the other all of them: 1 - (0.5 + 1) / 2
    assert det.score(window) == pytest.approx(0.25)


def test_too_few_training_queries_and_missing_setup_are_errors():
    det = Canary(n=500)
    det.dataset = "nfcorpus"
    with pytest.raises(ValueError, match="fewer than n=500"):
        det.canaries(exclude=[])
    with pytest.raises(RuntimeError, match="dataset"):
        Canary().canaries(exclude=[])
    with pytest.raises(RuntimeError, match="attach"):
        Canary().fit(None)
    with pytest.raises(RuntimeError, match="fit"):
        Canary().score(None)


@pytest.mark.parametrize("kw", [{"n": 0}, {"n": 2.5}, {"k": True}, {"seed": -1}])
def test_bad_parameters_are_rejected(kw):
    with pytest.raises(ValueError):
        Canary(**kw)


def test_k_deeper_than_the_run_fails_before_any_data():
    def loader(name):
        raise AssertionError("should have failed before loading data")

    cfg = {"datasets": ["nfcorpus"], "failures": {"H_query_topic_shift": [0.5]},
           "detectors": [{"canary": {"k": 20}}], "window_size": 15, "trials": 1}
    with pytest.raises(SystemExit, match="deeper than the k=10"):
        runner.run(cfg, loader=loader, make_retriever=lambda: Retriever("fake", encoder=HashEncoder()))
