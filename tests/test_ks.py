import dataclasses

import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY
from silent_rag.detect.ks import KSTop1
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.inject.base import Injector
from silent_rag.pipeline import Retriever
from silent_rag.types import Observation
from test_pipeline import HashEncoder, ScrambleDocs, toy_dataset


def obs(top1, seed=0):
    """Windows whose top-1 scores are `top1`, with the rest of the top 10 below it."""
    top1 = np.asarray(top1, dtype=np.float32)
    rest = top1[:, None] - np.sort(np.random.default_rng(seed).random((len(top1), 9)), axis=1) * 0.1
    scores = np.hstack([top1[:, None], rest]).astype(np.float32)
    return Observation([f"q{i}" for i in range(len(top1))], np.zeros((len(top1), 4)), [[]] * len(top1), scores)


def test_registered_under_the_dev_config_name():
    assert REGISTRY["ks_top1"] is KSTop1


def test_same_distribution_scores_low_shift_scores_high():
    rng = np.random.default_rng(0)
    det = KSTop1()
    det.fit(obs(rng.normal(0.6, 0.05, 100)))
    clean = det.score(obs(rng.normal(0.6, 0.05, 100), seed=1))
    shifted = det.score(obs(rng.normal(0.5, 0.05, 100), seed=2))
    assert clean < 0.2 < 0.6 < shifted <= 1.0


def test_score_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="fit"):
        KSTop1().score(obs(np.full(10, 0.5)))


class NoisyDocs(Injector):
    """Replace a fraction of doc vectors with random ones, a stand-in for F1's second model."""

    name = "test_noisy_docs"

    def apply(self, state, severity, seed):
        rng = np.random.default_rng(seed)
        emb = state.doc_embeddings.copy()
        hit = rng.choice(len(emb), int(severity * len(emb)), replace=False)
        noise = rng.normal(size=(len(hit), emb.shape[1]))
        emb[hit] = noise / np.linalg.norm(noise, axis=1, keepdims=True)
        return dataclasses.replace(state, doc_embeddings=emb.astype(np.float32))


def run_toy(monkeypatch, failure):
    for inj in (ScrambleDocs, NoisyDocs):
        monkeypatch.setitem(INJECTORS, inj.name, inj)
    cfg = {
        "datasets": ["nfcorpus"],
        "failures": {failure: [0.5]},
        "detectors": ["ks_top1"],
        "window_size": 15,
        "trials": 4,
        "seed": 0,
    }
    df = runner.run(
        cfg, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("fake", encoder=HashEncoder())
    )
    assert (df.detector == "ks_top1").all() and df.score.between(0, 1).all()
    return df.groupby("failure").score.mean()


def test_runs_through_the_runner_and_sees_score_changes(monkeypatch):
    scores = run_toy(monkeypatch, "test_noisy_docs")
    assert scores["test_noisy_docs"] > scores["clean"] + 0.1


def test_blind_to_damage_that_keeps_the_scores(monkeypatch):
    # scrambling moves vectors between doc ids: wrong docs come back, with the same scores
    scores = run_toy(monkeypatch, "test_scramble")
    assert scores["test_scramble"] == scores["clean"]
