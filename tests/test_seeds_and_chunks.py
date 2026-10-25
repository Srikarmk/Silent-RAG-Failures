import dataclasses

import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.detect.domain_classifier import RetrievedDocDomainClassifier
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.inject.base import Injector
from silent_rag.pipeline import Retriever
from test_pipeline import HashEncoder, MeanTop1, ScrambleDocs, toy_dataset


def fake_retriever():
    return Retriever("fake", encoder=HashEncoder())


class ChunkHalf(Injector):
    """Split every doc into a junk chunk and the real one (a stand-in for F3)."""

    name = "test_chunk"

    def attach(self, retriever):
        self.retriever = retriever

    def apply(self, state, severity, seed):
        junk = self.retriever.encode(["zzz qqq filler"] * len(state.doc_ids))
        return dataclasses.replace(
            state,
            doc_ids=[f"{d}#{j}" for d in state.doc_ids for j in (0, 1)],
            doc_texts=[t for text in state.doc_texts for t in ("zzz qqq filler", text)],
            doc_embeddings=np.stack([junk, state.doc_embeddings], axis=1).reshape(-1, junk.shape[1]),
            parent_ids=[d for d in state.doc_ids for _ in (0, 1)],
        )


def test_observation_carries_parent_ids_for_chunked_indexes(monkeypatch):
    seen = []

    class Spy(MeanTop1):
        name = "test_spy"

        def score(self, window):
            seen.append(window.parent_ids)
            return super().score(window)

    monkeypatch.setitem(INJECTORS, ChunkHalf.name, ChunkHalf)
    monkeypatch.setitem(DETECTORS, Spy.name, Spy)
    cfg = {"datasets": ["nfcorpus"], "failures": {ChunkHalf.name: [0.5]}, "detectors": [Spy.name],
           "window_size": 10, "trials": 1}
    runner.run(cfg, loader=lambda name: toy_dataset(), make_retriever=fake_retriever)
    clean, chunked = seen
    assert clean is None
    assert len(chunked) == 80 and chunked[0] == chunked[1] == "d0"


def test_domain_classifier_runs_on_a_chunked_index(monkeypatch):
    monkeypatch.setitem(INJECTORS, ChunkHalf.name, ChunkHalf)
    name = RetrievedDocDomainClassifier.name
    cfg = {"datasets": ["nfcorpus"], "failures": {ChunkHalf.name: [0.5]}, "detectors": [name],
           "window_size": 10, "trials": 2}
    df = runner.run(cfg, loader=lambda n: toy_dataset(), make_retriever=fake_retriever)
    assert np.isfinite(df.score).all() and set(df.failure) == {"clean", ChunkHalf.name}


def test_seeds_repeat_the_run_under_one_run_id(monkeypatch):
    references = []

    class RefSpy(MeanTop1):
        name = "test_ref_spy"

        def fit(self, reference):
            references.append(tuple(reference.query_ids))
            super().fit(reference)

    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)
    monkeypatch.setitem(DETECTORS, RefSpy.name, RefSpy)
    cfg = {"datasets": ["nfcorpus"], "failures": {ScrambleDocs.name: [0.5]}, "detectors": [RefSpy.name],
           "window_size": 10, "trials": 3, "seeds": [0, 1, 2]}
    df = runner.run(cfg, loader=lambda n: toy_dataset(), make_retriever=fake_retriever)
    assert sorted(df.seed.unique()) == [0, 1, 2]
    assert len(df) == 3 * 2 * 3 and df.run_id.nunique() == 1
    # each seed draws its own reference window, so reference variance is part of the replicates
    assert len(references) == 3 and len(set(references)) == 3
    assert runner.run_id({**runner.DEFAULTS, **cfg}) != runner.run_id({**runner.DEFAULTS, **cfg, "seeds": [0, 1]})


@pytest.mark.parametrize("bad", [{"seeds": []}, {"seeds": [0, 0]}, {"seeds": [0.5]}, {"seeds": 3},
                                 {"seeds": [True]}, {"seeds": [0, 1], "seed": 0}])
def test_bad_seeds_are_rejected(bad):
    with pytest.raises(SystemExit):
        runner.run_seeds(bad)
