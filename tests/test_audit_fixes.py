import dataclasses

import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.detect.base import Detector
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.inject.base import Injector
from silent_rag.pipeline import Retriever, load_cached, save_cached
from silent_rag.types import Dataset
from test_pipeline import HashEncoder, MeanTop1, ScrambleDocs, toy_dataset


def fake_retriever():
    return Retriever("fake", encoder=HashEncoder())


def run(cfg, **extra):
    base = {"datasets": ["nfcorpus"], "window_size": 10, "trials": 2}
    return runner.run({**base, **cfg, **extra}, loader=lambda name: toy_dataset(), make_retriever=fake_retriever)


# cache


def test_edited_corpus_text_misses_the_cache(tmp_path):
    ds = toy_dataset()
    a = fake_retriever().embed_corpus(ds, cache_dir=tmp_path)
    edited = Dataset(ds.name, {d: t + " extra words" for d, t in ds.corpus.items()}, ds.queries, ds.qrels)
    b = fake_retriever().embed_corpus(edited, cache_dir=tmp_path)
    assert a.config["cache_key"] != b.config["cache_key"]
    assert not np.array_equal(a.doc_embeddings, b.doc_embeddings)


@pytest.mark.parametrize("bad", [b"not a numpy file", "wrong shape"])
def test_broken_cache_file_is_re_embedded(tmp_path, bad):
    ds = toy_dataset()
    key = fake_retriever().embed_corpus(ds, cache_dir=tmp_path).config["cache_key"]
    path = tmp_path / f"{ds.name}-{key}.npy"
    if bad == "wrong shape":
        np.save(path, np.zeros((5, 64), dtype=np.float32))
    else:
        path.write_bytes(bad)
    enc = HashEncoder()
    state = Retriever("fake", encoder=enc).embed_corpus(ds, cache_dir=tmp_path)
    assert enc.calls == 1 and state.doc_embeddings.shape == (40, 64)
    assert load_cached(path, 40) is not None, "the bad file got replaced"


def test_cache_writes_leave_no_tmp_files(tmp_path):
    path = tmp_path / "x.npy"
    for _ in range(2):
        save_cached(path, np.ones((3, 2), dtype=np.float32))
    assert [p.name for p in tmp_path.iterdir()] == ["x.npy"]


# search


def test_k_bigger_than_the_index_is_an_error(tmp_path):
    r = fake_retriever()
    state = r.embed_corpus(toy_dataset(5), cache_dir=tmp_path)
    with pytest.raises(ValueError, match="k=8"):
        r.search(state, k=8, query_embeddings=state.query_embeddings)


def test_parent_ids_must_line_up_with_the_index(tmp_path):
    r = fake_retriever()
    state = r.embed_corpus(toy_dataset(), cache_dir=tmp_path)
    bad = dataclasses.replace(state, parent_ids=state.doc_ids[:-3])
    with pytest.raises(ValueError, match="parent_ids"):
        r.search(bad, k=5, query_embeddings=state.query_embeddings[:2])


# runner: nothing can quietly change shared state


class ScaleIndex(Detector):
    name = "test_scale_index"

    def fit(self, reference):
        pass

    def score(self, window):
        window.doc_embeddings *= 0.5
        return 0.0


class InPlaceInjector(Injector):
    name = "test_in_place"

    def apply(self, state, severity, seed):
        state.doc_embeddings[:5] = 0
        return state


def test_detector_cannot_modify_the_index(monkeypatch):
    monkeypatch.setitem(DETECTORS, ScaleIndex.name, ScaleIndex)
    with pytest.raises(ValueError, match="read-only"):
        run({"detectors": [ScaleIndex.name]})


def test_injector_cannot_modify_the_clean_state(monkeypatch):
    monkeypatch.setitem(INJECTORS, InPlaceInjector.name, InPlaceInjector)
    monkeypatch.setitem(DETECTORS, MeanTop1.name, MeanTop1)
    with pytest.raises(ValueError, match="read-only"):
        run({"failures": {InPlaceInjector.name: [0.5]}, "detectors": [MeanTop1.name]})


class RewriteKeepCopy(Injector):
    """Changes query texts but hands back a copy of the old embeddings."""

    name = "test_rewrite_copy"

    def apply(self, state, severity, seed):
        return dataclasses.replace(
            state,
            query_texts=[t + " shifted" for t in state.query_texts],
            query_embeddings=state.query_embeddings.copy(),
        )


def test_copied_stale_query_embeddings_are_caught(monkeypatch):
    monkeypatch.setitem(INJECTORS, RewriteKeepCopy.name, RewriteKeepCopy)
    monkeypatch.setitem(DETECTORS, MeanTop1.name, MeanTop1)
    with pytest.raises(RuntimeError, match="query_embeddings"):
        run({"failures": {RewriteKeepCopy.name: [0.5]}, "detectors": [MeanTop1.name]})


# runner: injectors


class CountingInjector(ScrambleDocs):
    name = "test_counting"
    made = 0
    attached = []

    def __init__(self):
        CountingInjector.made += 1

    def attach(self, retriever):
        CountingInjector.attached.append(retriever)


def test_one_injector_per_failure_attached_once(monkeypatch):
    monkeypatch.setitem(INJECTORS, CountingInjector.name, CountingInjector)
    monkeypatch.setitem(DETECTORS, MeanTop1.name, MeanTop1)
    CountingInjector.made, CountingInjector.attached = 0, []
    run({"failures": {CountingInjector.name: [0.1, 0.5, 0.9]}, "detectors": [MeanTop1.name]})
    assert CountingInjector.made == 1 and len(CountingInjector.attached) == 1


def test_injector_seeds_dont_collide_across_config_seeds():
    seeds = {runner.injector_seed(s, t) for s in range(5) for t in range(20)}
    assert len(seeds) == 100


# results and config


def test_p_value_is_stored_and_min_drop_gates_harmful(monkeypatch):
    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)
    monkeypatch.setitem(DETECTORS, MeanTop1.name, MeanTop1)
    cfg = {"failures": {ScrambleDocs.name: [0.9]}, "detectors": [MeanTop1.name], "window_size": 15}
    loose = run(cfg)
    strict = run(cfg, min_drop=2.0)  # nDCG can't drop by more than 1
    damaged = loose[loose.failure == ScrambleDocs.name]
    assert damaged.harmful.all() and (damaged.p_value < 0.05).all()
    assert not strict.harmful.any()


def test_run_id_ignores_spelled_out_defaults():
    cfg = {"datasets": ["nfcorpus"], "window_size": 10, "trials": 2, "detectors": []}
    assert runner.run_id({**runner.DEFAULTS, **cfg}) == runner.run_id({**runner.DEFAULTS, **cfg, "seed": 0})


@pytest.mark.parametrize("sevs", [[True], ["0.5"], 0.5, [0.0], [1.5]])
def test_bad_severities_are_rejected_readably(sevs):
    with pytest.raises(SystemExit):
        runner.check_severities({"F1": sevs})


def test_duplicate_datasets_run_once():
    assert runner.check_datasets(["nfcorpus", "NFCorpus", "BeIR/nfcorpus"], final=False) == ["nfcorpus"]
