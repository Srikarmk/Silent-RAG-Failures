import dataclasses

import numpy as np
import pytest
import yaml

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
    attached_states = []
    applied_states = []

    def __init__(self):
        CountingInjector.made += 1

    def attach(self, retriever, state):
        CountingInjector.attached.append(retriever)
        CountingInjector.attached_states.append(state)

    def apply(self, state, severity, seed):
        CountingInjector.applied_states.append(state)
        return super().apply(state, severity, seed)


def test_one_injector_per_failure_attached_once(monkeypatch):
    monkeypatch.setitem(INJECTORS, CountingInjector.name, CountingInjector)
    monkeypatch.setitem(DETECTORS, MeanTop1.name, MeanTop1)
    CountingInjector.made, CountingInjector.attached = 0, []
    run({"failures": {CountingInjector.name: [0.1, 0.5, 0.9]}, "detectors": [MeanTop1.name]})
    assert CountingInjector.made == 1 and len(CountingInjector.attached) == 1


def test_attach_and_every_apply_get_the_same_clean_state(monkeypatch):
    monkeypatch.setitem(INJECTORS, CountingInjector.name, CountingInjector)
    monkeypatch.setitem(DETECTORS, MeanTop1.name, MeanTop1)
    monkeypatch.setattr(CountingInjector, "attached_states", [])
    monkeypatch.setattr(CountingInjector, "applied_states", [])
    severities, trials = [0.1, 0.5], 3
    run({"failures": {CountingInjector.name: severities}, "detectors": [MeanTop1.name], "trials": trials})
    [clean] = CountingInjector.attached_states
    assert len(CountingInjector.applied_states) == len(severities) * trials
    assert all(s is clean for s in CountingInjector.applied_states)
    assert not clean.doc_embeddings.flags.writeable


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


class ParamInjector(Injector):
    name = "test_param"

    def __init__(self, strength=1):
        self.strength = strength

    def apply(self, state, severity, seed):
        return state


class OtherInjector(Injector):
    name = "test_other"

    def apply(self, state, severity, seed):
        return state


@pytest.fixture
def label_injectors(monkeypatch):
    monkeypatch.setitem(INJECTORS, ParamInjector.name, ParamInjector)
    monkeypatch.setitem(INJECTORS, OtherInjector.name, OtherInjector)


def test_failure_params_accepts_list_and_label_forms(label_injectors):
    assert runner.failure_params(None) == {}
    assert runner.failure_params({
        "test_other": [0.5],
        "test_param": {"injector": "test_param", "severities": [0.2]},  # a label may be its own injector's name
        "strong": {"injector": "test_param", "severities": [0.1, 0.9], "strength": 3},
    }) == {
        "test_other": ("test_other", [0.5], {}),
        "test_param": ("test_param", [0.2], {}),
        "strong": ("test_param", [0.1, 0.9], {"strength": 3}),
    }


ENTRY = {"injector": "test_param", "severities": [0.5]}


@pytest.mark.parametrize("failures, match", [
    pytest.param(["test_param"], "must be a mapping", id="not-a-mapping"),
    pytest.param({1: ENTRY}, "non-blank strings", id="label-not-str"),
    pytest.param({" ": ENTRY}, "non-blank strings", id="label-blank"),
    pytest.param({"clean": ENTRY}, "undamaged condition", id="label-clean"),
    pytest.param({" Clean ": ENTRY}, "undamaged condition", id="label-clean-case"),
    pytest.param({"test_other": ENTRY}, "registered name of 'test_other'", id="label-other-injector"),
    pytest.param({"TEST_Other": ENTRY}, "registered name of 'test_other'", id="label-other-injector-case"),
    pytest.param({"x": "test_param"}, "expected", id="entry-neither-form"),
    pytest.param({"x": {"severities": [0.5]}}, "needs `injector:`", id="injector-missing"),
    pytest.param({"x": {"injector": " ", "severities": [0.5]}}, "needs `injector:`", id="injector-blank"),
    pytest.param({"x": {"injector": None, "severities": [0.5]}}, "needs `injector:`", id="injector-null"),
    pytest.param({"x": {"injector": "nope", "severities": [0.5]}}, "no injector registered as 'nope'",
                 id="injector-unregistered"),
    pytest.param({"nope": [0.5]}, "no injector registered as 'nope'", id="list-form-unregistered"),
    pytest.param({"x": {"injector": "test_param"}}, "needs `severities", id="severities-missing"),
    pytest.param({"x": {"injector": "test_param", "severities": 0.5}}, "must be a list", id="severities-not-list"),
    pytest.param({"x": {"injector": "test_param", "severities": [1.5]}}, r"in \(0, 1\]", id="severity-out-of-range"),
    pytest.param({"a": ENTRY, "b": ENTRY}, "same parameters", id="same-params-twice"),
    pytest.param({"test_param": [0.5], "b": ENTRY}, "same parameters", id="same-params-as-list-form"),
])
def test_failure_params_rejects_bad_entries(label_injectors, failures, match):
    with pytest.raises(SystemExit, match=match):
        runner.failure_params(failures)


def test_labels_must_differ_beyond_case_and_spaces(label_injectors):
    stronger = {**ENTRY, "strength": 2}
    assert set(runner.failure_params({"test_param": ENTRY, "TEST_PARAM_2": stronger})) == {"test_param", "TEST_PARAM_2"}
    with pytest.raises(SystemExit, match="differ only in case or spaces"):
        runner.failure_params({"test_param": ENTRY, " TEST_PARAM ": stronger})


def test_config_with_a_repeated_key_is_rejected(tmp_path, monkeypatch):
    text = "datasets: [nfcorpus]\nfailures:\n  a: [0.5]\n  b: [0.5]\n"
    assert yaml.load(text, Loader=runner.UniqueKeyLoader) == yaml.safe_load(text)
    path = tmp_path / "cfg.yaml"
    path.write_text(text + "  a: [0.9]\n")
    monkeypatch.setattr("sys.argv", ["run", "--config", str(path)])
    with pytest.raises(SystemExit, match=r"line 5: repeated keys \['a'\]"):
        runner.main()


def test_duplicate_datasets_run_once():
    assert runner.check_datasets(["nfcorpus", "NFCorpus", "BeIR/nfcorpus"], final=False) == ["nfcorpus"]
