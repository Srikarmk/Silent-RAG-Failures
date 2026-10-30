import dataclasses

import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.detect.base import Detector
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.inject.base import Injector
from silent_rag.pipeline import Retriever
from silent_rag.types import Dataset


class HashEncoder:
    """Bag of words hashed into 64 dims. Deterministic and offline."""

    def __init__(self):
        self.calls = 0  # corpus encodes only, queries don't count

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True):
        self.calls += any("filler" in t for t in texts)
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, sum(map(ord, w)) % 64] += 1
        if normalize_embeddings:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


def toy_dataset(n=40):
    corpus = {f"d{i}": f"topic{i} word{i} filler" for i in range(n)}
    queries = {f"q{i}": f"topic{i} word{i}" for i in range(n)}
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(n)}
    return Dataset("nfcorpus", corpus, queries, qrels)


def test_search_finds_the_matching_doc(tmp_path):
    ds = toy_dataset()
    r = Retriever("fake", encoder=HashEncoder())
    state = r.embed_corpus(ds, cache_dir=tmp_path)
    ids, scores, q = r.search(state, state.query_texts[:5], k=3)
    assert [row[0] for row in ids] == ["d0", "d1", "d2", "d3", "d4"]
    assert scores.shape == (5, 3) and q.shape == (5, 64)


def test_second_embed_loads_from_cache(tmp_path):
    ds = toy_dataset()
    enc = HashEncoder()
    r = Retriever("fake", encoder=enc)
    a = r.embed_corpus(ds, cache_dir=tmp_path)
    b = r.embed_corpus(ds, cache_dir=tmp_path)
    assert enc.calls == 1
    np.testing.assert_array_equal(a.doc_embeddings, b.doc_embeddings)

    Retriever("fake", normalize=False, encoder=enc).embed_corpus(ds, cache_dir=tmp_path)
    assert enc.calls == 2, "changing normalize must miss the cache"


def test_embed_texts_caches_under_name_and_key(tmp_path):
    ds = toy_dataset()
    ids, texts = list(ds.corpus), list(ds.corpus.values())
    enc = HashEncoder()
    r = Retriever("fake", encoder=enc)

    a, key_a = r.embed_texts("toy", ids, texts, cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == [tmp_path / f"toy-{key_a}.npy"]
    assert enc.calls == 1

    b, key_b = r.embed_texts("toy", ids, texts, cache_dir=tmp_path)
    assert enc.calls == 1, "second call must load from disk, not re-encode"
    assert key_b == key_a
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("change", [{"model": "fake-alt"}, {"revision": "v2"}], ids=["model", "revision"])
def test_embed_texts_misses_cache_when_model_or_revision_changes(tmp_path, change):
    ds = toy_dataset()
    ids, texts = list(ds.corpus), list(ds.corpus.values())
    base = {"model": "fake", "revision": None}
    enc_a, enc_b = HashEncoder(), HashEncoder()

    _, key_a = Retriever(**base, encoder=enc_a).embed_texts("toy", ids, texts, cache_dir=tmp_path)
    _, key_b = Retriever(**{**base, **change}, encoder=enc_b).embed_texts("toy", ids, texts, cache_dir=tmp_path)

    assert key_a != key_b
    assert sorted(tmp_path.iterdir()) == sorted(tmp_path / f"toy-{k}.npy" for k in (key_a, key_b))
    assert enc_b.calls == 1, f"changing {change} must encode, not load the first file"


@pytest.mark.parametrize("name", ["SciFact", "BeIR/scifact", "fiqa-2018", " FIQA "])
def test_held_out_aliases_are_blocked(name):
    with pytest.raises(SystemExit):
        runner.check_datasets([name], final=False)


def test_unknown_dataset_is_rejected():
    with pytest.raises(SystemExit):
        runner.check_datasets(["msmarco"], final=True)


class ScrambleDocs(Injector):
    name = "test_scramble"

    def apply(self, state, severity, seed):
        rng = np.random.default_rng(seed)
        emb = state.doc_embeddings.copy()
        hit = rng.choice(len(emb), int(severity * len(emb)), replace=False)
        emb[hit] = rng.permutation(emb[hit])
        return dataclasses.replace(state, doc_embeddings=emb)


class MeanTop1(Detector):
    name = "test_mean_top1"

    def fit(self, reference):
        self.ref = reference.topk_scores[:, 0].mean()

    def score(self, window):
        return float(self.ref - window.topk_scores[:, 0].mean())


@pytest.fixture
def fake_plugins(monkeypatch):
    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)
    monkeypatch.setitem(DETECTORS, MeanTop1.name, MeanTop1)


def test_runner_end_to_end(tmp_path, fake_plugins):
    cfg = {
        "datasets": ["nfcorpus"],
        "failures": {"test_scramble": [0.1, 0.9]},
        "detectors": ["test_mean_top1"],
        "window_size": 15,
        "trials": 4,
        "seed": 0,
    }
    df = runner.run(
        cfg,
        loader=lambda name: toy_dataset(),
        make_retriever=lambda: Retriever("fake", encoder=HashEncoder()),
    )
    assert len(df) == 3 * 4
    clean = df[df.failure == "clean"]
    assert (clean.ndcg_drop == 0).all() and not clean.harmful.any()
    by_sev = df[df.failure == "test_scramble"].groupby("severity").ndcg_drop.mean()
    assert by_sev[0.9] > by_sev[0.1] > 0
    assert df[df.severity == 0.9].harmful.all()
    assert (df.expected_harmful == (df.failure != "clean")).all()
    assert (df.model == "fake").all() and (df.seed == 0).all()
    assert df.model_revision.isna().all() and df.run_id.nunique() == 1
    assert runner.run_id(cfg) != runner.run_id({**cfg, "trials": 5})


def test_runner_names_missing_plugins():
    cfg = {"datasets": ["nfcorpus"], "failures": {"nope": [0.5]}, "detectors": [], "window_size": 5, "trials": 1}
    with pytest.raises(SystemExit, match="nope"):
        runner.run(cfg, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("fake", encoder=HashEncoder()))


def test_state_carries_query_embeddings(tmp_path):
    state = Retriever("fake", encoder=HashEncoder()).embed_corpus(toy_dataset(), cache_dir=tmp_path)
    assert state.query_embeddings.shape == (40, 64)


def test_chunked_index_ranks_parent_docs_by_best_chunk(tmp_path):
    r = Retriever("fake", encoder=HashEncoder())
    state = r.embed_corpus(toy_dataset(), cache_dir=tmp_path)
    # every doc split into a useless chunk and the real one
    junk = r.encode(["zzz qqq filler"] * len(state.doc_ids))
    chunked = dataclasses.replace(
        state,
        doc_ids=[f"{d}#{j}" for d in state.doc_ids for j in (0, 1)],
        doc_embeddings=np.stack([junk, state.doc_embeddings], axis=1).reshape(-1, 64),
        parent_ids=[d for d in state.doc_ids for _ in (0, 1)],
    )
    ids, scores, _ = r.search(chunked, k=5, query_embeddings=state.query_embeddings[:3])
    assert [row[0] for row in ids] == ["d0", "d1", "d2"]
    assert all(len(set(row)) == 5 for row in ids), "each doc shows up once"
    flat_ids, flat_scores, _ = r.search(state, k=1, query_embeddings=state.query_embeddings[:3])
    np.testing.assert_allclose(scores[:, 0], flat_scores[:, 0], rtol=1e-6)


def test_register_needs_a_unique_name():
    from silent_rag.inject.base import register

    class NoName(Injector):
        def apply(self, state, severity, seed):
            return state

    with pytest.raises(TypeError):
        register(NoName)

    class Clash(NoName):
        name = "test_scramble"

    INJECTORS["test_scramble"] = ScrambleDocs
    try:
        with pytest.raises(ValueError):
            register(Clash)
    finally:
        del INJECTORS["test_scramble"]


class RewriteQueries(Injector):
    """Changes query texts but forgets to clear the embeddings."""

    name = "test_rewrite"

    def apply(self, state, severity, seed):
        return dataclasses.replace(state, query_texts=[t + " shifted" for t in state.query_texts])


def test_runner_catches_stale_query_embeddings(monkeypatch):
    monkeypatch.setitem(INJECTORS, RewriteQueries.name, RewriteQueries)
    monkeypatch.setitem(DETECTORS, MeanTop1.name, MeanTop1)
    cfg = {"datasets": ["nfcorpus"], "failures": {"test_rewrite": [0.5]}, "detectors": ["test_mean_top1"],
           "window_size": 10, "trials": 1}
    with pytest.raises(RuntimeError, match="query_embeddings"):
        runner.run(cfg, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("fake", encoder=HashEncoder()))


@pytest.mark.parametrize("sev", [0.0, 1.5, -0.1])
def test_severity_must_be_a_fraction(sev, fake_plugins):
    cfg = {"datasets": ["nfcorpus"], "failures": {"test_scramble": [sev]}, "detectors": ["test_mean_top1"],
           "window_size": 10, "trials": 1}
    with pytest.raises(SystemExit, match="fraction"):
        runner.run(cfg, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("fake", encoder=HashEncoder()))


def test_detector_params_from_config():
    assert runner.detector_params(["a", {"b": {"sample": 5}}, {"c": None}]) == {"a": {}, "b": {"sample": 5}, "c": {}}


def test_re_embed_scores_zero_when_clean_and_grows_with_damage(tmp_path):
    from silent_rag.detect.re_embed import ReEmbed

    r = Retriever("fake", encoder=HashEncoder())
    state = r.embed_corpus(toy_dataset(200), cache_dir=tmp_path)
    idx = np.arange(20)

    def window(s):
        return runner.observe(s, idx, [[]] * len(idx), np.zeros((len(idx), 1)), s.query_embeddings[idx])

    det = ReEmbed(sample=100)
    det.attach(r)
    det.fit(window(state))
    assert det.reference_score_ == pytest.approx(0.0, abs=1e-6)
    low = det.score(window(ScrambleDocs().apply(state, 0.2, seed=0)))
    high = det.score(window(ScrambleDocs().apply(state, 0.8, seed=0)))
    assert 0 < low < high
    assert det.score(window(state)) == det.score(window(state)), "same window, same sample"


def test_re_embed_runs_in_the_runner(fake_plugins):
    from silent_rag.detect.re_embed import ReEmbed  # noqa: F401  registers it

    cfg = {
        "datasets": ["nfcorpus"],
        "failures": {"test_scramble": [0.5]},
        "detectors": [{"re_embed": {"sample": 30}}],
        "window_size": 10,
        "trials": 2,
    }
    df = runner.run(cfg, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("fake", encoder=HashEncoder()))
    clean = df[df.failure == "clean"].score
    damaged = df[df.failure == "test_scramble"].score
    assert (clean.abs() < 1e-6).all() and (damaged > 0.05).all()
    assert (df.cost_s > 0).all()
