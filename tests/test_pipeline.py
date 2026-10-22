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
        self.calls = 0

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True):
        self.calls += 1
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


def test_runner_names_missing_plugins():
    cfg = {"datasets": ["nfcorpus"], "failures": {"nope": [0.5]}, "detectors": [], "window_size": 5, "trials": 1}
    with pytest.raises(SystemExit, match="nope"):
        runner.run(cfg, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("fake", encoder=HashEncoder()))
