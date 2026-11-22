import gzip
import json
import re
import zlib

import numpy as np
import pytest

from silent_rag import data
from silent_rag import run as runner
from silent_rag.inject.f3_chunk_fallback import ChunkFallback
from silent_rag.pipeline import Retriever, chunk_text
from silent_rag.types import Dataset
from test_pipeline import MeanTop1


class WordTokenizer:
    """One token per whitespace-separated word, with character offsets."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True, verbose=False):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


class WordEncoder:
    """Bag of words hashed into 4096 dims (no collisions on these toy words), and
    every text truncated at two words, like a model at its max_seq_length."""

    tokenizer = WordTokenizer()

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True):
        out = np.zeros((len(texts), 4096), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split()[:2]:
                out[i, zlib.crc32(w.encode()) % 4096] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def chunked(size=2, **kw):
    return Retriever("fake", encoder=WordEncoder(), chunk_size=size, **kw)


def long_dataset(n=30):
    # the matching word sits at the end of each doc, past the first chunk
    corpus = {f"d{i}": f"filler{i} pad{i} more{i} topic{i}" for i in range(n)}
    queries = {f"q{i}": f"topic{i}" for i in range(n)}
    return Dataset("nfcorpus", corpus, queries, {f"q{i}": {f"d{i}": 1} for i in range(n)})


# chunking


def test_chunks_join_back_into_the_text():
    text = "  alpha beta\tgamma  delta epsilon "
    pieces = chunk_text(text, WordTokenizer(), 2)
    assert "".join(pieces) == text
    assert [p.split() for p in pieces] == [["alpha", "beta"], ["gamma", "delta"], ["epsilon"]]


def test_a_short_text_is_one_chunk():
    assert chunk_text("one two", WordTokenizer(), 5) == ["one two"]
    assert chunk_text("", WordTokenizer(), 5) == [""]


def test_chunked_corpus_has_parent_ids_and_its_own_cache_key():
    ds = long_dataset(4)
    state = chunked(2).embed_corpus(ds)
    assert state.doc_ids[:2] == ["d0#0", "d0#1"] and state.parent_ids[:2] == ["d0", "d0"]
    assert len(state.doc_ids) == 8
    assert state.config["chunking"] == {"unit": "tokens", "size": 2, "overlap": 0}
    flat = Retriever("fake", encoder=WordEncoder()).embed_corpus(ds)
    assert flat.parent_ids is None and flat.config["cache_key"] != state.config["cache_key"]


def hit_rate(retriever, state):
    top, _, _ = retriever.search(state, state.query_texts, k=1)
    return np.mean([t[0] == f"d{i}" for i, t in enumerate(top)])


def test_chunking_finds_what_truncation_misses():
    ds = long_dataset()
    flat = Retriever("fake", encoder=WordEncoder())
    assert hit_rate(chunked(2), chunked(2).embed_corpus(ds)) == 1.0
    assert hit_rate(flat, flat.embed_corpus(ds)) < 0.5


@pytest.mark.parametrize("size", [0, -1, True, 2.5])
def test_chunk_size_must_be_a_positive_int(size):
    with pytest.raises(ValueError, match="chunk_size"):
        Retriever("fake", chunk_size=size)


def test_chunk_size_must_leave_room_for_special_tokens():
    Retriever("fake", chunk_size=254, max_seq_length=256)
    with pytest.raises(ValueError, match="max_seq_length - 2"):
        Retriever("fake", chunk_size=255, max_seq_length=256)


# the injector


def attached():
    retriever = chunked(2)
    state = retriever.embed_corpus(long_dataset())
    inj = ChunkFallback()
    inj.attach(retriever, state)
    return retriever, state, inj


def test_fallback_replaces_a_docs_chunks_with_one_row():
    _, state, inj = attached()
    out = inj.apply(state, 0.5, seed=0)
    parents = out.parent_ids
    counts = {d: parents.count(d) for d in set(parents)}
    assert sorted(set(counts.values())) == [1, 2]
    assert sum(c == 1 for c in counts.values()) == 15
    single = [i for i, d in enumerate(out.doc_ids) if d.endswith("#all")]
    assert all(out.doc_texts[i] == long_dataset().corpus[out.parent_ids[i]] for i in single)
    # unchanged docs keep the clean rows exactly
    kept = [i for i, d in enumerate(out.doc_ids) if not d.endswith("#all")]
    clean_row = {d: i for i, d in enumerate(state.doc_ids)}
    assert np.array_equal(out.doc_embeddings[kept], state.doc_embeddings[[clean_row[out.doc_ids[i]] for i in kept]])
    assert len(set(parents)) == 30


def test_fallback_is_deterministic_and_nested():
    _, state, inj = attached()
    a, b = inj.apply(state, 0.3, seed=4), inj.apply(state, 0.3, seed=4)
    assert a.doc_ids == b.doc_ids and np.array_equal(a.doc_embeddings, b.doc_embeddings)
    low = {d for d in inj.apply(state, 0.3, seed=4).doc_ids if d.endswith("#all")}
    high = {d for d in inj.apply(state, 0.7, seed=4).doc_ids if d.endswith("#all")}
    assert low < high


def test_fallback_leaves_the_state_alone():
    _, state, inj = attached()
    ids, emb = list(state.doc_ids), state.doc_embeddings.copy()
    inj.apply(state, 0.9, seed=1)
    assert state.doc_ids == ids and np.array_equal(state.doc_embeddings, emb)


def test_fallback_hurts_when_the_answer_is_past_the_first_chunk():
    retriever, state, inj = attached()
    assert hit_rate(retriever, state) == 1.0
    assert hit_rate(retriever, inj.apply(state, 0.5, seed=0)) < 0.8


def test_fallback_needs_a_chunked_index():
    retriever = Retriever("fake", encoder=WordEncoder())
    state = retriever.embed_corpus(long_dataset())
    with pytest.raises(ValueError, match="chunked clean index"):
        ChunkFallback().attach(retriever, state)


def test_fallback_needs_attach_and_the_same_state():
    _, state, inj = attached()
    with pytest.raises(RuntimeError, match="attach"):
        ChunkFallback().apply(state, 0.5, 0)
    other = chunked(2).embed_corpus(long_dataset(10))
    with pytest.raises(ValueError, match="don't match"):
        inj.apply(other, 0.5, 0)


def test_runner_with_a_chunked_index_and_f3():
    cfg = {"datasets": ["nfcorpus"], "failures": {"F3_chunk_fallback": [0.9]}, "detectors": [MeanTop1.name],
           "window_size": 10, "trials": 2}
    import silent_rag.detect.base as dbase
    dbase.REGISTRY.setdefault(MeanTop1.name, MeanTop1)
    df = runner.run(cfg, loader=lambda n: long_dataset(), make_retriever=lambda: chunked(2))
    f3 = df[df.failure == "F3_chunk_fallback"]
    assert len(f3) == 2 and (f3.ndcg_drop > 0).all() and f3.expected_harmful.all()


# MLDR loader


def fake_mldr(tmp_path, n_corpus=50):
    files = {}
    rows = [{"query_id": f"q-en-{i}", "query": f"question {i}",
             "positive_passages": [{"docid": f"doc-en-{i}", "text": "x"}],
             "negative_passages": [{"docid": f"doc-en-{i + 10}", "text": "y"}]} for i in range(3)]
    files["mldr-v1.0-en/test.jsonl.gz"] = rows
    files["mldr-v1.0-en/corpus.jsonl.gz"] = [{"docid": f"doc-en-{i}", "text": f" text {i} "} for i in range(n_corpus)]
    out = {}
    for name, items in files.items():
        p = tmp_path / name.replace("/", "_")
        with gzip.open(p, "wt") as f:
            f.write("\n".join(json.dumps(r) for r in items) + "\n")
        out[name] = str(p)
    q = tmp_path / "qrels.tsv"
    q.write_text("".join(f"q-en-{i}\tQ0\tdoc-en-{i}\t1\n" for i in range(3)))
    out["qrels/qrels.mldr-v1.0-en-test.tsv"] = str(q)
    return out.__getitem__


def test_mldr_subsample_keeps_judged_and_negative_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path / "datasets")
    monkeypatch.setitem(data.MLDR, "n_docs", 20)
    ds = data.load_mldr(fetch=fake_mldr(tmp_path))
    assert ds.name == "mldr_en" and len(ds.corpus) == 20
    assert {f"doc-en-{i}" for i in (0, 1, 2, 10, 11, 12)} <= set(ds.corpus)
    assert ds.corpus["doc-en-0"] == "text 0"
    assert list(ds.queries) == ["q-en-0", "q-en-1", "q-en-2"] and ds.qrels["q-en-1"] == {"doc-en-1": 1}
    # cached: a second load doesn't need the files, and gives the same subsample
    again = data.load_mldr(fetch=lambda path: pytest.fail("should read the cache"))
    assert again.corpus == ds.corpus


def test_mldr_subsample_is_fixed_by_the_seed(tmp_path, monkeypatch):
    monkeypatch.setitem(data.MLDR, "n_docs", 20)
    monkeypatch.setattr(data, "DATA_DIR", tmp_path / "a")
    a = data.load_mldr(fetch=fake_mldr(tmp_path))
    monkeypatch.setattr(data, "DATA_DIR", tmp_path / "b")
    b = data.load_mldr(fetch=fake_mldr(tmp_path))
    assert list(a.corpus) == list(b.corpus)


def test_mldr_refuses_a_subsample_smaller_than_the_judged_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path / "datasets")
    monkeypatch.setitem(data.MLDR, "n_docs", 4)
    with pytest.raises(ValueError, match="more than"):
        data.load_mldr(fetch=fake_mldr(tmp_path))


def test_mldr_is_a_dev_dataset():
    assert runner.check_datasets(["mldr_en"], final=False) == ["mldr_en"]
