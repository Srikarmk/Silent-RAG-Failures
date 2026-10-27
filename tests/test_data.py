"""Offline regression tests for the BEIR test-split loader (Issue #10)."""

import sys
from types import SimpleNamespace

import pytest

from silent_rag import data
from silent_rag.types import Dataset


@pytest.fixture
def fake_hub(monkeypatch):
    """Mimic the Hugging Face dataset calls; never download real datasets."""
    calls = []

    def fake_load_dataset(repo, config=None, *, split, revision):
        calls.append((repo, config, split, revision))
        if config == "corpus":
            assert split == "corpus"
            return [
                {"_id": 101, "title": "  First title  ", "text": "  First body  "},
                {"_id": "d2", "title": None, "text": "Second body"},
            ]
        if config == "queries":
            assert split == "queries"
            return [
                {"_id": "train-only", "text": "No test qrels for this query"},
                {"_id": 1, "text": "First test query"},
                {"_id": "q2", "text": "Second test query"},
            ]
        assert config is None
        return [
            {"query-id": 1, "corpus-id": 101, "score": 2},
            {"query-id": 1, "corpus-id": "d2", "score": 0},
            {"query-id": "q2", "corpus-id": "d2", "score": 1},
        ]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    return calls


@pytest.mark.parametrize("name", ["nfcorpus", "scifact", "fiqa"])
def test_load_beir_returns_dataset_with_test_only_queries(name, fake_hub):
    loaded = data.load_beir(name)

    assert isinstance(loaded, Dataset)
    assert loaded.name == name
    assert loaded.corpus == {"101": "First title First body", "d2": "Second body"}
    assert loaded.queries == {"1": "First test query", "q2": "Second test query"}
    assert loaded.qrels == {"1": {"101": 2, "d2": 0}, "q2": {"d2": 1}}
    assert set(loaded.queries) == set(loaded.qrels)
    assert all(isinstance(doc_id, str) for doc_id in loaded.corpus)
    assert all(isinstance(query_id, str) for query_id in loaded.queries)
    assert all(isinstance(query_id, str) for query_id in loaded.qrels)
    assert all(isinstance(doc_id, str) for rels in loaded.qrels.values() for doc_id in rels)
    assert all(isinstance(score, int) for rels in loaded.qrels.values() for score in rels.values())

    src = data.SOURCES[name]
    assert fake_hub == [
        (src["repo"], "corpus", "corpus", src["revision"]),
        (src["repo"], "queries", "queries", src["revision"]),
        (src["qrels"], None, "test", src["qrels_revision"]),
    ]


def test_existing_loader_still_uses_test_split_by_default(fake_hub):
    loaded = data.load("nfcorpus")
    assert list(loaded.queries) == ["1", "q2"]
    assert fake_hub[-1][2] == "test"


def test_existing_loader_preserves_explicit_split_argument(fake_hub):
    # Existing callers may request a different qrels split; avoid an API break.
    data.load("nfcorpus", split="validation")
    assert fake_hub[-1][2] == "validation"


@pytest.mark.parametrize(
    "title,text,expected",
    [
        (" Title ", " Body ", "Title Body"),
        (None, "Body", "Body"),
        ("Title", None, "Title"),
        (None, None, ""),
    ],
)
def test_doc_text_handles_missing_fields(title, text, expected):
    assert data.doc_text(title, text) == expected
