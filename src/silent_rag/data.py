"""BEIR loaders from the Hugging Face Hub, pinned to the revisions the EDA used.

Only queries that have qrels in the chosen split are kept, so every query the
runner sees can be scored.
"""

from __future__ import annotations

from silent_rag.types import Dataset

SOURCES = {
    "nfcorpus": {
        "repo": "BeIR/nfcorpus",
        "revision": "b5026a0e96e8a7ac4f95f482a596389289d46269",
        "qrels": "BeIR/nfcorpus-qrels",
        "qrels_revision": "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
    },
    "scifact": {
        "repo": "BeIR/scifact",
        "revision": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
        "qrels": "BeIR/scifact-qrels",
        "qrels_revision": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
    },
    "fiqa": {
        "repo": "BeIR/fiqa",
        "revision": "979c07a7cb5ccc6ca009792241fa1250b98055dd",
        "qrels": "BeIR/fiqa-qrels",
        "qrels_revision": "252958f2d646e22cab6d0c72dd3f0d5de6d0655a",
    },
}


def doc_text(title: str | None, text: str | None) -> str:
    return f"{(title or '').strip()} {(text or '').strip()}".strip()


def load(name: str, split: str = "test") -> Dataset:
    from datasets import load_dataset

    src = SOURCES[name]
    corpus = load_dataset(src["repo"], "corpus", split="corpus", revision=src["revision"])
    queries = load_dataset(src["repo"], "queries", split="queries", revision=src["revision"])
    qrels_rows = load_dataset(src["qrels"], split=split, revision=src["qrels_revision"])

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_rows:
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])

    query_text = {str(q["_id"]): q["text"] for q in queries}
    return Dataset(
        name=name,
        corpus={str(d["_id"]): doc_text(d["title"], d["text"]) for d in corpus},
        # qrels order, so query order matches the pilot runs
        queries={q: query_text[q] for q in qrels},
        qrels=qrels,
    )
