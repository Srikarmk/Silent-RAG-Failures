"""Dataset loaders from the Hugging Face Hub, pinned to fixed revisions.

BEIR sets are the revisions the EDA used. Only queries that have qrels in the
chosen split are kept, so every query the runner sees can be scored.

MLDR (English) is the long-document set for chunking failures (#12): Wikipedia
articles, median ~1,400 words, one relevant doc per query. Its 200k-doc corpus is
subsampled to a fixed 20k docs, see `load_mldr`.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np

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


MLDR = {
    "repo": "Shitao/MLDR",
    "revision": "d67138e705d963e346253a80e59676ddb418810a",
    "n_docs": 20_000,  # every judged and hard-negative doc of the split, then random others
    "seed": 0,
}
DATA_DIR = Path(os.environ.get("SILENT_RAG_CACHE", Path(__file__).resolve().parents[2] / "cache")) / "datasets"


def _hub_file(path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(MLDR["repo"], path, repo_type="dataset", revision=MLDR["revision"])


def load_mldr(split: str = "test", fetch=_hub_file) -> Dataset:
    """MLDR English, `split` queries, over a fixed 20k-doc subsample of the corpus.

    The full corpus is 200k long articles (about 2.4M chunks of 256 tokens), too slow
    to embed on a laptop for every setting. The subsample keeps every relevant doc and
    every hard negative listed for the split's queries, then adds random others
    (numpy seed 0, from the sorted remaining ids) up to 20k. Built once per split and
    cached as JSON; the corpus hash in the embedding cache key covers any change.
    """
    tag = f"mldr_en-{split}-{MLDR['revision'][:7]}-{MLDR['n_docs']}-s{MLDR['seed']}"
    cached = DATA_DIR / f"{tag}.json"
    if cached.exists():
        d = json.loads(cached.read_text())
        return Dataset(name="mldr_en", corpus=d["corpus"], queries=d["queries"], qrels=d["qrels"])

    queries, keep = {}, set()
    with gzip.open(fetch(f"mldr-v1.0-en/{split}.jsonl.gz"), "rt") as f:
        for line in f:
            row = json.loads(line)
            queries[row["query_id"]] = row["query"]
            keep |= {p["docid"] for p in row["positive_passages"]} | {p["docid"] for p in row["negative_passages"]}
    qrels: dict[str, dict[str, int]] = {}
    with open(fetch(f"qrels/qrels.mldr-v1.0-en-{split}.tsv")) as f:
        for line in f:
            q, _, d, rel = line.split()
            qrels.setdefault(q, {})[d] = int(rel)
    keep |= {d for docs in qrels.values() for d in docs}

    corpus_file = fetch("mldr-v1.0-en/corpus.jsonl.gz")
    with gzip.open(corpus_file, "rt") as f:
        all_ids = [json.loads(line)["docid"] for line in f]
    if len(keep) > MLDR["n_docs"]:
        raise ValueError(f"MLDR {split} needs {len(keep)} docs, more than the {MLDR['n_docs']}-doc subsample")
    rest = sorted(set(all_ids) - keep)
    extra = np.random.default_rng(MLDR["seed"]).choice(len(rest), MLDR["n_docs"] - len(keep), replace=False)
    keep |= {rest[i] for i in extra}
    corpus = {}
    with gzip.open(corpus_file, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["docid"] in keep:
                corpus[row["docid"]] = row["text"].strip()
    missing = keep - set(corpus)
    if missing:
        raise ValueError(f"MLDR corpus is missing {len(missing)} docs the {split} split refers to")
    corpus = {d: corpus[d] for d in sorted(corpus)}
    # qrels order, as for BEIR
    queries = {q: queries[q] for q in qrels}
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"corpus": corpus, "queries": queries, "qrels": qrels}))
    os.replace(tmp, cached)
    return Dataset(name="mldr_en", corpus=corpus, queries=queries, qrels=qrels)


def doc_text(title: str | None, text: str | None) -> str:
    return f"{(title or '').strip()} {(text or '').strip()}".strip()


def load(name: str, split: str = "test") -> Dataset:
    if name == "mldr_en":
        return load_mldr(split)
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


def load_beir(name: str) -> Dataset:
    """Load a BEIR dataset using test-split relevance judgments.

    The runner keeps using ``load`` for compatibility. This explicit entry point
    makes the intended evaluation split clear to other callers.
    """
    return load(name, split="test")
