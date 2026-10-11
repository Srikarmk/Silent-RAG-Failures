"""Reproduce the NFCorpus 256-vs-512-token retrieval comparison (Issue #3).

From the repository root:
    python experiments/nfcorpus_retrieval.py

Writes per-query scores and run metadata under results/baseline/nfcorpus/.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import date
from pathlib import Path

import datasets
import faiss
import numpy as np
import pandas as pd
import sentence_transformers
import torch
import transformers
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from silent_rag.eval.ground_truth import bootstrap_ci, per_query_ndcg
from silent_rag.run import check_datasets

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "baseline" / "nfcorpus"
CORPUS = {"repo": "BeIR/nfcorpus", "revision": "b5026a0e96e8a7ac4f95f482a596389289d46269"}
QRELS = {
    "repo": "BeIR/nfcorpus-qrels",
    "revision": "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
    "split": "test",
}
MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
MODEL_REVISION = "b207367332321f8e44f96e224ef15bc607f4dbf0"


def load_nfcorpus():
    check_datasets(["nfcorpus"], final=False)
    corpus = load_dataset(CORPUS["repo"], "corpus", split="corpus", revision=CORPUS["revision"]).to_pandas()
    queries = load_dataset(CORPUS["repo"], "queries", split="queries", revision=CORPUS["revision"]).to_pandas()
    qrels_df = load_dataset(QRELS["repo"], split=QRELS["split"], revision=QRELS["revision"]).to_pandas()

    corpus["_id"] = corpus["_id"].astype(str)
    queries["_id"] = queries["_id"].astype(str)
    qrels_df = qrels_df.astype({"query-id": str, "corpus-id": str})
    qrels = {}
    for query_id, doc_id, relevance in qrels_df[["query-id", "corpus-id", "score"]].itertuples(index=False):
        qrels.setdefault(query_id, {})[doc_id] = int(relevance)
    query_ids = list(qrels_df["query-id"].unique())
    query_texts = queries.set_index("_id").loc[query_ids, "text"].tolist()
    return corpus, query_ids, query_texts, qrels


def run_retrieval(model, corpus, query_vecs, query_ids, qrels, max_seq_length):
    model.max_seq_length = max_seq_length
    doc_texts = [f"{title.strip()} {text.strip()}".strip() for title, text in zip(
        corpus["title"].fillna(""), corpus["text"].fillna("")
    )]
    doc_tokens = model.tokenizer(doc_texts, add_special_tokens=True)["input_ids"]
    start = time.time()
    doc_vecs = model.encode(doc_texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    doc_seconds = time.time() - start

    start = time.time()
    index = faiss.IndexFlatIP(doc_vecs.shape[1])
    index.add(doc_vecs)
    _, positions = index.search(query_vecs, 10)
    doc_ids = corpus["_id"].to_numpy()
    topk = [list(doc_ids[row]) for row in positions]
    search_seconds = time.time() - start

    scores = per_query_ndcg(topk, query_ids, qrels, k=10)
    per_query = pd.DataFrame({
        "query_id": query_ids,
        "ndcg@10": scores,
        "top10": [" ".join(docs) for docs in topk],
    })
    stats = {
        "n_docs": len(corpus),
        "n_vectors": len(doc_vecs),
        "dim": doc_vecs.shape[1],
        "n_queries": len(query_ids),
        "query_tokens_max": None,
        "doc_inputs_over_max_seq_length": sum(len(ids) > max_seq_length for ids in doc_tokens),
        "doc_input_tokens_max": max(map(len, doc_tokens)),
        "doc_encode_seconds": round(doc_seconds, 1),
        "search_seconds": round(search_seconds, 2),
        "ndcg@10": float(scores.mean()),
    }
    return per_query, stats


def main():
    corpus, query_ids, query_texts, qrels = load_nfcorpus()
    model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
    model.max_seq_length = 512
    query_tokens_max = max(map(len, model.tokenizer(query_texts, add_special_tokens=True)["input_ids"]))
    start = time.time()
    query_vecs = model.encode(query_texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
    query_seconds = round(time.time() - start, 2)
    assert query_tokens_max <= 256, "The experiment assumes queries are not truncated at either limit"
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence-transformers": sentence_transformers.__version__,
        "faiss": faiss.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
    }
    for limit in (512, 256):
        name = f"minilm_trunc{limit}"
        per_query, stats = run_retrieval(model, corpus, query_vecs, query_ids, qrels, limit)
        stats["query_tokens_max"] = query_tokens_max
        stats["query_encode_seconds"] = query_seconds
        _, lo, hi = bootstrap_ci(per_query["ndcg@10"].to_numpy())
        config = {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "max_seq_length": limit,
            "doc_text": "title_text",
            "normalize": True,
            "query_prefix": "",
            "doc_prefix": "",
            "chunk_tokens": None,
            "chunk_overlap": 0,
            "chunk_title": True,
            "aggregate": "max",
            "k": 10,
            "doc_batch_size": 32,
            "query_batch_size": 16,
            "device": "cpu",
        }
        record = {
            "name": name,
            "exploratory": True,
            "computed_on": date.today().isoformat(),
            "dataset": "nfcorpus",
            "corpus": CORPUS,
            "qrels": QRELS,
            "overrides": {} if limit == 512 else {"max_seq_length": 256},
            "config": config,
            **stats,
            "ndcg@10_ci95": [lo, hi],
            "ci_method": "percentile bootstrap over queries, 10000 resamples, seed 0",
            "torch_threads": torch.get_num_threads(),
            "versions": versions,
            "platform": platform.platform(),
        }
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        per_query.to_csv(folder / "per_query.csv", index=False)
        (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name}: nDCG@10 = {stats['ndcg@10']:.4f} [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
