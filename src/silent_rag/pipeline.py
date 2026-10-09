"""Clean retrieval pipeline: embed, index, search. Embeddings are cached on disk
so injections don't re-embed the whole corpus every run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("cache/embeddings")


def cache_key(dataset: str, model: str, chunking: dict) -> str:
    blob = json.dumps({"dataset": dataset, "model": model, "chunking": chunking}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class Retriever:
    """TODO (#7): sentence-transformers + FAISS IndexFlatIP.

    Notes from the SciFact EDA: embed title + text, normalize embeddings,
    and decide truncation vs chunking (most docs are past ~250 tokens).
    """

    def __init__(self, model: str, normalize: bool = True, query_prefix: str = ""):
        self.model = model
        self.normalize = normalize
        self.query_prefix = query_prefix

    def embed_corpus(self, dataset):
        raise NotImplementedError

    def search(self, state, query_texts, k: int = 10):
        raise NotImplementedError
