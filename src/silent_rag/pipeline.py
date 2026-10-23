"""Clean retrieval pipeline: embed, index, search. Embeddings are cached on disk
so injections don't re-embed the whole corpus every run."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from silent_rag.types import Dataset, RetrievalState

CACHE_DIR = Path(os.environ.get("SILENT_RAG_CACHE", Path(__file__).resolve().parents[2] / "cache")) / "embeddings"


def cache_key(dataset: str, model: str, chunking: dict, **settings) -> str:
    """Anything that changes the doc vectors has to be in here, or a run will
    silently reuse stale ones."""
    blob = json.dumps(
        {"dataset": dataset, "model": model, "chunking": chunking, **settings}, sort_keys=True
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class Retriever:
    """sentence-transformers + exact FAISS inner-product search.

    Defaults match the SciFact pilot: title + text, normalized embeddings,
    IndexFlatIP, no query prefix. Chunking is still open (#5), so for now
    each doc is one vector and gets truncated at max_seq_length.
    """

    def __init__(
        self,
        model: str,
        normalize: bool = True,
        query_prefix: str = "",
        max_seq_length: int | None = None,
        revision: str | None = None,
        batch_size: int = 32,
        device: str = "cpu",
        encoder=None,
    ):
        self.model = model
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.max_seq_length = max_seq_length
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self._encoder = encoder

    @property
    def config(self) -> dict:
        return {
            "model": self.model,
            "revision": self.revision,
            "normalize": self.normalize,
            "query_prefix": self.query_prefix,
            "max_seq_length": self.max_seq_length,
            "doc_text": "title_text",
            "chunking": None,
        }

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model, revision=self.revision, device=self.device)
            if self.max_seq_length is not None:
                self._encoder.max_seq_length = self.max_seq_length
        return self._encoder

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def embed_corpus(self, dataset: Dataset, cache_dir: Path | None = None) -> RetrievalState:
        doc_ids = list(dataset.corpus)
        doc_texts = [dataset.corpus[d] for d in doc_ids]
        query_ids = list(dataset.queries)

        cfg = self.config
        key = cache_key(
            dataset.name,
            self.model,
            cfg["chunking"],
            revision=self.revision,
            normalize=self.normalize,
            max_seq_length=self.max_seq_length,
            doc_text=cfg["doc_text"],
            # ids and texts, so an edited corpus (new revision, perturbed copy) can't hit old vectors
            corpus=hashlib.sha1("\0".join(doc_ids + doc_texts).encode()).hexdigest(),
        )
        path = (cache_dir or CACHE_DIR) / f"{dataset.name}-{key}.npy"
        doc_embeddings = load_cached(path, len(doc_ids))
        if doc_embeddings is None:
            doc_embeddings = self.encode(doc_texts)
            save_cached(path, doc_embeddings)

        query_texts = [dataset.queries[q] for q in query_ids]
        return RetrievalState(
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            doc_embeddings=doc_embeddings,
            query_ids=query_ids,
            query_texts=query_texts,
            query_embeddings=self.embed_queries(query_texts),
            config={**cfg, "cache_key": key},
        )

    def embed_queries(self, query_texts: list[str]) -> np.ndarray:
        return self.encode([self.query_prefix + q for q in query_texts])

    def search(self, state: RetrievalState, query_texts=None, k: int = 10, query_embeddings=None):
        """Top-k over the state's index. Returns (topk_ids, topk_scores, query_embeddings).

        With chunking (state.parent_ids set) the ids are parent docs ranked by
        their best chunk, so nDCG is always computed on docs.
        """
        import faiss

        if query_embeddings is None:
            query_embeddings = self.embed_queries(query_texts)
        n_docs = len(state.doc_ids) if state.parent_ids is None else len(set(state.parent_ids))
        if not 1 <= k <= n_docs:
            raise ValueError(f"k={k} but the index has {n_docs} docs")
        index = faiss.IndexFlatIP(state.doc_embeddings.shape[1])
        index.add(np.ascontiguousarray(state.doc_embeddings, dtype=np.float32))
        q = np.ascontiguousarray(query_embeddings, dtype=np.float32)

        if state.parent_ids is None:
            scores, positions = index.search(q, k)
            ids = np.asarray(state.doc_ids)
            return [list(ids[row[row >= 0]]) for row in positions], scores, query_embeddings

        topk_ids, topk_scores = best_chunk_per_doc(index, q, state.parent_ids, k)
        return topk_ids, topk_scores, query_embeddings


def load_cached(path: Path, n_rows: int) -> np.ndarray | None:
    """Cached vectors, or None if missing, unreadable or the wrong shape (then we re-embed)."""
    try:
        vecs = np.load(path)
    except (OSError, ValueError, EOFError):
        return None
    return vecs if vecs.ndim == 2 and len(vecs) == n_rows else None


def save_cached(path: Path, vecs: np.ndarray) -> None:
    # unique tmp file per process, so two runs of the same setup can't clobber each other
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, vecs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def best_chunk_per_doc(index, q: np.ndarray, parent_ids: list[str], k: int):
    """Search chunks, keep each doc's best-scoring chunk, return the top k docs.
    Fetches more chunks until every query has k distinct docs (or the index runs out)."""
    n = index.ntotal
    if len(parent_ids) != n:
        raise ValueError(f"parent_ids has {len(parent_ids)} entries for {n} index rows")
    n_docs = len(set(parent_ids))
    fetch = min(n, 4 * k)
    while True:
        scores, positions = index.search(q, fetch)
        topk_ids, topk_scores = [], np.full((len(q), k), -np.finfo(np.float32).max, dtype=np.float32)
        short = False
        for i, (row_s, row_p) in enumerate(zip(scores, positions)):
            seen = []
            for score, pos in zip(row_s, row_p):
                if pos < 0:
                    break
                parent = parent_ids[pos]
                if parent not in seen:
                    topk_scores[i, len(seen)] = score
                    seen.append(parent)
                    if len(seen) == k:
                        break
            short |= len(seen) < min(k, n_docs)
            topk_ids.append(seen)
        if not short or fetch == n:
            return topk_ids, topk_scores
        fetch = min(n, fetch * 2)
