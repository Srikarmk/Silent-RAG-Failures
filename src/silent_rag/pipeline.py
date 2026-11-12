"""Clean retrieval pipeline: embed, index, search. Embeddings are cached on disk
so injections don't re-embed the whole corpus every run."""

from __future__ import annotations

import hashlib
import json
import os
import re
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


def is_commit_hash(revision) -> bool:
    """A pinned Hub revision: a full 40-character commit hash, lowercase only, since the
    revision goes into the cache key and one commit should have one spelling."""
    return isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None


class Retriever:
    """sentence-transformers + exact FAISS inner-product search.

    Defaults match the SciFact pilot: title + text, normalized embeddings,
    IndexFlatIP, no query prefix. By default each doc is one vector, truncated at
    max_seq_length (#5). With `chunk_size` every doc is cut into consecutive pieces
    of that many tokens (no overlap) and each piece is one row; search ranks docs by
    their best chunk. Long-document sets like MLDR need it (#12).
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
        chunk_size: int | None = None,
    ):
        if chunk_size is not None:
            if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
                raise ValueError(f"chunk_size must be a positive number of tokens, got {chunk_size!r}")
            # the encoder adds [CLS] and [SEP]; a longer chunk would be cut without anyone noticing
            if max_seq_length is not None and chunk_size > max_seq_length - 2:
                raise ValueError(f"chunk_size {chunk_size} doesn't fit max_seq_length {max_seq_length} "
                                 "with the two special tokens; use at most max_seq_length - 2")
        self.model = model
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.max_seq_length = max_seq_length
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self._encoder = encoder
        self.chunk_size = chunk_size

    @property
    def config(self) -> dict:
        return {
            "model": self.model,
            "revision": self.revision,
            "normalize": self.normalize,
            "query_prefix": self.query_prefix,
            "max_seq_length": self.max_seq_length,
            "doc_text": "title_text",
            "chunking": None if self.chunk_size is None else {"unit": "tokens", "size": self.chunk_size, "overlap": 0},
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

    @property
    def tokenizer(self):
        return self.encoder.tokenizer

    def chunk(self, doc_ids: list[str], texts: list[str], size: int) -> tuple[list[str], list[str], list[str]]:
        """Rows for docs cut into `size`-token pieces: (row ids, row texts, parent ids).
        Row ids are `<doc>#<j>`; a doc's pieces joined give its text back exactly."""
        ids, rows, parents = [], [], []
        for d, text in zip(doc_ids, texts):
            for j, piece in enumerate(chunk_text(text, self.tokenizer, size)):
                ids.append(f"{d}#{j}")
                rows.append(piece)
                parents.append(d)
        return ids, rows, parents

    def embed_corpus(self, dataset: Dataset, cache_dir: Path | None = None) -> RetrievalState:
        doc_ids = list(dataset.corpus)
        doc_texts = [dataset.corpus[d] for d in doc_ids]
        query_ids = list(dataset.queries)
        parent_ids = None
        if self.chunk_size is not None:
            doc_ids, doc_texts, parent_ids = self.chunk(doc_ids, doc_texts, self.chunk_size)
        doc_embeddings, key = self.embed_texts(dataset.name, doc_ids, doc_texts, cache_dir)

        query_texts = [dataset.queries[q] for q in query_ids]
        return RetrievalState(
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            doc_embeddings=doc_embeddings,
            query_ids=query_ids,
            query_texts=query_texts,
            query_embeddings=self.embed_queries(query_texts),
            parent_ids=parent_ids,
            config={**self.config, "cache_key": key},
        )

    def embed_texts(
        self, name: str, ids: list[str], texts: list[str], cache_dir: Path | None = None
    ) -> tuple[np.ndarray, str]:
        """Encode texts with this retriever's model, through the disk cache.
        Returns (vectors, cache_key); the file is `<name>-<key>.npy`."""
        cfg = self.config
        key = cache_key(
            name,
            self.model,
            cfg["chunking"],
            revision=self.revision,
            normalize=self.normalize,
            max_seq_length=self.max_seq_length,
            doc_text=cfg["doc_text"],
            # ids and texts, so an edited corpus (new revision, perturbed copy) can't hit old vectors
            corpus=hashlib.sha1("\0".join(ids + texts).encode()).hexdigest(),
        )
        path = (cache_dir or CACHE_DIR) / f"{name}-{key}.npy"
        vecs = load_cached(path, len(ids))
        if vecs is None:
            vecs = self.encode(texts)
            save_cached(path, vecs)
        return vecs, key

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


def chunk_text(text: str, tokenizer, size: int) -> list[str]:
    """Consecutive pieces of `size` tokens. Each piece runs from its first token to the
    next piece's first token, so the pieces join back into the text exactly (no overlap,
    whitespace kept). A text of at most `size` tokens is one piece."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
    starts = [offsets[i][0] for i in range(0, len(offsets), size)]
    if len(starts) <= 1:
        return [text]
    starts[0] = 0
    return [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]


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
