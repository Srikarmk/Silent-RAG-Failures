"""Checks that a replacement model can stand in for the baseline model's document vectors (F1).

Reads only the local Hugging Face cache, never the network, so config validation stays
offline. F1 re-encodes documents with the replacement model and encodes nothing else with it,
so the model must normalise its output, match the baseline's output dimension, and must not
expect a document/passage prefix (F1 encodes documents bare). A query-only instruction
(bge-small's "Represent this sentence for searching relevant passages: ") is allowed.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

from silent_rag.pipeline import is_commit_hash

# prompt names in config_sentence_transformers.json that sentence-transformers applies to documents
DOCUMENT_PROMPT_NAMES = {"document", "passage", "corpus"}
CARD_DOCUMENT_PREFIX = "passage: "  # e5; "relevant passages: " (bge's query instruction) does not contain it


def cached_file(repo: str, revision: str | None, filename: str, cache_dir=None, required: bool = True) -> Path | None:
    """Path of `filename` in the local cache. Not cached: ValueError naming the file and how to fetch it.
    Cached as absent from the repo: ValueError if required, else None."""
    path = try_to_load_from_cache(repo, filename, cache_dir=cache_dir, revision=revision)
    if isinstance(path, str):
        return Path(path)
    if path is None:
        rev = f" --revision {revision}" if revision else ""
        raise ValueError(
            f"{repo}@{revision or 'main'}: {filename} is not in the local Hugging Face cache; "
            f"fetch it with `huggingface-cli download {repo} {filename}{rev}`"
        )
    if required:
        raise ValueError(f"{repo}@{revision or 'main'}: the repo has no {filename}")
    return None


def read_json(repo, revision, filename, cache_dir=None, required=True):
    path = cached_file(repo, revision, filename, cache_dir, required)
    return None if path is None else json.loads(path.read_text())


def module_types(modules: list[dict]) -> list[str]:
    return [m["type"].rsplit(".", 1)[-1] for m in modules]


def output_dimension(repo: str, revision: str | None, cache_dir=None) -> int:
    """Dimension of the vectors the model returns: the last Dense layer's out_features, else the pooling dimension."""
    modules = read_json(repo, revision, "modules.json", cache_dir)
    dense = [m for m in modules if m["type"].endswith("Dense")]
    if dense:
        return int(read_json(repo, revision, f"{dense[-1]['path']}/config.json", cache_dir)["out_features"])
    pooling = next((m for m in modules if m["type"].endswith("Pooling")), None)
    if pooling is None:
        raise ValueError(f"{repo}@{revision or 'main'}: modules.json has no Pooling module")
    return int(read_json(repo, revision, f"{pooling['path']}/config.json", cache_dir)["word_embedding_dimension"])


def check_replacement_model(model: str, revision: str, baseline_model: str, baseline_revision: str | None,
                            cache_dir=None) -> None:
    """Raise ValueError if `model`@`revision` can't replace the baseline's document vectors in F1.

    `revision` must be a pinned commit hash; the baseline may be unpinned (resolved through the
    cache's refs). Rejects: no Normalize module; an output dimension different from the
    baseline's; a document-side prompt or a default prompt in config_sentence_transformers.json
    (a default prompt is prepended to documents too); "passage: " in the model card. That file
    and the card may be absent from the repo, but must be cached if the repo has them.
    """
    where = f"{model}@{revision}"
    if not is_commit_hash(revision):
        raise ValueError(f"{model}: revision must be a 40-character lowercase commit hash, got {revision!r}")

    types = module_types(read_json(model, revision, "modules.json", cache_dir))
    if "Normalize" not in types:
        raise ValueError(f"{where} has no Normalize module ({'+'.join(types)}); F1 needs unit-length vectors")

    dim, base_dim = output_dimension(model, revision, cache_dir), output_dimension(baseline_model, baseline_revision, cache_dir)
    if dim != base_dim:
        raise ValueError(f"{where} gives {dim}-dim vectors, the baseline {baseline_model} gives {base_dim}")

    st_config = read_json(model, revision, "config_sentence_transformers.json", cache_dir, required=False) or {}
    prompts = {k: v for k, v in (st_config.get("prompts") or {}).items() if v}
    document_prompts = {k: v for k, v in prompts.items() if k.lower() in DOCUMENT_PROMPT_NAMES}
    if document_prompts:
        raise ValueError(f"{where} has document prompts {document_prompts}, which F1 doesn't add")
    if st_config.get("default_prompt_name"):
        raise ValueError(f"{where} sets default_prompt_name {st_config['default_prompt_name']!r}, which would prefix documents")

    card = cached_file(model, revision, "README.md", cache_dir, required=False)
    if card is not None and CARD_DOCUMENT_PREFIX in card.read_text(errors="ignore"):
        raise ValueError(f"{where}: the model card documents a {CARD_DOCUMENT_PREFIX!r} document prefix, which F1 doesn't add")
