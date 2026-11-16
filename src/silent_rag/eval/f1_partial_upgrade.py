"""F1-specific measurements: how different the replacement model's document vectors and
rankings are from the baseline model's. Kept small; another failure gets its own module."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from silent_rag import data
from silent_rag import run as runner
from silent_rag.detect.re_embed import cosine_distance
from silent_rag.eval.ground_truth import bootstrap_ci, per_query_ndcg
from silent_rag.inject.base import REGISTRY as INJECTORS

F1 = "F1_partial_upgrade"


def mean_cosine_baseline_vs_replacement_document_vectors(baseline_vectors, replacement_vectors) -> float:
    """Mean over documents of the cosine between each document's baseline and replacement vector.

    Rows are documents, in the same order in both arrays. The cosine is 1 - re_embed.cosine_distance,
    which normalises each row itself, so vectors need not be unit length.
    """
    baseline, replacement = np.asarray(baseline_vectors, dtype=float), np.asarray(replacement_vectors, dtype=float)
    if baseline.ndim != 2 or baseline.shape != replacement.shape:
        raise ValueError(f"need two (n_documents, dim) arrays of the same shape, got {baseline.shape} and {replacement.shape}")
    return float((1.0 - cosine_distance(baseline, replacement)).mean())


def topk_documents_shared_per_question(baseline_ids, switched_ids) -> np.ndarray:
    """Number of documents in both top-k lists, per question: len(set(a) & set(b)).
    Order within a list does not matter. Both arguments list one top-k per question, same order."""
    if len(baseline_ids) != len(switched_ids):
        raise ValueError(f"{len(baseline_ids)} baseline lists but {len(switched_ids)} switched lists")
    return np.array([len(set(a) & set(b)) for a, b in zip(baseline_ids, switched_ids)], dtype=int)


def replacement_model_difference_table(run_config: dict, analysis_config: dict, loader=data.load,
                                       make_retriever=None, final: bool = False) -> pd.DataFrame:
    """One row per F1 label of the run config: how far the replacement model is from the baseline.

    - mean_cosine: baseline vs replacement vector of each document, averaged (the replacement
      vectors are the injector's alt_embeddings, built by attach() through the f1_alt cache);
    - mean_topk_shared: documents in both the baseline's and the fully switched system's top k;
    - switched_ndcg10: nDCG@k of the fully switched system, where questions and documents are
      both encoded with the replacement model. Questions use the prefix listed for the model in
      the analysis config (injector_specific.F1_partial_upgrade.replacement_model_difference_table
      .query_prefixes), else "": the replacement model's own setting, not the baseline's.
    CIs are question bootstraps over the test questions (analysis_config["question_bootstrap"]).
    The held-out rule applies: SciFact/FiQA need final=True. loader and make_retriever are run()'s
    injection points.
    """
    names = runner.check_datasets(run_config["datasets"], final)
    if len(names) != 1:
        raise ValueError(f"one dataset per analysis, got {names}")
    runner.load_plugins()
    labels = {label: params for label, (name, _, params) in runner.failure_params(run_config.get("failures")).items()
              if name == F1}
    settings = analysis_config.get("injector_specific", {}).get(F1, {}).get("replacement_model_difference_table", {})
    prefixes = settings.get("query_prefixes") or {}
    boot = analysis_config["question_bootstrap"]
    k = run_config.get("k", runner.DEFAULTS["k"])

    ds = loader(names[0])
    baseline = (make_retriever or (lambda: runner.default_retriever(run_config)))()
    state = runner.read_only(baseline.embed_corpus(ds))
    baseline_ids, _, _ = baseline.search(state, k=k, query_embeddings=state.query_embeddings)
    baseline_ndcg = per_query_ndcg(baseline_ids, state.query_ids, ds.qrels, k)

    rows = []
    for label, params in labels.items():
        injector = INJECTORS[F1](**params)
        injector.attach(baseline, state)
        switched_retriever = injector.build_replacement_model_retriever(
            baseline, query_prefix=prefixes.get(injector.alt_model, ""))
        switched = dataclasses.replace(state, doc_embeddings=injector.alt_embeddings)
        switched_ids, _, _ = baseline.search(
            switched, k=k, query_embeddings=switched_retriever.embed_queries(state.query_texts))
        shared, shared_low, shared_high = bootstrap_ci(
            topk_documents_shared_per_question(baseline_ids, switched_ids), n_boot=boot["resamples"], seed=boot["seed"])
        ndcg, ndcg_low, ndcg_high = bootstrap_ci(
            per_query_ndcg(switched_ids, state.query_ids, ds.qrels, k), n_boot=boot["resamples"], seed=boot["seed"])
        rows.append({
            "label": label, "replacement_model": injector.alt_model, "replacement_revision": injector.alt_revision,
            "k": k, "mean_cosine": mean_cosine_baseline_vs_replacement_document_vectors(state.doc_embeddings,
                                                                                         injector.alt_embeddings),
            "mean_topk_shared": shared, "topk_shared_ci_low": shared_low, "topk_shared_ci_high": shared_high,
            "baseline_ndcg10": float(baseline_ndcg.mean()), "switched_ndcg10": ndcg,
            "switched_ndcg10_ci_low": ndcg_low, "switched_ndcg10_ci_high": ndcg_high,
        })
    return pd.DataFrame(rows)
