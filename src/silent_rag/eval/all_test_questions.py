"""Whole-test-set analyses, recomputed outside the runner: every test question at once,
not 100-question windows. Uses qrels only through eval.ground_truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silent_rag import data
from silent_rag import run as runner
from silent_rag.eval.ground_truth import per_query_ndcg
from silent_rag.inject.base import REGISTRY as INJECTORS


def per_question_ndcg10_by_failure_severity_seed(run_config: dict, analysis_config: dict, loader=data.load,
                                                 make_retriever=None, final: bool = False) -> pd.DataFrame:
    """One row per test question x failure label x severity x seed, for every label in the run config.

    Columns: question_id, failure, injector, severity, seed, baseline_ndcg10, modified_ndcg10,
    n_relevant (documents with grade >= 1), n_relevant_modified (relevant documents among the rows
    the injector modifies, from modified_document_row_indices; null when the injector returns None).

    Each injector is applied with the raw seed, apply(state, severity, seed), for every seed of the
    run config, as notebook 08 does. Runner windows use injector_seed(seed, trial) instead, so the
    modified rows here differ from those in any runner window. Questions are searched with the
    baseline question vectors (via run.window_queries, which re-embeds only if an injector changes
    question texts). The held-out rule applies: SciFact/FiQA need final=True.
    """
    names = runner.check_datasets(run_config["datasets"], final)
    if len(names) != 1:
        raise ValueError(f"one dataset per analysis, got {names}")
    runner.load_plugins()
    failures = runner.failure_params(run_config.get("failures"))
    seeds = runner.run_seeds(run_config)
    k = run_config.get("k", runner.DEFAULTS["k"])

    ds = loader(names[0])
    retriever = (make_retriever or (lambda: runner.default_retriever(run_config)))()
    state = runner.read_only(retriever.embed_corpus(ds))
    everyone = np.arange(len(state.query_ids))
    baseline_ids, _, _ = retriever.search(state, k=k, query_embeddings=state.query_embeddings)
    baseline_ndcg = per_query_ndcg(baseline_ids, state.query_ids, ds.qrels, k)
    relevant = [{d for d, grade in ds.qrels[q].items() if grade >= 1} for q in state.query_ids]
    row_doc = np.asarray(state.parent_ids if state.parent_ids is not None else state.doc_ids)

    tables = []
    for label, (name, severities, params) in failures.items():
        injector = INJECTORS[name](**params)
        injector.attach(retriever, state)
        for severity in severities:
            for seed in seeds:
                modified = runner.read_only(injector.apply(state, severity, seed))
                ids, _, _ = retriever.search(
                    modified, k=k, query_embeddings=runner.window_queries(retriever, state, modified, everyone))
                rows = injector.modified_document_row_indices(len(state.doc_ids), severity, seed)
                if rows is None:
                    n_modified = pd.array([pd.NA] * len(relevant), dtype="Int64")
                else:
                    docs = set(row_doc[np.asarray(rows, dtype=int)].tolist())
                    n_modified = pd.array([len(r & docs) for r in relevant], dtype="Int64")
                tables.append(pd.DataFrame({
                    "question_id": state.query_ids, "failure": label, "injector": name,
                    "severity": float(severity), "seed": seed, "baseline_ndcg10": baseline_ndcg,
                    "modified_ndcg10": per_query_ndcg(ids, state.query_ids, ds.qrels, k),
                    "n_relevant": [len(r) for r in relevant], "n_relevant_modified": n_modified,
                }))
    return pd.concat(tables, ignore_index=True)
