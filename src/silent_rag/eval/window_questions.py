"""Rebuild the runner's windows question by question, from the run config and its results table.

The runner stores one mean drop and one p-value per window; this recovers the per-question
nDCG@10 behind them. Bootstrap resamples are never stored: paired_bootstrap(baseline, modified,
seed=trial) on a rebuilt window regenerates them and reproduces the stored values exactly.
"""

from __future__ import annotations

import pandas as pd

from silent_rag import data
from silent_rag import run as runner
from silent_rag.eval.ground_truth import bootstrap_ci, per_query_ndcg
from silent_rag.eval.window_analyses import WINDOW
from silent_rag.inject.base import REGISTRY as INJECTORS


def _one_dataset(run_config: dict, final: bool) -> str:
    names = runner.check_datasets(run_config["datasets"], final)
    if len(names) != 1:
        raise ValueError(f"one dataset per analysis, got {names}")
    return names[0]


def window_question_ids(run_config: dict, seed: int, trial: int, loader=data.load, final: bool = False):
    """(reference question ids, window question ids) of the runner's window for (seed, trial),
    in the runner's order, via run.window_question_indices."""
    ds = loader(_one_dataset(run_config, final))
    question_ids = list(ds.queries)  # the order Retriever.embed_corpus gives the runner
    reference, window = runner.window_question_indices(len(question_ids), run_config["window_size"], seed, trial)
    return [question_ids[i] for i in reference], [question_ids[i] for i in window]


def window_per_question_ndcg10_differences(run_config: dict, runs: pd.DataFrame, loader=data.load,
                                           make_retriever=None, final: bool = False) -> pd.DataFrame:
    """One row per window x question for every window in `runs`, rebuilt exactly as run_dataset
    builds it: the same injector (attached once per label), apply with run.injector_seed(seed,
    trial), question vectors via run.window_queries, the same search and k. Unmodified windows are
    included (difference 0). Rows are in the window's index order. Columns: run_id, failure,
    severity, seed, trial, question_id, baseline_ndcg10, modified_ndcg10.

    `runs` must hold exactly one run_id, equal to run_id_from_run_config(run_config).

    Each label's injector is attached once, whereas the runner attaches a fresh instance per seed.
    That matches the runner only for injectors whose attach has no random state, which is true for
    F1 (attach only loads the replacement model's cached document vectors).
    """
    name = _one_dataset(run_config, final)
    run_ids = sorted(set(runs["run_id"]))
    expected = runner.run_id_from_run_config(run_config)
    if run_ids != [expected]:
        raise ValueError(f"results table has run_id {run_ids}, the run config gives {expected!r}")
    runner.load_plugins()
    failures = runner.failure_params(run_config.get("failures"))
    k, window_size = run_config.get("k", runner.DEFAULTS["k"]), run_config["window_size"]

    ds = loader(name)
    retriever = (make_retriever or (lambda: runner.default_retriever(run_config)))()
    state = runner.read_only(retriever.embed_corpus(ds))
    clean_ids, _, _ = retriever.search(state, k=k, query_embeddings=state.query_embeddings)
    clean_ndcg = per_query_ndcg(clean_ids, state.query_ids, ds.qrels, k)

    injectors, tables = {}, []
    windows = runs[["failure", "severity", "seed", "trial"]].drop_duplicates().sort_values(["failure", "severity", "seed", "trial"])
    for failure, severity, seed, trial in windows.itertuples(index=False):
        _, idx = runner.window_question_indices(len(state.query_ids), window_size, int(seed), int(trial))
        question_ids = [state.query_ids[i] for i in idx]
        if failure == "clean":
            modified = clean_ndcg[idx]
        else:
            if failure not in injectors:
                injector_name, _, params = failures[failure]
                injectors[failure] = INJECTORS[injector_name](**params)
                injectors[failure].attach(retriever, state)
            damaged = runner.read_only(injectors[failure].apply(state, severity, seed=runner.injector_seed(int(seed), int(trial))))
            ids, _, _ = retriever.search(damaged, k=k, query_embeddings=runner.window_queries(retriever, state, damaged, idx))
            modified = per_query_ndcg(ids, question_ids, ds.qrels, k)
        tables.append(pd.DataFrame({
            "run_id": expected, "failure": failure, "severity": severity, "seed": seed, "trial": trial,
            "question_id": question_ids, "baseline_ndcg10": clean_ndcg[idx], "modified_ndcg10": modified,
        }))
    return pd.concat(tables, ignore_index=True)


def window_bootstrap_ci_vs_mean_ndcg10_drop(per_question_differences: pd.DataFrame, analysis_config: dict) -> pd.DataFrame:
    """Mean nDCG@10 drop of each window with a 95% percentile CI, from
    window_per_question_ndcg10_differences' table.

    The CI is a question bootstrap within one window: bootstrap_ci resamples that window's
    per-question drops (baseline - modified) with analysis_config["window_bootstrap"] (resamples,
    seed; 2,000 resamples in the analysis config). It is separate from the runner's one-sided paired
    test, whose p_value is stored in the results table. Columns: failure, severity, seed, trial,
    mean_ndcg10_drop, ci_low, ci_high, n_questions.
    """
    boot = analysis_config["window_bootstrap"]
    rows = []
    for (failure, severity, seed, trial), window in per_question_differences.groupby(WINDOW, sort=True):
        drops = window["baseline_ndcg10"].to_numpy() - window["modified_ndcg10"].to_numpy()
        mean, low, high = bootstrap_ci(drops, n_boot=boot["resamples"], seed=boot["seed"])
        rows.append({"failure": failure, "severity": severity, "seed": seed, "trial": trial,
                     "mean_ndcg10_drop": mean, "ci_low": low, "ci_high": high, "n_questions": len(drops)})
    return pd.DataFrame(rows)
