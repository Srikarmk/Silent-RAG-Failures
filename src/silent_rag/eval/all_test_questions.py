"""Whole-test-set analyses, recomputed outside the runner: every test question at once,
not 100-question windows. Uses qrels only through eval.ground_truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silent_rag import data
from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.eval.ground_truth import bootstrap_ci, per_query_ndcg
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


def statistic_all_test_questions_vs_validation_reference_by_severity(
    run_config: dict, analysis_config: dict, loader=data.load, make_retriever=None, final: bool = False,
) -> pd.DataFrame:
    """Each monitoring statistic on all test questions, fitted on all validation questions.

    Statistics are the run config's detectors, built from run.detector_params, attached to the
    baseline retriever and fitted on the baseline search of the validation split, as the runner
    does (which fits on a 100-question window instead). Scored on all test questions for the
    unmodified baseline (failure "clean", severity 0, one value, n_seeds 0) and for every failure
    label x severity x raw seed, apply(state, severity, seed), as in
    per_question_ndcg10_by_failure_severity_seed. The test set is compared with the validation
    set (about 323 vs 324 questions on NFCorpus), so values are not on the 100-question window
    scale. min and max are the range over seeds, not a confidence interval.
    """
    names = runner.check_datasets(run_config["datasets"], final)
    if len(names) != 1:
        raise ValueError(f"one dataset per analysis, got {names}")
    runner.load_plugins()
    failures = runner.failure_params(run_config.get("failures"))
    params = runner.detector_params(run_config["detectors"])
    classes = runner.lookup(DETECTORS, params, "detector")
    k, depth = run_config.get("k", runner.DEFAULTS["k"]), run_config.get("score_depth")

    ds, reference_ds = loader(names[0]), loader(names[0], split="validation")
    if reference_ds.corpus != ds.corpus:
        raise ValueError(f"{names[0]}: the validation split's corpus differs from the test split's")
    shared = set(ds.queries) & set(reference_ds.queries)
    if shared:
        raise ValueError(f"{names[0]}: {len(shared)} question ids are in both the test and validation splits")
    retriever = (make_retriever or (lambda: runner.default_retriever(run_config)))()

    def observation(clean, state):
        """All questions of `state` searched as the runner searches a window (state = clean or modified)."""
        everyone = np.arange(len(clean.query_ids))
        ids, scores, query_vectors = retriever.search(
            state, k=k, query_embeddings=runner.window_queries(retriever, clean, state, everyone))
        deep = None if depth is None else retriever.search(state, k=depth, query_embeddings=query_vectors)[1]
        return runner.observe(state, everyone, ids, scores, query_vectors, deep_scores=deep)

    reference = runner.read_only(retriever.embed_corpus(reference_ds))
    fitted = {}
    for name, cls in classes.items():
        fitted[name] = cls(**params[name])
        fitted[name].attach(retriever)
        fitted[name].fit(observation(reference, reference))

    baseline = runner.read_only(retriever.embed_corpus(ds))
    scored = [("clean", None, 0.0, None, observation(baseline, baseline))]
    for label, (injector_name, severities, injector_params) in failures.items():
        injector = INJECTORS[injector_name](**injector_params)
        injector.attach(retriever, baseline)
        for severity in severities:
            for seed in runner.run_seeds(run_config):
                modified = runner.read_only(injector.apply(baseline, severity, seed))
                scored.append((label, injector_name, float(severity), seed, observation(baseline, modified)))
    values = pd.DataFrame([
        {"failure": f, "injector": inj, "severity": s, "seed": seed, "statistic": name, "score": float(stat.score(obs))}
        for f, inj, s, seed, obs in scored for name, stat in fitted.items()])
    table = values.groupby(["failure", "injector", "severity", "statistic"], dropna=False, sort=False).agg(
        mean=("score", "mean"), min=("score", "min"), max=("score", "max"), n_seeds=("seed", "count"))
    return table.reset_index()


def mean_ndcg10_drop_all_test_questions_by_severity(per_question_table: pd.DataFrame, analysis_config: dict) -> pd.DataFrame:
    """Mean nDCG@10 drop over all test questions, per failure x severity, with a question-bootstrap CI.

    Takes per_question_ndcg10_by_failure_severity_seed's table. The drop is baseline_ndcg10 -
    modified_ndcg10. Each question's drop is first averaged over its seeds; the bootstrap then
    resamples questions (analysis_config["question_bootstrap"]: resamples, seed). Questions x seeds
    are never pooled, since the seeds of one question are not independent samples. Seeds are raw
    seeds, as in that table, not the runner's per-window seeds. baseline_ndcg10 is the mean baseline
    nDCG@10 over the questions (taken once per question); it is the same in every row of a run and
    equals replacement_model_difference_table's baseline_ndcg10.

    The first row is the unmodified system (failure "clean", injector null, severity 0), as in the
    other whole-test-set table: drop 0 and CI [0, 0] by definition (written, not bootstrapped),
    n_questions = the distinct questions in the input, n_seeds 0, and the same baseline_ndcg10.
    An empty input gives an empty table, with no clean row.
    """
    boot = analysis_config["question_bootstrap"]
    table = per_question_table.assign(drop=per_question_table["baseline_ndcg10"] - per_question_table["modified_ndcg10"])
    keys = ["failure", "injector", "severity"]
    per_question = table.groupby([*keys, "question_id"], sort=True).agg(
        drop=("drop", "mean"), n_seeds=("seed", "nunique"), baseline=("baseline_ndcg10", "first"))
    rows = []
    for (failure, injector, severity), group in per_question.groupby(level=keys, sort=True):
        if group["n_seeds"].nunique() != 1:
            raise ValueError(f"({failure}, {severity}): questions have different numbers of seeds")
        mean, low, high = bootstrap_ci(group["drop"].to_numpy(), n_boot=boot["resamples"], seed=boot["seed"])
        rows.append({"failure": failure, "injector": injector, "severity": float(severity), "mean_ndcg10_drop": mean,
                     "ci_low": low, "ci_high": high, "n_questions": len(group), "n_seeds": int(group["n_seeds"].iloc[0]),
                     "baseline_ndcg10": float(group["baseline"].mean())})
    if table.empty:
        return pd.DataFrame(rows)
    baselines = per_question.groupby(level="question_id", sort=True)["baseline"].first()
    clean = {"failure": "clean", "injector": None, "severity": 0.0, "mean_ndcg10_drop": 0.0, "ci_low": 0.0,
             "ci_high": 0.0, "n_questions": len(baselines), "n_seeds": 0, "baseline_ndcg10": float(baselines.mean())}
    return pd.DataFrame([clean, *rows])
