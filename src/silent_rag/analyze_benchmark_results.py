"""Analysis entry point: every table for one run config and its results parquet.

    python -m silent_rag.analyze_benchmark_results --config RUN.yaml --analysis ANALYSIS.yaml \\
        --results results/RUN.parquet --out results/RUN/ [--final] [--only NAME ...] [--skip NAME ...]

Writes one parquet per table, named after the function that makes it.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"  # before anything imports torch or faiss (spec 5.9)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from silent_rag import data
from silent_rag import run as runner
from silent_rag.eval import window_analyses as wa
from silent_rag.eval.all_test_questions import (
    mean_ndcg10_drop_all_test_questions_by_severity,
    per_question_ndcg10_by_failure_severity_seed,
    statistic_all_test_questions_vs_validation_reference_by_severity,
)
from silent_rag.eval.f1_partial_upgrade import F1, replacement_model_difference_table
from silent_rag.eval.question_vulnerability import (
    question_ndcg10_drop_by_severity,
    question_vulnerability_clustering,
    window_mean_question_vulnerability_by_label,
)
from silent_rag.eval.window_questions import window_bootstrap_ci_vs_mean_ndcg10_drop, window_per_question_ndcg10_differences
from silent_rag.inject.base import REGISTRY as INJECTORS


class TableNotApplicable(Exception):
    """A table that does not apply to this run or analysis config (checked before computing it)."""


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=runner.UniqueKeyLoader)


def _require(analysis: dict, *paths) -> None:
    for path in paths:
        node = analysis
        for key in path:
            if not isinstance(node, dict) or key not in node:
                raise TableNotApplicable(f"the analysis config has no {'.'.join(path)}")
            node = node[key]


def analyze(run_config: dict, analysis: dict, runs: pd.DataFrame, out_dir: Path, final: bool = False,
            only=None, skip=None, loader=data.load, make_retriever=None) -> dict[str, str]:
    """Write every selected table to out_dir; return {table: "written" or "skipped: <reason>"}.

    Stops if `runs` is not exactly this run config's run_id, or if the reference check fails. A
    table is skipped only when it does not apply, checked before computing it: an analysis setting
    it needs is absent, or the run config has no labels for its injector (TableNotApplicable).
    Every other exception, ValueError included, propagates and stops the run.
    """
    found, expected = sorted(set(runs["run_id"])), runner.run_id_from_run_config(run_config)
    if found != [expected]:
        raise SystemExit(f"results have run_id {found}, the run config gives {expected!r}")
    cfg = {**runner.DEFAULTS, **run_config}
    alpha, min_drop = cfg["alpha"], cfg["min_drop"]
    runner.load_plugins()
    failures = runner.failure_params(run_config["failures"])
    harmful = [label for label, (name, _, _) in failures.items() if INJECTORS[name].harmful]
    statistics = list(runner.detector_params(run_config["detectors"]))
    io = {"loader": loader, "make_retriever": make_retriever, "final": final}
    boot = analysis.get("seed_cluster_bootstrap", {})
    f1 = analysis.get("injector_specific", {}).get(F1, {})
    share_drops = analysis.get("min_mean_ndcg10_drops_for_harmful_share", [])
    cache = {}

    def grid():
        g = analysis["min_mean_ndcg10_drop_grid_for_auroc"]
        return np.round(np.arange(g["start"], g["stop"] + g["step"] / 2, g["step"]), 10).tolist()

    def once(key, compute):  # the expensive inputs are computed at most once per call
        if key not in cache:
            cache[key] = compute()
        return cache[key]

    def per_question():
        return once("per_question", lambda: per_question_ndcg10_by_failure_severity_seed(run_config, analysis, **io))

    def window_questions():
        return once("window_questions", lambda: window_per_question_ndcg10_differences(run_config, runs, **io))

    def drops():
        return once("drops", lambda: question_ndcg10_drop_by_severity(per_question()))

    def question_vectors():
        def compute():
            ds = loader(runner.check_datasets(run_config["datasets"], final)[0])
            retriever = (make_retriever or (lambda: runner.default_retriever(run_config)))()
            return pd.DataFrame(retriever.embed_queries(list(ds.queries.values())), index=list(ds.queries))
        return once("question_vectors", compute)

    def each(table):  # one table per statistic x harmful failure label
        return pd.concat([table(s, f).assign(statistic=s, failure=f) for s in statistics for f in harmful], ignore_index=True)

    def reference_check():
        reference = f1["reference_mean_window_ndcg10_drop"]
        try:
            return wa.check_mean_window_ndcg10_drop_against_reference(
                runs, reference["values"], reference["seeds"], trials=reference.get("trials"))
        except ValueError as e:
            raise SystemExit(f"reference check failed: {e}") from None

    def auroc_by_severity(s, f):
        return pd.concat([wa.auroc_statistic_harmful_vs_negative_windows_by_severity(
            runs, s, f, m, alpha, boot["resamples"], boot["seed"]).assign(min_mean_ndcg10_drop=float(m))
            for m in share_drops], ignore_index=True)

    f1_labels = any(name == F1 for name, _, _ in failures.values())
    reference = ("injector_specific", F1, "reference_mean_window_ndcg10_drop")
    shares, seed_boot = ("min_mean_ndcg10_drops_for_harmful_share",), ("seed_cluster_bootstrap",)
    tables = {  # name: (required analysis settings, applies to this run config, compute)
        "check_mean_window_ndcg10_drop_against_reference": ([reference], f1_labels, reference_check),  # first: stops early
        "share_of_windows_with_significant_ndcg10_drop_by_severity": ([shares], True, lambda: (
            wa.share_of_windows_with_significant_ndcg10_drop_by_severity(runs, share_drops, alpha))),
        "median_seconds_per_statistic_evaluation": ([], True, lambda: wa.median_seconds_per_statistic_evaluation(runs)),
        "auroc_statistic_harmful_vs_negative_windows_by_severity": ([shares, seed_boot], True, lambda: each(auroc_by_severity)),
        "auroc_statistic_harmful_vs_negative_windows_by_min_ndcg10_drop": (
            [("min_mean_ndcg10_drop_grid_for_auroc",), seed_boot], True,
            lambda: each(lambda s, f: wa.auroc_statistic_harmful_vs_negative_windows_by_min_ndcg10_drop(
                runs, s, f, grid(), alpha, boot["resamples"], boot["seed"]))),
        "spearman_statistic_vs_window_mean_ndcg10_drop": ([seed_boot], True, lambda: each(
            lambda s, f: wa.spearman_statistic_vs_window_mean_ndcg10_drop(runs, s, f, boot["resamples"], boot["seed"]))),
        "window_p_value_vs_mean_ndcg10_drop": ([shares], True, lambda: wa.window_p_value_vs_mean_ndcg10_drop(
            runs, alpha, share_drops)),
        "per_question_ndcg10_by_failure_severity_seed": ([], True, per_question),
        "mean_ndcg10_drop_all_test_questions_by_severity": ([("question_bootstrap",)], True, lambda: (
            mean_ndcg10_drop_all_test_questions_by_severity(per_question(), analysis))),
        "statistic_all_test_questions_vs_validation_reference_by_severity": ([], True, lambda: (
            statistic_all_test_questions_vs_validation_reference_by_severity(run_config, analysis, **io))),
        "window_per_question_ndcg10_differences": ([], True, window_questions),
        "window_bootstrap_ci_vs_mean_ndcg10_drop": ([("window_bootstrap",)], True, lambda: (
            window_bootstrap_ci_vs_mean_ndcg10_drop(window_questions(), analysis))),
        "question_ndcg10_drop_by_severity": ([], True, drops),
        "question_vulnerability_clustering": ([("question_vulnerability",)], True, lambda: pd.concat([
            question_vulnerability_clustering(drops(), question_vectors(), analysis, fraction).assign(top_fraction=fraction)
            for fraction in analysis["question_vulnerability"]["top_fractions"]], ignore_index=True)),
        "window_mean_question_vulnerability_by_label": ([], True, lambda: dict(zip(
            ("per_window", "summary"), window_mean_question_vulnerability_by_label(drops(), window_questions(), runs, alpha, min_drop)))),
        "replacement_model_difference_table": ([("question_bootstrap",)], f1_labels, lambda: (
            replacement_model_difference_table(run_config, analysis, **io))),
    }
    unknown = sorted((set(only or []) | set(skip or [])) - set(tables))
    if unknown:
        raise SystemExit(f"unknown tables {unknown}, available: {list(tables)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for name, (requires, applies, compute) in tables.items():
        if (only and name not in only) or name in (skip or []):
            continue
        try:  # only the explicit checks before computing; nothing raised by compute() is caught
            if not applies:
                raise TableNotApplicable(f"the run config has no {F1} labels")
            _require(analysis, *requires)
        except TableNotApplicable as e:
            status[name] = f"skipped: {e}"
            continue
        result = compute()
        for suffix, frame in (result.items() if isinstance(result, dict) else [("", result)]):
            frame.to_parquet(out_dir / f"{name}{'_' + suffix if suffix else ''}.parquet", index=False)
        status[name] = "written"
    return status


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the run config YAML")
    parser.add_argument("--analysis", required=True, help="the analysis config YAML")
    parser.add_argument("--results", required=True, help="the runner's results parquet")
    parser.add_argument("--out", required=True, help="output directory for the tables")
    parser.add_argument("--final", action="store_true", help="allow the held-out datasets")
    parser.add_argument("--only", nargs="+", help="compute only these tables")
    parser.add_argument("--skip", nargs="+", help="skip these tables")
    args = parser.parse_args(argv)
    status = analyze(load_yaml(args.config), load_yaml(args.analysis), pd.read_parquet(args.results), Path(args.out),
                     final=args.final, only=args.only, skip=args.skip)
    for name, outcome in status.items():
        print(f"{name}: {outcome}")


if __name__ == "__main__":
    main()
