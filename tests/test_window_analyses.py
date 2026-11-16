import pandas as pd
import pytest

from silent_rag.eval.window_analyses import (
    check_mean_window_ndcg10_drop_against_reference,
    median_seconds_per_statistic_evaluation,
    one_row_per_window,
    share_of_windows_with_significant_ndcg10_drop_by_severity,
)


def table(windows, statistics=("ks_top1", "re_embed")):
    """One row per window x statistic, as the runner writes. windows: (failure, severity, seed, trial, drop, p)."""
    return pd.DataFrame([
        {"failure": f, "severity": s, "seed": seed, "trial": t, "ndcg_drop": d, "p_value": p,
         "detector": stat, "cost_s": 1.0}
        for f, s, seed, t, d, p in windows for stat in statistics])


CLEAN = [("clean", 0.0, seed, t, 0.0, 1.0) for seed in (0, 1) for t in (0, 1)]
F1 = [("F1", 0.5, 0, 0, 0.10, 0.01), ("F1", 0.5, 0, 1, 0.03, 0.01),
      ("F1", 0.5, 1, 0, 0.10, 0.20), ("F1", 0.5, 1, 1, 0.00, 0.04)]


def test_share_counts_windows_not_rows_and_includes_the_baseline():
    got = share_of_windows_with_significant_ndcg10_drop_by_severity(table(CLEAN + F1), [0.0, 0.05], alpha=0.05)
    got = got.set_index(["failure", "min_mean_ndcg10_drop"])
    # two statistics per window: without deduplication n_windows would be 8 and n_significant 6
    assert got.loc[("F1", 0.0), ["n_windows", "n_significant", "share_percent"]].tolist() == [4, 3, 75.0]
    assert got.loc[("F1", 0.05), ["n_windows", "n_significant", "share_percent"]].tolist() == [4, 1, 25.0]
    assert got.loc[("clean", 0.0), ["n_windows", "share_percent"]].tolist() == [4, 0.0]


def test_rows_of_one_window_must_agree():
    runs = table(F1)
    runs.loc[1, "ndcg_drop"] = 0.5  # second statistic's row of the first window
    with pytest.raises(ValueError, match="disagree"):
        one_row_per_window(runs)


def test_median_seconds_uses_every_row_and_counts_evaluations():
    runs = table(CLEAN)
    runs.loc[runs.detector == "re_embed", "cost_s"] = [1.0, 2.0, 3.0, 10.0]
    runs.loc[runs.detector == "ks_top1", "cost_s"] = 0.1
    got = median_seconds_per_statistic_evaluation(runs).set_index("statistic")
    assert got.loc["re_embed"].tolist() == [2.5, 4]
    assert got.loc["ks_top1"].tolist() == [pytest.approx(0.1), 4]
    assert got.index.tolist() == ["ks_top1", "re_embed"]  # cheapest first


# seeds 0-2 average 0.044 at severity 0.5; seed 3 is far off
REF_RUNS = [("F1", 0.5, seed, 0, d, 0.01) for seed, d in [(0, 0.040), (1, 0.044), (2, 0.048), (3, 0.300)]]
REFERENCE = {"F1": {0.5: 0.044}}


def test_reference_check_uses_only_the_given_seeds():
    got = check_mean_window_ndcg10_drop_against_reference(table(REF_RUNS), REFERENCE, seeds=[0, 1, 2])
    assert got[["n_windows", "expected"]].iloc[0].tolist() == [3, 0.044]
    assert got["observed"].iloc[0] == pytest.approx(0.044)


def test_reference_check_fails_when_other_seeds_are_included():
    with pytest.raises(ValueError, match=r"\(F1, 0.5\): observed 0.10800, expected 0.044"):
        check_mean_window_ndcg10_drop_against_reference(table(REF_RUNS), REFERENCE, seeds=[0, 1, 2, 3])


@pytest.mark.parametrize("reference, seeds, match", [
    ({"F1": {0.9: 0.1}}, [0, 1, 2], r"\(F1, 0.9\): no windows"),
    ({"F2": {0.5: 0.1}}, [0, 1, 2], r"\(F2, 0.5\): no windows"),
    (REFERENCE, [0, 1, 7], r"seeds \[7\] are not in the results table"),
])
def test_reference_check_reports_missing_conditions_and_seeds(reference, seeds, match):
    with pytest.raises(ValueError, match=match):
        check_mean_window_ndcg10_drop_against_reference(table(REF_RUNS), reference, seeds=seeds)
