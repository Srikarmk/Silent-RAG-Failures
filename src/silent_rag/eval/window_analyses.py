"""Per-window analyses of one runner results table (one run config, one parquet).

The table has one row per window x monitoring statistic. A window is one (failure, severity,
seed, trial); its nDCG@10 drop and p-value repeat on every statistic's row, so per-window
quantities first reduce the table to one row per window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silent_rag.eval.ground_truth import window_has_significant_ndcg10_drop

WINDOW = ["failure", "severity", "seed", "trial"]


def one_row_per_window(runs: pd.DataFrame) -> pd.DataFrame:
    """WINDOW columns plus ndcg_drop and p_value, one row per window. Raises if a window's rows disagree."""
    per_window = runs.groupby(WINDOW, sort=True)[["ndcg_drop", "p_value"]]
    if (per_window.nunique(dropna=False) > 1).any(axis=None):
        raise ValueError("rows of one window disagree on ndcg_drop or p_value; is this one run's table?")
    return per_window.first().reset_index()


def share_of_windows_with_significant_ndcg10_drop_by_severity(runs, min_mean_ndcg10_drops, alpha: float) -> pd.DataFrame:
    """Percent of windows with a significant mean nDCG@10 drop, per failure x severity x minimum drop.

    Relabels every window with window_has_significant_ndcg10_drop at each minimum drop, so the
    result does not depend on the run's own min_drop. Includes the unmodified baseline
    (failure "clean", severity 0). n_windows counts windows, not rows.
    """
    windows = one_row_per_window(runs)
    out = []
    for min_drop in min_mean_ndcg10_drops:
        labelled = window_has_significant_ndcg10_drop(
            windows["p_value"].to_numpy(), windows["ndcg_drop"].to_numpy(), alpha, min_drop)
        grouped = pd.Series(labelled).groupby([windows["failure"], windows["severity"]])
        table = grouped.agg(n_windows="size", n_significant="sum").reset_index()
        table.insert(2, "min_mean_ndcg10_drop", float(min_drop))
        out.append(table)
    result = pd.concat(out, ignore_index=True)
    result["share_percent"] = 100 * result["n_significant"] / result["n_windows"]
    return result


def median_seconds_per_statistic_evaluation(runs: pd.DataFrame) -> pd.DataFrame:
    """Median wall-clock seconds per evaluation of each monitoring statistic, over all rows
    (cost_s is measured per statistic), with the number of evaluations."""
    table = runs.groupby("detector")["cost_s"].agg(median_seconds="median", n_evaluations="size")
    return table.rename_axis("statistic").reset_index().sort_values("median_seconds", ignore_index=True)


def check_mean_window_ndcg10_drop_against_reference(runs, reference_values: dict, seeds, decimals: int = 3) -> pd.DataFrame:
    """Compare the mean window nDCG@10 drop per (failure, severity) with reference values, using
    only the windows of `seeds`. Raises ValueError listing observed vs expected on any mismatch at
    `decimals`, a missing (failure, severity), or a requested seed absent from the table.

    reference_values: {failure label: {severity: expected mean drop}}. The reference values hold
    only for the seeds they were computed on (notebook 07: seeds 0-2), so the analysis YAML must
    record those seeds next to the values. Returns the comparison table.
    """
    windows = one_row_per_window(runs)
    missing_seeds = sorted(set(seeds) - set(windows["seed"]))
    if missing_seeds:
        raise ValueError(f"seeds {missing_seeds} are not in the results table")
    observed = windows[windows["seed"].isin(seeds)].groupby(["failure", "severity"])["ndcg_drop"].agg(["mean", "size"])
    rows, problems = [], []
    for failure, by_severity in reference_values.items():
        for severity, expected in by_severity.items():
            match = observed[(observed.index.get_level_values(0) == failure)
                             & np.isclose(observed.index.get_level_values(1), severity)]
            if match.empty:
                problems.append(f"({failure}, {severity}): no windows")
                continue
            mean, n = float(match["mean"].iloc[0]), int(match["size"].iloc[0])
            rows.append({"failure": failure, "severity": float(severity), "n_windows": n,
                         "observed": mean, "expected": float(expected)})
            if round(mean, decimals) != round(float(expected), decimals):
                problems.append(f"({failure}, {severity}): observed {mean:.{decimals + 2}f}, expected {expected}")
    if problems:
        raise ValueError(f"mean window nDCG@10 drop differs from the reference at {decimals} decimals: " + "; ".join(problems))
    return pd.DataFrame(rows)
