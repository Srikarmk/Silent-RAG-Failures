"""Per-window analyses of one runner results table (one run config, one parquet).

The table has one row per window x monitoring statistic. A window is one (failure, severity,
seed, trial); its nDCG@10 drop and p-value repeat on every statistic's row, so per-window
quantities first reduce the table to one row per window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silent_rag.eval.detector_scoring import bootstrap_auroc_ci, compute_auroc
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


def auroc_statistic_harmful_vs_negative_windows_by_severity(
    runs, statistic: str, failure: str, min_mean_ndcg10_drop: float, alpha: float, resamples: int, seed: int,
) -> pd.DataFrame:
    """AUROC of a monitoring statistic separating harmful from negative windows, per severity of `failure`.

    Every window is first relabelled with window_has_significant_ndcg10_drop at `alpha` and
    `min_mean_ndcg10_drop`. At each severity of the target failure:
    - positives: that severity's windows with a significant drop;
    - negatives: all unmodified windows (whatever their label), plus, in the
      "unmodified_and_negative_controls" rows, negative-control windows without a significant drop;
    - excluded: that severity's windows without a significant drop (n_modified_windows_excluded) and
      negative-control windows with one (n_negative_control_windows_excluded; 0 in the "unmodified"
      rows, where negative controls are not part of the comparison).
    The second definition appears only when the table has negative controls. The CI is a
    seed-cluster bootstrap (bootstrap_auroc_ci): it resamples whole seeds over the same questions,
    so it reflects variability across seeds, not question-level sampling uncertainty. An empty
    class gives NaN plus a note; a CI that can't be formed (e.g. one seed) gives NaN CI plus a note.
    """
    rows = runs[runs["detector"].eq(statistic)].copy()
    rows["harmful"] = window_has_significant_ndcg10_drop(
        rows["p_value"].to_numpy(), rows["ndcg_drop"].to_numpy(), alpha, min_mean_ndcg10_drop)
    clean = rows["failure"].eq("clean")
    control = ~rows["expected_harmful"] & ~clean
    target = rows["failure"].eq(failure)
    definitions = {"unmodified": clean}
    if control.any():
        definitions["unmodified_and_negative_controls"] = clean | (control & ~rows["harmful"])
    out = []
    for severity in sorted(rows.loc[target, "severity"].unique()):
        at = target & rows["severity"].eq(severity)
        for name, negatives in definitions.items():
            counts = {
                "n_positive": int((at & rows["harmful"]).sum()),
                "n_negative": int(negatives.sum()),
                "n_modified_windows_excluded": int((at & ~rows["harmful"]).sum()),
                "n_negative_control_windows_excluded": int((control & rows["harmful"]).sum()) if name != "unmodified" else 0,
            }
            pool = rows[at | negatives]  # _auc_pool drops this severity's non-significant windows
            out.append({"severity": float(severity), "negatives": name,
                        **_auroc_and_ci(pool, statistic, failure, counts, resamples, seed), **counts})
    return pd.DataFrame(out)[["severity", "negatives", "auroc", "ci_low", "ci_high", "n_positive", "n_negative",
                              "n_modified_windows_excluded", "n_negative_control_windows_excluded", "note"]]


def _auroc_and_ci(pool, statistic, failure, counts, resamples, seed) -> dict:
    nan = float("nan")
    for cls in ("positive", "negative"):
        if counts[f"n_{cls}"] == 0:
            return {"auroc": nan, "ci_low": nan, "ci_high": nan, "note": f"undefined: no {cls} windows"}
    try:
        auroc, low, high = bootstrap_auroc_ci(pool, statistic, failure, n_boot=resamples, seed=seed)
    except ValueError as e:
        return {"auroc": compute_auroc(pool, statistic, failure), "ci_low": nan, "ci_high": nan, "note": f"no CI: {e}"}
    return {"auroc": auroc, "ci_low": low, "ci_high": high, "note": ""}


def check_mean_window_ndcg10_drop_against_reference(runs, reference_values: dict, seeds, decimals: int = 3,
                                                    trials=None) -> pd.DataFrame:
    """Compare the mean window nDCG@10 drop per (failure, severity) with reference values, using
    only the windows of `seeds` (and of `trials`, default all). Raises ValueError listing observed
    vs expected on any mismatch at `decimals`, a missing (failure, severity), or a requested seed
    or trial absent from the table.

    reference_values: {failure label: {severity: expected mean drop}}. The reference values hold
    only for the seeds (and trials) they were computed on (notebook 07: seeds 0-2, trials 0-19), so
    the analysis YAML must record those next to the values. Returns the comparison table.
    """
    windows = one_row_per_window(runs)
    keep = windows["seed"].isin(seeds)
    for name, wanted in (("seeds", seeds), ("trials", trials)):
        missing = sorted(set(wanted) - set(windows[name[:-1]])) if wanted is not None else []
        if missing:
            raise ValueError(f"{name} {missing} are not in the results table")
    if trials is not None:
        keep &= windows["trial"].isin(trials)
    observed = windows[keep].groupby(["failure", "severity"])["ndcg_drop"].agg(["mean", "size"])
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
