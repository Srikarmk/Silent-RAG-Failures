"""Per-window analyses of one runner results table (one run config, one parquet).

The table has one row per window x monitoring statistic. A window is one (failure, severity,
seed, trial); its nDCG@10 drop and p-value repeat on every statistic's row, so per-window
quantities first reduce the table to one row per window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


AUROC_COLUMNS = ["negatives", "auroc", "ci_low", "ci_high", "n_positive", "n_negative",
                 "n_modified_windows_excluded", "n_negative_control_windows_excluded", "note"]


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
    out = [{"severity": key, "negatives": name, **_auroc_and_ci(pool, statistic, failure, counts, resamples, seed), **counts}
           for key, name, pool, counts in _comparisons(runs, statistic, failure, alpha, min_mean_ndcg10_drop, True)]
    return pd.DataFrame(out)[["severity", *AUROC_COLUMNS]]


def auroc_statistic_harmful_vs_negative_windows_by_min_ndcg10_drop(
    runs, statistic: str, failure: str, min_drops, alpha: float, resamples: int, seed: int,
) -> pd.DataFrame:
    """AUROC of a monitoring statistic at each minimum drop, all severities of `failure` pooled:
    the threshold-sensitivity curve. Positives, negatives, excluded windows, the CI and the
    notes are as in auroc_statistic_harmful_vs_negative_windows_by_severity, with every severity
    of the failure in place of one. Pooling across severities weights the result toward the
    severities with more significant windows (usually the higher ones), so it is not an average
    of the per-severity AUROCs.
    """
    out = [{"min_mean_ndcg10_drop": float(min_drop), "negatives": name,
            **_auroc_and_ci(pool, statistic, failure, counts, resamples, seed), **counts}
           for min_drop in min_drops
           for _, name, pool, counts in _comparisons(runs, statistic, failure, alpha, min_drop, False)]
    return pd.DataFrame(out)[["min_mean_ndcg10_drop", *AUROC_COLUMNS]]


def _comparisons(runs, statistic, failure, alpha, min_drop, per_severity: bool):
    """Yield (severity or None, negatives definition, pool, counts) for each comparison, after
    relabelling every window at alpha and min_drop. The pool holds the target windows and the
    negatives; _auc_pool drops the target's non-significant windows itself."""
    rows = runs[runs["detector"].eq(statistic)].copy()
    rows["harmful"] = window_has_significant_ndcg10_drop(
        rows["p_value"].to_numpy(), rows["ndcg_drop"].to_numpy(), alpha, min_drop)
    clean = rows["failure"].eq("clean")
    control = ~rows["expected_harmful"] & ~clean
    target = rows["failure"].eq(failure)
    definitions = {"unmodified": clean}
    if control.any():
        definitions["unmodified_and_negative_controls"] = clean | (control & ~rows["harmful"])
    if per_severity:
        groups = [(float(s), target & rows["severity"].eq(s)) for s in sorted(rows.loc[target, "severity"].unique())]
    else:
        groups = [(None, target)]
    for key, at in groups:
        for name, negatives in definitions.items():
            counts = {
                "n_positive": int((at & rows["harmful"]).sum()),
                "n_negative": int(negatives.sum()),
                "n_modified_windows_excluded": int((at & ~rows["harmful"]).sum()),
                "n_negative_control_windows_excluded": int((control & rows["harmful"]).sum()) if name != "unmodified" else 0,
            }
            yield key, name, rows[at | negatives], counts


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


def spearman_statistic_vs_window_mean_ndcg10_drop(runs, statistic: str, failure: str, resamples: int, seed: int) -> pd.DataFrame:
    """Spearman correlation between a monitoring statistic and the window's mean nDCG@10 drop,
    over the target failure's windows only (no unmodified windows, no labels).

    One row per severity, plus one row with pooled=True (severity NaN) over all severities.
    The pooled rho mixes the statistic's response to severity with its tracking of the drop;
    the per-severity rho isolates tracking of the drop. The CI is a seed-cluster bootstrap:
    whole seeds are resampled with replacement and rho is recomputed on all windows of the drawn
    seeds, so it reflects variability across seeds over the same questions, not question-level
    sampling uncertainty. A constant statistic or constant drop gives NaN plus a note.
    """
    rows = runs[runs["detector"].eq(statistic) & runs["failure"].eq(failure)]
    if rows.duplicated(WINDOW).any():
        raise ValueError(f"more than one {statistic} row per window; is this one run's table?")
    groups = [(float(s), False, rows[rows["severity"].eq(s)]) for s in sorted(rows["severity"].unique())]
    groups.append((float("nan"), True, rows))
    out = [{"severity": severity, "pooled": pooled, **_spearman_and_seed_ci(group, resamples, seed)}
           for severity, pooled, group in groups]
    return pd.DataFrame(out, columns=["severity", "pooled", "rho", "ci_low", "ci_high", "n_windows", "n_seeds", "note"])


def _rho(score, drop) -> float:
    if np.ptp(score) == 0 or np.ptp(drop) == 0:
        return float("nan")
    return float(spearmanr(score, drop).statistic)


def _spearman_and_seed_ci(group, resamples, seed) -> dict:
    nan = float("nan")
    counts = {"n_windows": len(group), "n_seeds": int(group["seed"].nunique())}
    score, drop = group["score"].to_numpy(dtype=float), group["ndcg_drop"].to_numpy(dtype=float)
    for name, values in (("statistic", score), ("drop", drop)):
        if len(values) == 0 or np.ptp(values) == 0:
            return {"rho": nan, "ci_low": nan, "ci_high": nan, **counts, "note": f"undefined: constant {name}"}
    rho = _rho(score, drop)
    if counts["n_seeds"] < 2:
        return {"rho": rho, "ci_low": nan, "ci_high": nan, **counts, "note": "no CI: fewer than two seeds"}
    by_seed = [(g["score"].to_numpy(dtype=float), g["ndcg_drop"].to_numpy(dtype=float)) for _, g in group.groupby("seed")]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(resamples):
        drawn = [by_seed[i] for i in rng.integers(0, len(by_seed), size=len(by_seed))]
        boots.append(_rho(np.concatenate([s for s, _ in drawn]), np.concatenate([d for _, d in drawn])))
    boots = np.asarray(boots)
    boots = boots[~np.isnan(boots)]  # resamples where the drawn seeds are constant
    if len(boots) < 0.8 * resamples:
        return {"rho": rho, "ci_low": nan, "ci_high": nan, **counts, "note": "no CI: too many constant resamples"}
    low, high = np.quantile(boots, [0.025, 0.975])
    return {"rho": rho, "ci_low": float(low), "ci_high": float(high), **counts, "note": ""}


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
    missing_seeds = sorted(set(seeds) - set(windows["seed"]))
    if missing_seeds:
        raise ValueError(f"seeds {missing_seeds} are not in the results table")
    keep = windows["seed"].isin(seeds)
    if trials is not None:
        missing_trials = sorted(set(trials) - set(windows["trial"]))
        if missing_trials:
            raise ValueError(f"trials {missing_trials} are not in the results table")
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
