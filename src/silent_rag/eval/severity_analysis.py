"""NFCorpus-only diagnostic analysis of retrieval failures by severity (#16).

This module consumes ResultRow tables. It never reads qrels or fits detectors.
The unit of observation is a (run, failure, severity, seed, trial) window,
not a detector row: all detectors see the same window-level nDCG drop.

Intervals
---------
Every interval is built from *seed-level* values: trials inside a seed overlap
(they draw from the same query pool), so each seed is reduced to one number
first. `ndcg_drop` is already a paired difference (clean minus damaged on the
same queries), and adjacent-severity trends pair the same (seed, trial) window
at both severities before averaging per seed.

Two intervals are reported side by side:

- `ci_low` / `ci_high`: Student t interval on the seed means (main analysis).
- `boot_ci_low` / `boot_ci_high`: percentile bootstrap over seed blocks.

With few seeds the percentile bootstrap is usually the narrower of the two.
Neither is known to have its nominal coverage here: the t interval assumes the
seed means are roughly normal, the bootstrap needs more seeds than a typical
run has. Wider is not the same as better calibrated, so read a conclusion that
holds under only one of them as fragile.

Both describe variation across random seeds (windows, reference split and
damage randomization) on the fixed NFCorpus test queries. They are NOT
intervals for a new population of user queries. With one seed no interval is
available.

Several comparisons are made per run (one per adjacent severity pair) and no
multiplicity correction is applied; pass a smaller `alpha` if one is wanted.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

WINDOW_KEY = (
    "run_id", "dataset", "model", "model_revision", "window_size",
    "failure", "severity", "seed", "trial",
)
REQUIRED = set(WINDOW_KEY) | {"detector", "ndcg_drop", "expected_harmful"}


def prepare_windows(results: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate identical ground-truth windows repeated across detectors.

    Reject mixed runs, held-out data, and inconsistent duplicate rows rather
    than silently double-counting experiments or combining incomparable setups.
    """
    missing = REQUIRED.difference(results.columns)
    if missing:
        raise ValueError(f"Missing ResultRow columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("No experiment rows")
    if results["dataset"].nunique(dropna=False) != 1 or results["dataset"].iloc[0] != "nfcorpus":
        raise ValueError("Severity diagnostics are NFCorpus-only; do not use held-out datasets")
    for field in ("run_id", "model", "model_revision", "window_size"):
        if results[field].nunique(dropna=False) != 1:
            raise ValueError(f"Mixed {field}; analyze one comparable run at a time")
    if results[list(WINDOW_KEY) + ["detector"]].isna().any().loc[[
        "run_id", "dataset", "model", "window_size", "failure", "severity", "seed", "trial", "detector"
    ]].any():
        raise ValueError("Missing identifiers in ResultRow data")
    if not is_bool_dtype(results["expected_harmful"]) or results["expected_harmful"].isna().any():
        raise ValueError("expected_harmful must be nonmissing booleans")
    drops = pd.to_numeric(results["ndcg_drop"], errors="coerce").to_numpy(dtype=float)
    sevs = pd.to_numeric(results["severity"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(drops).all() or (np.abs(drops) > 1 + 1e-9).any():
        raise ValueError("ndcg_drop must be finite and in [-1, 1]")
    if not np.isfinite(sevs).all() or ((sevs < 0) | (sevs > 1)).any():
        raise ValueError("severity must be finite and in [0, 1]")
    if results.duplicated(list(WINDOW_KEY) + ["detector"]).any():
        raise ValueError("Duplicate rows for the same detector/window")

    by_window = results.groupby(list(WINDOW_KEY), dropna=False, sort=False)
    for value in ("ndcg_drop", "expected_harmful"):
        if by_window[value].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Different detectors disagree on {value} for the same window")
    if results.groupby("failure")["expected_harmful"].nunique().gt(1).any():
        raise ValueError("expected_harmful changes within a failure")
    is_clean = results["failure"].eq("clean")
    if ((is_clean & (results["severity"] != 0)) |
            (is_clean & results["expected_harmful"]) |
            (~is_clean & (results["severity"] == 0))).any():
        raise ValueError("Clean must have severity 0 and be non-harmful; injections must have severity > 0")

    windows = results.drop_duplicates(subset=list(WINDOW_KEY)).copy()
    return windows.loc[:, list(WINDOW_KEY) + ["expected_harmful", "ndcg_drop"]].reset_index(drop=True)


# F1 at severity 1.0 has every document on the second model: nothing is mixed any
# more, so it is a different regime from the 0.1-0.9 mixing fractions, not the next step.
FULL_REPLACEMENT = 1.0


def _check_boot_args(n_boot: int, alpha: float):
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def _finite_seed_values(seed_values) -> np.ndarray:
    v = np.asarray(seed_values, dtype=float)
    if len(v) == 0 or not np.isfinite(v).all():
        raise ValueError("Expected one or more finite seed means")
    return v


def _t_bounds(v: np.ndarray, tail: float) -> tuple[float, float]:
    """mean -/+ t(1 - tail, n - 1) * standard error; NaN with fewer than two seeds."""
    if len(v) < 2:
        return float("nan"), float("nan")
    from scipy import stats

    half = float(stats.t.ppf(1 - tail, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return float(v.mean() - half), float(v.mean() + half)


def _seed_bootstrap(seed_means, *, n_boot: int, alpha: float, seed: int):
    """Percentile bootstrap over equally weighted seed means; no interval for one seed."""
    v = _finite_seed_values(seed_means)
    mean = float(v.mean())
    if len(v) < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return mean, float(low), float(high)


def _seed_intervals(seed_means, *, n_boot: int, alpha: float, seed: int) -> dict:
    """Mean of the seed values with the t interval (main) and the bootstrap (comparison)."""
    v = _finite_seed_values(seed_means)
    low, high = _t_bounds(v, alpha / 2)
    mean, boot_low, boot_high = _seed_bootstrap(v, n_boot=n_boot, alpha=alpha, seed=seed)
    return {"mean": mean, "ci_low": low, "ci_high": high,
            "boot_ci_low": boot_low, "boot_ci_high": boot_high}


def severity_summary(windows: pd.DataFrame, *, n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Mean window damage by failure/severity with seed-level intervals.

    ci_low/ci_high are the t interval on seed means; boot_ci_* the seed-block
    percentile bootstrap. See the module docstring for what they do and don't cover.
    """
    _check_boot_args(n_boot, alpha)
    rows = []
    grouped = windows.groupby(["failure", "severity", "expected_harmful"], sort=True)
    for i, ((failure, severity, expected_harmful), group) in enumerate(grouped):
        per_seed = group.groupby("seed")["ndcg_drop"].mean().to_numpy(dtype=float)
        ci = _seed_intervals(per_seed, n_boot=n_boot, alpha=alpha, seed=seed + i)
        rows.append({
            "failure": failure, "severity": float(severity),
            "expected_harmful": bool(expected_harmful),
            "n_windows": len(group), "n_seeds": len(per_seed),
            "mean_drop": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "boot_ci_low": ci["boot_ci_low"], "boot_ci_high": ci["boot_ci_high"],
        })
    return pd.DataFrame(rows)


def interval_note(summary: pd.DataFrame) -> str | None:
    """A plain message when intervals are missing because the run has one seed, else None."""
    if summary.empty or int(summary["n_seeds"].min()) >= 2:
        return None
    return (
        "Only one seed in these results, so no confidence interval can be computed: the "
        "empty ci_low/ci_high columns are expected, not a bug. Rerun with `seeds: [0, 1, 2, ...]` "
        "in the config instead of a single `seed:` (configs/dev_full.yaml uses ten)."
    )


def _trend(low: float, high: float) -> str:
    if not (np.isfinite(low) and np.isfinite(high)):
        return "no_interval"
    if low > 0:
        return "increasing"
    if high < 0:
        return "decreasing"
    return "inconclusive"


def adjacent_severity_trends(windows: pd.DataFrame, *, n_boot: int = 2000, alpha: float = 0.05,
                             seed: int = 0,
                             endpoint_severity: float | None = FULL_REPLACEMENT) -> pd.DataFrame:
    """Paired adjacent-severity differences (higher minus lower nDCG drop).

    The same (seed, trial) window is paired at both severities, the differences are
    averaged per seed, and the interval is over those seed-level differences.

    `trend` comes from the t interval, not from the sign of the mean: "increasing"
    when the whole interval is above 0, "decreasing" when it is below, otherwise
    "inconclusive" ("no_interval" with one seed). `boot_trend` is the same call made
    with the bootstrap interval, so disagreements are visible.

    `scope` is "endpoint" for the step that ends at `endpoint_severity` (1.0: the whole
    index replaced, nothing mixed) and "range" for the others. The endpoint is kept in
    the table and the figures but judged apart from the range (monotonicity_report).
    Pass endpoint_severity=None to treat every step alike.

    A "decreasing" step is an observation about the failure, not evidence of a bug in
    its injector. Control rows are descriptive only; control_validation gives controls
    their verdict.
    """
    _check_boot_args(n_boot, alpha)
    rows = []
    for failure, f in windows.loc[windows["failure"] != "clean"].groupby("failure", sort=True):
        levels = sorted(f["severity"].unique())
        for low, high in zip(levels[:-1], levels[1:]):
            a = f.loc[f.severity == low, ["seed", "trial", "ndcg_drop"]]
            b = f.loc[f.severity == high, ["seed", "trial", "ndcg_drop"]]
            matched = a.merge(b, on=["seed", "trial"], suffixes=("_low", "_high"), validate="one_to_one")
            if len(matched) != len(a) or len(matched) != len(b):
                raise ValueError(f"{failure}: severity {low} and {high} lack paired seed/trial windows")
            matched["increase"] = matched["ndcg_drop_high"] - matched["ndcg_drop_low"]
            per_seed = matched.groupby("seed")["increase"].mean().to_numpy(dtype=float)
            ci = _seed_intervals(per_seed, n_boot=n_boot, alpha=alpha, seed=seed + len(rows))
            is_endpoint = endpoint_severity is not None and bool(np.isclose(high, endpoint_severity))
            rows.append({
                "failure": failure, "lower_severity": float(low), "higher_severity": float(high),
                "scope": "endpoint" if is_endpoint else "range",
                "expected_harmful": bool(f["expected_harmful"].iloc[0]),
                "n_paired_windows": len(matched), "n_seeds": len(per_seed),
                "mean_increase": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                "boot_ci_low": ci["boot_ci_low"], "boot_ci_high": ci["boot_ci_high"],
                "trend": _trend(ci["ci_low"], ci["ci_high"]),
                "boot_trend": _trend(ci["boot_ci_low"], ci["boot_ci_high"]),
            })
    return pd.DataFrame(rows)


def monotonicity_report(trends: pd.DataFrame) -> pd.DataFrame:
    """One row per expected-harmful failure: does damage grow across the severity range?

    range_verdict looks only at "range" steps (below the endpoint):
      - "increasing": every step's interval is above 0
      - "non_monotonic": at least one step's interval is below 0
      - "inconclusive": no step decreases but at least one interval includes 0
      - "no_interval": one seed, nothing can be classified
      - "single_step_only": the failure has no range steps
    The step into the endpoint is reported next to it (endpoint_trend), never folded in.

    "non_monotonic" is a description of the measurements. Whether it is a property of
    the failure or a problem with the injector is a separate question for its owner.
    """
    rows = []
    harmful = trends.loc[trends["expected_harmful"]] if len(trends) else trends
    for failure, t in harmful.groupby("failure", sort=True):
        in_range, end = t.loc[t["scope"] == "range"], t.loc[t["scope"] == "endpoint"]
        counts = in_range["trend"].value_counts()
        n_inc, n_dec = int(counts.get("increasing", 0)), int(counts.get("decreasing", 0))
        n_unclear, n_none = int(counts.get("inconclusive", 0)), int(counts.get("no_interval", 0))
        if in_range.empty:
            verdict = "single_step_only"
        elif n_none:
            verdict = "no_interval"
        elif n_dec:
            verdict = "non_monotonic"
        elif n_unclear:
            verdict = "inconclusive"
        else:
            verdict = "increasing"
        rows.append({
            "failure": failure, "n_range_steps": len(in_range),
            "n_increasing": n_inc, "n_decreasing": n_dec, "n_inconclusive": n_unclear,
            "range_verdict": verdict,
            "endpoint_trend": end["trend"].iloc[0] if len(end) else None,
            "endpoint_mean_increase": float(end["mean_increase"].iloc[0]) if len(end) else float("nan"),
        })
    return pd.DataFrame(rows)


def control_validation(windows: pd.DataFrame, *, margin: float,
                       by_construction: Iterable[str] = (), alpha: float = 0.05) -> pd.DataFrame:
    """Does each intended control keep the nDCG drop within an equivalence margin?

    `margin` (absolute nDCG@10) has no default: it is a judgement about what counts as
    practically harmless and has to be chosen and reported explicitly. It need not be
    the runner's `min_drop`, which thresholds single windows rather than a mean.

    One row per (control, severity), each judged on its own, so a control that only
    runs at one severity (a full swap) is validated here even though it has no
    adjacent-severity trend; `n_severities` shows which those are.

    The test is two one-sided t tests at level `alpha` on the seed means, read off the
    bounds mean -/+ t(1 - alpha) * se (lower_bound, upper_bound):
      - "equivalent": the whole interval is inside (-margin, margin), i.e. both
        one-sided tests reject
      - "not_worse": the one-sided upper bound is below margin (degradation of
        `margin` or more is rejected), but the lower bound is not above -margin,
        because nDCG improved or the seeds are too noisy on that side. Not shown flat.
      - "exceeds_margin": lower bound above margin, the control is not harmless
      - "inconclusive": the upper bound reaches margin without the lower bound
        clearing it. Degradation is neither shown nor ruled out; this is NOT
        evidence that the control is harmless.
      - "no_interval": one seed, nothing is claimed either way

    `by_construction` names controls whose drop is 0 by design (a query-traffic shift
    that leaves the index alone). They get evidence "by_construction" and verdict
    "not_tested": a zero there is a property of the design, not a measurement, and
    must not be counted as a passed control. Naming one whose drop is not exactly 0
    is an error. A control with all-zero drops that was *not* named is flagged
    "all_zero_unexplained" rather than passed.
    """
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not np.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be a positive, finite nDCG difference; there is no default")
    if not np.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if isinstance(by_construction, str):
        raise ValueError("by_construction is a collection of failure labels, not one string")
    declared = set(by_construction)
    controls = windows.loc[(windows["failure"] != "clean") & ~windows["expected_harmful"]]
    unknown = declared.difference(controls["failure"])
    if unknown:
        raise ValueError(f"by_construction names labels that are not controls in this run: {sorted(unknown)}")

    rows = []
    for failure, f in controls.groupby("failure", sort=True):
        n_levels = f["severity"].nunique()
        for severity, group in f.groupby("severity", sort=True):
            all_zero = bool((group["ndcg_drop"] == 0).all())
            if failure in declared and not all_zero:
                raise ValueError(f"{failure} was declared zero by construction but has non-zero ndcg_drop")
            per_seed = group.groupby("seed")["ndcg_drop"].mean().to_numpy(dtype=float)
            low, high = _t_bounds(per_seed, alpha)
            if failure in declared:
                evidence, verdict = "by_construction", "not_tested"
            elif all_zero:
                evidence, verdict = "all_zero_unexplained", "not_tested"
            elif not np.isfinite(low):
                evidence, verdict = "measured", "no_interval"
            elif high < margin and low > -margin:
                evidence, verdict = "measured", "equivalent"
            elif high < margin:
                evidence, verdict = "measured", "not_worse"
            elif low > margin:
                evidence, verdict = "measured", "exceeds_margin"
            else:
                evidence, verdict = "measured", "inconclusive"
            rows.append({
                "failure": failure, "severity": float(severity), "n_severities": int(n_levels),
                "n_windows": len(group), "n_seeds": len(per_seed),
                "mean_drop": float(per_seed.mean()), "lower_bound": low, "upper_bound": high,
                "margin": float(margin), "evidence": evidence, "verdict": verdict,
            })
    return pd.DataFrame(rows)


def plot_severity_curves(summary: pd.DataFrame, output_dir: str | Path, *,
                         endpoint_severity: float | None = FULL_REPLACEMENT) -> list[Path]:
    """Save one figure per failure/control. Empty/one-seed CIs are not plotted.

    Error bars are the t interval over seeds. For expected failures the point at
    `endpoint_severity` is drawn hollow and joined by a dashed line: it stays in the
    figure, but it is the fully replaced index, not one more mixing fraction.
    """
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for failure, group in summary.loc[summary.failure != "clean"].groupby("failure", sort=True):
        group = group.sort_values("severity")
        x = group["severity"].to_numpy(dtype=float)
        y = group["mean_drop"].to_numpy(dtype=float)
        lower = group["ci_low"].to_numpy(dtype=float)
        upper = group["ci_high"].to_numpy(dtype=float)
        valid = np.isfinite(lower) & np.isfinite(upper)
        harmful = bool(group.expected_harmful.iloc[0])
        end = np.zeros(len(x), dtype=bool)
        if harmful and endpoint_severity is not None and len(x) > 1:
            end = np.isclose(x, endpoint_severity)
        fig, ax = plt.subplots(figsize=(7, 4))
        (line,) = ax.plot(x[~end], y[~end], marker="o", label="Mean nDCG@10 drop")
        if end.any():
            last = np.flatnonzero(~end)[-1]
            ax.plot([x[last], x[end][0]], [y[last], y[end][0]], linestyle="--", color=line.get_color())
            ax.plot(x[end], y[end], marker="o", linestyle="none", markerfacecolor="white",
                    color=line.get_color(),
                    label=f"Severity {endpoint_severity:g}: fully replaced, reported separately")
        if valid.any():
            ax.errorbar(x[valid], y[valid],
                        yerr=[np.maximum(0, y[valid] - lower[valid]),
                              np.maximum(0, upper[valid] - y[valid])],
                        fmt="none", capsize=3, color=line.get_color(), label="95% t interval over seeds")
        if not valid.all():
            ax.text(0.02, 0.98, "CI unavailable for <2 independent seeds",
                    transform=ax.transAxes, va="top", fontsize=9)
        ax.axhline(0, linestyle="--", linewidth=1, color="grey")
        ax.set_xlabel("Severity (fraction affected)")
        ax.set_ylabel("Clean − damaged nDCG@10")
        ax.set_title(f"{failure} — {'expected failure' if harmful else 'intended control'}")
        ax.legend(loc="best")
        fig.tight_layout()
        fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(failure)).strip("._") or "unnamed"
        path = out / f"severity_{fname}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
