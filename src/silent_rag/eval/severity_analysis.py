"""NFCorpus-only diagnostic analysis of retrieval failures by severity (#16).

This module consumes ResultRow tables. It never reads qrels or fits detectors.
The unit of observation is a (run, failure, severity, seed, trial) window,
not a detector row: all detectors see the same window-level nDCG drop.

Intervals resample entire *seed blocks*, not overlapping trials. They describe
variation across randomizations of these fixed NFCorpus queries, not a CI for
an independent population of user queries. With only one seed, CI is unavailable.
"""

from __future__ import annotations

import re
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


def _seed_bootstrap(seed_means: np.ndarray, *, n_boot: int, alpha: float, seed: int):
    """CI of equally weighted independent-seed means; no false precision for one seed."""
    v = np.asarray(seed_means, dtype=float)
    if len(v) == 0 or not np.isfinite(v).all():
        raise ValueError("Expected one or more finite seed means")
    mean = float(v.mean())
    if len(v) < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return mean, float(low), float(high)


def _check_boot_args(n_boot: int, alpha: float):
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def severity_summary(windows: pd.DataFrame, *, n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Mean window damage by failure/severity with a seed-block percentile CI."""
    _check_boot_args(n_boot, alpha)
    rows = []
    grouped = windows.groupby(["failure", "severity", "expected_harmful"], sort=True)
    for i, ((failure, severity, expected_harmful), group) in enumerate(grouped):
        per_seed = group.groupby("seed")["ndcg_drop"].mean().to_numpy(dtype=float)
        mean, lo, hi = _seed_bootstrap(per_seed, n_boot=n_boot, alpha=alpha, seed=seed + i)
        rows.append({
            "failure": failure, "severity": float(severity),
            "expected_harmful": bool(expected_harmful),
            "n_windows": len(group), "n_seeds": len(per_seed),
            "mean_drop": mean, "ci_low": lo, "ci_high": hi,
        })
    return pd.DataFrame(rows)


def adjacent_severity_trends(windows: pd.DataFrame, *, n_boot: int = 2000,
                             alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Paired adjacent-severity differences (higher minus lower nDCG drop).

    For harmful failures, a negative mean increase is a diagnostic warning.
    For intended harmless controls, 'flat' is *not* certified by a nonsignificant
    p-value: that needs an explicitly chosen equivalence margin.
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
            mean, lo, hi = _seed_bootstrap(per_seed, n_boot=n_boot, alpha=alpha, seed=seed + len(rows))
            harmful = bool(f["expected_harmful"].iloc[0])
            rows.append({
                "failure": failure, "lower_severity": float(low), "higher_severity": float(high),
                "expected_harmful": harmful, "n_paired_windows": len(matched),
                "n_seeds": len(per_seed), "mean_increase": mean,
                "ci_low": lo, "ci_high": hi,
                "diagnostic": ("mean_decreased" if mean < -1e-10 else "mean_not_decreased")
                if harmful else "control_flatness_not_yet_established",
            })
    return pd.DataFrame(rows)


def plot_severity_curves(summary: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save one figure per failure/control. Empty/one-seed CIs are not plotted."""
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
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, y, marker="o", label="Mean nDCG@10 drop")
        if valid.any():
            ax.errorbar(x[valid], y[valid],
                        yerr=[np.maximum(0, y[valid] - lower[valid]),
                              np.maximum(0, upper[valid] - y[valid])],
                        fmt="none", capsize=3, label="95% seed-block CI")
        if not valid.all():
            ax.text(0.02, 0.98, "CI unavailable for <2 independent seeds",
                    transform=ax.transAxes, va="top", fontsize=9)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Severity (fraction affected)")
        ax.set_ylabel("Clean − damaged nDCG@10")
        ax.set_title(f"{failure} — {'expected failure' if bool(group.expected_harmful.iloc[0]) else 'intended control'}")
        ax.legend(loc="best")
        fig.tight_layout()
        fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(failure)).strip("._") or "unnamed"
        path = out / f"severity_{fname}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
