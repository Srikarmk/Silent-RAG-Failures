"""Detector evaluation on the experiment runner's ResultRow table.

Higher scores must mean stronger evidence of retrieval degradation.
Thresholds are supplied externally; they must be calibrated on NFCorpus, not
SciFact/FiQA. All results are conditional on the chosen ground-truth labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from sklearn.metrics import roc_auc_score

_REQUIRED = {
    "run_id", "dataset", "model", "model_revision", "failure",
    "expected_harmful", "severity", "seed", "trial", "window_size",
    "detector", "score", "harmful",
}
_SCOPE = ("dataset", "model", "model_revision", "window_size")


def _selected(df: pd.DataFrame, detector: str, allow_multiple_runs: bool = False) -> pd.DataFrame:
    missing = _REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing ResultRow columns: {sorted(missing)}")

    selected = df.loc[df["detector"].eq(detector)].copy()
    if selected.empty:
        raise ValueError(f"No results for detector {detector!r}")

    for name in _SCOPE:
        if selected[name].nunique(dropna=False) != 1:
            raise ValueError(f"Mixed {name} values; evaluate comparable runs separately")
    if not allow_multiple_runs and selected["run_id"].nunique(dropna=False) != 1:
        raise ValueError("Mixed run_id values; filter a run or explicitly allow multiple runs")

    for name in ("harmful", "expected_harmful"):
        if not is_bool_dtype(selected[name]) or selected[name].isna().any():
            raise ValueError(f"{name} must contain non-missing booleans")

    keys = ("run_id", "seed", "trial", "detector", "failure", "severity")
    if selected.duplicated(list(keys)).any():
        raise ValueError("Duplicate ResultRows could bias the metrics")

    return selected


def _valid_scores(df: pd.DataFrame) -> None:
    if not np.isfinite(df["score"].to_numpy(dtype=float)).all():
        raise ValueError("Detector scores must be finite")


def _threshold(value: float) -> None:
    if not np.isfinite(value):
        raise ValueError("Threshold must be finite")


def _auc_pool(df: pd.DataFrame, detector: str, failure: str,
              allow_multiple_runs: bool = False) -> pd.DataFrame:
    if failure == "clean":
        raise ValueError("Choose a failure, not clean")
    selected = _selected(df, detector, allow_multiple_runs)
    target = selected.loc[selected["failure"].eq(failure)]
    if target.empty:
        raise ValueError(f"No rows for failure {failure!r}")
    if not target["expected_harmful"].all():
        raise ValueError("Target failure must be an expected harmful condition")

    # Positives: harmful windows of the target failure. Negatives: clean windows
    # and controls that turned out harmless. Damaged-but-not-significant target
    # windows are dropped, so AUROC doesn't depend on the bootstrap knife-edge,
    # and harmful controls are failed controls, not positives for F1/F2/etc.
    is_target = selected["failure"].eq(failure)
    is_clean = selected["failure"].eq("clean")
    harmless_control = ~selected["expected_harmful"] & ~is_clean & ~selected["harmful"]
    pool = selected.loc[(is_target & selected["harmful"]) | is_clean | harmless_control].copy()
    pool["harmful"] = pool["failure"].eq(failure)
    _valid_scores(pool)
    if pool["harmful"].nunique() != 2:
        raise ValueError("AUROC requires both harmful and non-harmful windows")
    return pool


def compute_auroc(df: pd.DataFrame, detector: str, failure: str,
                  allow_multiple_runs: bool = False) -> float:
    """AUROC of harmful windows of a failure against clean and harmless control windows."""
    pool = _auc_pool(df, detector, failure, allow_multiple_runs)
    return float(roc_auc_score(pool["harmful"], pool["score"]))


def bootstrap_auroc_ci(
    df: pd.DataFrame,
    detector: str,
    failure: str,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    allow_multiple_runs: bool = False,
) -> tuple[float, float, float]:
    """Percentile CI from resampling (run_id, seed) blocks, conditional on dataset.

    Requires >=2 independent randomization blocks. Multiple run IDs require an
    explicit opt-in AND manual verification their configs differ only by seed.
    This is NOT a query-population CI: blocks can reuse the same corpus/queries.
    A single-seed benchmark run cannot produce this CI.
    """
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    pool = _auc_pool(df, detector, failure, allow_multiple_runs)
    auc = float(roc_auc_score(pool["harmful"], pool["score"]))
    groups = [g for _, g in pool.groupby(["run_id", "seed"], dropna=False)]
    if len(groups) < 2:
        raise ValueError("Bootstrap CI requires at least two experiment blocks/seeds")

    rng = np.random.default_rng(seed)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[i] for i in idx], ignore_index=True)
        if sample["harmful"].nunique() == 2:
            boot_aucs.append(float(roc_auc_score(sample["harmful"], sample["score"])))
    if len(boot_aucs) < n_boot * 0.8:
        raise ValueError("Too few bootstrap samples contain both classes")

    low, high = np.quantile(boot_aucs, [alpha / 2, 1 - alpha / 2])
    return auc, float(low), float(high)


def _controls(df: pd.DataFrame, detector: str,
              failure: str | None = None,
              allow_multiple_runs: bool = False) -> pd.DataFrame:
    selected = _selected(df, detector, allow_multiple_runs)
    controls = selected.loc[
        selected["failure"].ne("clean") & (~selected["expected_harmful"])
    ]
    if failure is not None:
        controls = controls.loc[controls["failure"].eq(failure)]
    if controls.empty:
        raise ValueError("No negative controls for evaluation")
    _valid_scores(controls)
    return controls


def negative_control_report(
    df: pd.DataFrame, detector: str, threshold: float,
    failure: str | None = None, allow_multiple_runs: bool = False,
) -> dict:
    """Report alarms on all intended controls and on actually harmless ones."""
    _threshold(threshold)
    controls = _controls(df, detector, failure, allow_multiple_runs)
    harmless = controls.loc[~controls["harmful"]]
    return {
        "n_controls": len(controls),
        "n_harmful_controls": int(controls["harmful"].sum()),
        "n_harmless_controls": len(harmless),
        "harmful_control_rate": float(controls["harmful"].mean()),
        "control_alarm_rate": float(controls["score"].ge(threshold).mean()),
        "false_alarm_rate": (
            float(harmless["score"].ge(threshold).mean())
            if len(harmless) else None
        ),
    }


def false_alarm_rate(df: pd.DataFrame, detector: str, threshold: float,
                     failure: str | None = None,
                     allow_multiple_runs: bool = False) -> float:
    """False alarms among controls empirically labeled non-harmful."""
    report = negative_control_report(df, detector, threshold, failure, allow_multiple_runs)
    if report["false_alarm_rate"] is None:
        raise ValueError("No valid negative controls for false-alarm evaluation")
    return report["false_alarm_rate"]


def severity_detection_table(
    df: pd.DataFrame, detector: str, failure: str, threshold: float,
    allow_multiple_runs: bool = False,
) -> pd.DataFrame:
    """Per-severity damage prevalence, true detections, and alert rate."""
    _threshold(threshold)
    selected = _selected(df, detector, allow_multiple_runs)
    if failure == "clean":
        raise ValueError("Earliness is defined for a failure, not clean")
    rows = selected.loc[selected["failure"].eq(failure)]
    if rows.empty:
        raise ValueError(f"No rows for failure {failure!r}")
    _valid_scores(rows)
    result = []
    for severity, group in rows.groupby("severity", sort=True):
        harmful = group["harmful"]
        alarm = group["score"].ge(threshold)
        n_harmful = int(harmful.sum())
        true_alarms = int((harmful & alarm).sum())
        result.append({
            "severity": float(severity),
            "n_windows": len(group),
            "n_harmful": n_harmful,
            "harmful_fraction": float(harmful.mean()),
            "true_detection_rate": true_alarms / len(group),
            "tpr_when_harmful": true_alarms / n_harmful if n_harmful else np.nan,
            "alarm_rate": float(alarm.mean()),
        })
    return pd.DataFrame(result)


def earliest_detectable_severity(
    df: pd.DataFrame,
    detector: str,
    failure: str,
    threshold: float,
    min_detection_rate: float = 0.8,
    min_harmful_windows: int = 5,
    allow_multiple_runs: bool = False,
) -> float | None:
    """Earliest severity from which the detector catches harmful windows and keeps doing so.

    A severity qualifies when it has >= min_harmful_windows harmful windows and
    tpr_when_harmful >= min_detection_rate. The answer is the lowest qualifying
    severity that every higher severity with enough harmful windows also meets.
    An alarm on a non-harmful window is NOT a successful early detection.

    Uses tpr_when_harmful, not true_detection_rate: the latter can't exceed
    harmful_fraction, so it would describe the failure rather than the detector.
    """
    if not np.isfinite(min_detection_rate) or not 0 < min_detection_rate <= 1:
        raise ValueError("min_detection_rate must be in (0, 1]")
    if (not isinstance(min_harmful_windows, int)
            or isinstance(min_harmful_windows, bool) or min_harmful_windows < 1):
        raise ValueError("min_harmful_windows must be a positive integer")
    table = severity_detection_table(df, detector, failure, threshold, allow_multiple_runs)
    table = table.loc[table["n_harmful"] >= min_harmful_windows]
    caught = table["tpr_when_harmful"].ge(min_detection_rate).to_numpy()
    # stays caught: caught here and at every higher severity
    stays = np.logical_and.accumulate(caught[::-1])[::-1]
    return float(table["severity"].iloc[stays.argmax()]) if stays.any() else None
