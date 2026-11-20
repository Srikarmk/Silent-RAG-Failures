"""Two ways to turn a drift statistic into an alarm, compared on the same windows (#73).

- clean-window rule (#23): alarm when the statistic is above the (1 - alpha) quantile
  of its scores on clean windows. The quantile comes from the calibration seeds only.
- permutation rule: alarm when the window's own permutation p-value is below alpha.
  Needs no clean windows. In the results table a `<detector>_perm` row holds 1 - p in
  `score`, so the rule is score > 1 - alpha.

Both are evaluated on the seeds that were NOT used for calibration, so the two rates
in a row are over exactly the same windows.

Reading the output:
- on clean windows and on each negative control, a rate is a false-alarm rate over the
  windows the ground truth marks harmless, as in detector_scoring.false_alarm_rate. Windows
  marked harmful are left out of it and counted in `n_harmful_excluded`
- on harmful windows of a failure, it is a detection rate
- either rule fires when the scores changed. A small permutation p is evidence that the
  window differs from the reference, not that retrieval got worse; harm is the
  `harmful` label, which comes from qrels and which no detector sees.

The table's `p_value` column is the ground-truth bootstrap p-value for the nDCG drop.
It has nothing to do with the permutation p-value.

Windows of one seed share a reference window, so their alarms are not independent:
one unusual reference moves every window of that seed. The per-seed range is reported
next to each rate for that reason, and a rate should be judged across seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = ["seed", "failure", "severity", "trial"]
SUFFIX = "_perm"


def compare_threshold_rules(df: pd.DataFrame, calibration_seeds, *, detectors=None,
                            alpha: float = 0.05, harm_alpha: float = 0.05,
                            min_drop: float = 0.0) -> pd.DataFrame:
    """Alarm rates of both rules per detector, on clean windows, each control at each
    severity, and each failure's harmful windows.

    detectors: base names (e.g. "ks_top1"); default is every detector in `df` that has
    a `_perm` partner.

    alpha is the detectors' alarm level: it sets the clean-window quantile and the
    permutation cut-off. harm_alpha and min_drop define the ground truth: a window is
    harmful when its bootstrap `p_value` < harm_alpha and `ndcg_drop` >= min_drop. They
    re-threshold `harmful` from the stored columns, so they need not be the values the
    run was made with, and changing alpha never changes which windows are harmful.
    """
    for name, value in (("alpha", alpha), ("harm_alpha", harm_alpha)):
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    if df["run_id"].nunique() != 1 or df["dataset"].nunique() != 1:
        raise ValueError("compare one run on one dataset at a time")
    calibration = set(calibration_seeds)
    seeds = {int(s) for s in df["seed"].unique()}
    if not calibration or not calibration <= seeds:
        raise ValueError(f"calibration_seeds must be a non-empty subset of the run's seeds {sorted(seeds)}")
    if calibration == seeds:
        raise ValueError("every seed is a calibration seed, none are left to evaluate on")

    names = set(df["detector"].unique())
    if detectors is None:
        detectors = sorted(n for n in names if not n.endswith(SUFFIX) and n + SUFFIX in names)
        if not detectors:
            raise ValueError(f"no detector in these results has a {SUFFIX} partner")
    missing = sorted({d for base in detectors for d in (base, base + SUFFIX)} - names)
    if missing:
        raise ValueError(f"results have no rows for {missing}")

    out = []
    for base in detectors:
        stat = df[df["detector"] == base].set_index(WINDOW)
        perm = df[df["detector"] == base + SUFFIX].set_index(WINDOW)
        if stat.index.has_duplicates or perm.index.has_duplicates or not stat.index.sort_values().equals(
                perm.index.sort_values()):
            raise ValueError(f"{base} and {base}{SUFFIX} were not scored on the same windows")
        wide = stat[["expected_harmful", "p_value", "ndcg_drop", "score"]].join(
            perm["score"].rename("score_perm")).reset_index()

        clean_calibration = wide[(wide["failure"] == "clean") & wide["seed"].isin(calibration)]
        if clean_calibration.empty:
            raise ValueError("no clean windows in the calibration seeds")
        threshold = float(np.quantile(clean_calibration["score"], 1 - alpha))

        test = wide[~wide["seed"].isin(calibration)].copy()
        test["alarm_clean_rule"] = test["score"] > threshold
        test["alarm_permutation"] = test["score_perm"] > 1 - alpha
        test["harmful"] = (test["p_value"] < harm_alpha) & (test["ndcg_drop"] >= min_drop)
        is_clean = test["failure"] == "clean"
        test["kind"] = np.where(is_clean, "clean", np.where(test["expected_harmful"], "failure", "control"))
        # a failure's windows that did no measurable harm are neither false alarms nor misses
        test = test[(test["kind"] != "failure") | test["harmful"]]

        for (kind, failure, severity), group in test.groupby(["kind", "failure", "severity"], sort=True):
            # a false alarm is an alarm on a window that is actually harmless (detector_scoring.false_alarm_rate)
            excluded = 0 if kind == "failure" else int(group["harmful"].sum())
            if excluded:
                group = group[~group["harmful"]]
            by_seed = group.groupby("seed")[["alarm_clean_rule", "alarm_permutation"]].mean()
            out.append({
                "detector": base,
                "kind": kind,  # clean / control: false alarms. failure: detection on harmful windows
                "failure": failure,
                "severity": float(severity),
                "n_windows": len(group),  # the rate's denominator
                "n_harmful_excluded": excluded,
                "n_seeds": len(by_seed),
                "clean_threshold": threshold,
                "rate_clean_rule": float(group["alarm_clean_rule"].mean()),
                "rate_permutation": float(group["alarm_permutation"].mean()),
                "both_alarm": float((group["alarm_clean_rule"] & group["alarm_permutation"]).mean()),
                "seed_range_clean_rule": (float(by_seed["alarm_clean_rule"].min()),
                                          float(by_seed["alarm_clean_rule"].max())),
                "seed_range_permutation": (float(by_seed["alarm_permutation"].min()),
                                           float(by_seed["alarm_permutation"].max())),
            })
    order = {"clean": 0, "control": 1, "failure": 2}
    table = pd.DataFrame(out)
    return table.sort_values(
        ["detector", "kind", "failure", "severity"], key=lambda c: c.map(order) if c.name == "kind" else c,
        kind="stable", ignore_index=True)
