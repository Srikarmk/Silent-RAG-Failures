"""Two-stage monitoring (#74): a cheap detector raises the alarm, and only then an
expensive check decides whether it stands.

Evaluated offline from one results table, where every detector scored every window:
a window alarms when stage 1 is above its threshold and stage 2 is above its own.
Stage 2's cost is paid only on windows stage 1 flags, so the mean cost per window is

    cost(stage 1) + P(stage 1 fires) * cost(stage 2)

Thresholds come from clean windows of the calibration seeds only (a quantile of their
scores), never from failure or control windows, the same rule as #23.

The point is the change-vs-damage finding: cheap drift detectors fire on harmless
changes. If stage 2 measures quality (canary queries #24, the LLM judge #26), the
pair should keep stage 1's detection and drop its false alarms. An AND of two 5%
thresholds is stricter than either alone, so part of that drop is built in; `evaluate`
also reports stage 1 alone at the pair's own clean false-alarm rate (`stage1_matched`),
which shows whether stage 2 adds information or only raises the bar. Until those exist,
any registered detector can stand in as stage 2, which tests the machinery, not the
idea: re_embed, for one, fires on harmless changes too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# one window of the results table
WINDOW = ["run_id", "dataset", "seed", "failure", "severity", "trial"]


def scores(df: pd.DataFrame, detectors, alpha: float = 0.05, min_drop: float = 0.0) -> pd.DataFrame:
    """One row per window with each detector's score and cost, and `harmful` recomputed."""
    missing = sorted(set(detectors) - set(df.detector))
    if missing:
        raise ValueError(f"results have no rows for {missing}")
    rows = df[df.detector.isin(detectors)]
    if rows.duplicated(WINDOW + ["detector"]).any():
        raise ValueError("a detector scored the same window twice; pivoting would average them")
    score = rows.pivot_table(index=WINDOW, columns="detector", values="score")
    cost = rows.pivot_table(index=WINDOW, columns="detector", values="cost_s").add_prefix("cost_")
    meta = rows.drop_duplicates(WINDOW).set_index(WINDOW)[["expected_harmful", "p_value", "ndcg_drop"]]
    wide = score.join(cost).join(meta).reset_index()
    if wide[list(detectors)].isna().any().any():
        raise ValueError("some windows are missing a detector score")
    wide["harmful"] = (wide.p_value < alpha) & (wide.ndcg_drop >= min_drop)
    return wide


def threshold(wide: pd.DataFrame, detector: str, calibration_seeds, false_alarm: float = 0.05) -> float:
    clean = wide[wide.failure.eq("clean") & wide.seed.isin(calibration_seeds)]
    if clean.empty:
        raise ValueError("no clean windows in the calibration seeds")
    return float(np.quantile(clean[detector], 1 - false_alarm))


def matched_threshold(wide: pd.DataFrame, stage1: str, stage2: str, calibration_seeds,
                      false_alarm: float = 0.05) -> float:
    """Stage 1's threshold at the pair's clean false-alarm rate on the calibration seeds.

    The pair alarms only when both stages do, so on clean windows it fires less often than
    either stage at `false_alarm`. Stage 1 alone at that stricter rate is the fair
    comparison: if the pair still beats it, stage 2 added information."""
    clean = wide[wide.failure.eq("clean") & wide.seed.isin(calibration_seeds)]
    t1 = threshold(wide, stage1, calibration_seeds, false_alarm)
    t2 = threshold(wide, stage2, calibration_seeds, false_alarm)
    rate = float(((clean[stage1] > t1) & (clean[stage2] > t2)).mean())
    return float(np.quantile(clean[stage1], 1 - rate))


def evaluate(wide: pd.DataFrame, stage1: str, stage2: str, calibration_seeds,
             false_alarm: float = 0.05, by_seed: bool = False) -> pd.DataFrame:
    """Alarm rates on the seeds not used for calibration, per condition, for stage 1 alone,
    stage 2 alone, the pair, and stage 1 alone at the pair's clean false-alarm rate
    (`stage1_matched`), plus mean cost per window. On harmful failures read the rates as
    detection over the harmful windows; on clean windows and controls, as false alarms over
    the windows that did no harm (as in `false_alarm_rate`, #90, #91). `by_seed` splits
    every row by test seed, since each seed has its own reference window."""
    t1 = threshold(wide, stage1, calibration_seeds, false_alarm)
    t2 = threshold(wide, stage2, calibration_seeds, false_alarm)
    t1m = matched_threshold(wide, stage1, stage2, calibration_seeds, false_alarm)
    test = wide[~wide.seed.isin(calibration_seeds)].copy()
    if test.empty:
        raise ValueError("calibration_seeds has to leave some seeds for testing")
    test["stage1"] = test[stage1] > t1
    test["stage2"] = test[stage2] > t2
    test["pair"] = test.stage1 & test.stage2
    test["stage1_matched"] = test[stage1] > t1m
    c1, c2 = test[f"cost_{stage1}"], test[f"cost_{stage2}"]
    test["cost_pair"] = c1 + test.stage1 * c2
    keep = test.harmful == test.expected_harmful
    key = (["seed"] if by_seed else []) + ["failure", "severity"]
    out = test[keep].groupby(key).agg(
        expected_harmful=("expected_harmful", "first"), windows=("stage1", "size"),
        stage1=("stage1", "mean"), stage2=("stage2", "mean"), pair=("pair", "mean"),
        stage1_matched=("stage1_matched", "mean"),
        cost_stage1_s=(f"cost_{stage1}", "mean"), cost_stage2_s=(f"cost_{stage2}", "mean"),
        cost_pair_s=("cost_pair", "mean"))
    return out


def summary(table: pd.DataFrame) -> pd.Series:
    """Headline numbers, pooled over windows: detection over every harmful window, false
    alarms over every clean and harmless-control window, and mean cost per window."""
    harmful, harmless = table[table.expected_harmful], table[~table.expected_harmful]

    def pooled(rows, option):
        return float((rows[option] * rows.windows).sum() / rows.windows.sum())

    out = {}
    for option in ("stage1", "stage2", "pair", "stage1_matched"):
        out[f"{option}_detection"] = pooled(harmful, option)
        out[f"{option}_false_alarm"] = pooled(harmless, option)
    w = table.windows
    for option in ("stage1", "stage2", "pair"):
        out[f"{option}_cost_s"] = float((table[f"cost_{option}_s"] * w).sum() / w.sum())
    return pd.Series(out)


def gated(df: pd.DataFrame, judge_threshold: float, detector: str = "two_stage", alpha: float = 0.05,
          min_drop: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    """Rates for a run of the gated `two_stage` detector, where stage 2 only ran on the
    windows stage 1 flagged and every other window scored `UNJUDGED` (-1).

    Utilization and cost are operational, so they cover every window: at run time nobody
    knows which windows are harmful. `judged` is the share of windows the judge ran on
    (score >= 0; a judged window can score 0) and `cost_s` the mean cost per window.

    Alarm rates use the ground truth. A window alarms when its score is above
    `judge_threshold`, the judge's threshold set on clean windows. On a harmful failure
    the rate is detection over its harmful windows; on clean windows and controls it is
    false alarms over the windows that did no harm, as in `false_alarm_rate` (#90, #91).
    `scored` is that denominator."""
    rows = df[df.detector.eq(detector)].copy()
    if rows.empty:
        raise ValueError(f"results have no rows for {detector!r}")
    rows["harmful"] = (rows.p_value < alpha) & (rows.ndcg_drop >= min_drop)
    rows["judged"] = rows.score >= 0
    rows["alarm"] = rows.score > judge_threshold
    rows["scored"] = rows.harmful == rows.expected_harmful
    key = ["failure", "severity"]
    ops = rows.groupby(key).agg(expected_harmful=("expected_harmful", "first"), windows=("alarm", "size"),
                                judged=("judged", "mean"), cost_s=("cost_s", "mean"))
    scored = rows[rows.scored]
    rates = scored.groupby(key).agg(scored=("alarm", "size"), alarm=("alarm", "mean"))
    table = ops.join(rates)
    table["scored"] = table.scored.fillna(0).astype(int)
    harm, ok = scored[scored.expected_harmful], scored[~scored.expected_harmful]
    summary = pd.Series({"detection": harm.alarm.mean(), "false_alarm": ok.alarm.mean(),
                         "judged": rows.judged.mean(), "cost_s": rows.cost_s.mean(),
                         "harmful_windows": len(harm), "harmless_windows": len(ok), "windows": len(rows)})
    return table, summary
