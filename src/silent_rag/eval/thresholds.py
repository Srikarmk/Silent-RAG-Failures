"""Default alarm thresholds (#23): for each detector, the score that 5% of clean NFCorpus
windows exceed.

    python -m silent_rag.eval.thresholds results/dev_full.parquet --seeds 0,1,2,3,4,5,6 \\
        --out configs/thresholds.json

A threshold is the (1 - false_alarm) quantile of the detector's scores on clean windows
from the calibration seeds, never on failure or control windows. It is the same rule as
the two-stage evaluation (#78) and the combined detector (#80), with the same split:
seeds 0-6 calibrate, 7-9 are held out. A window alarms when its score is above the
threshold, so a detector whose clean windows all score 0 (re_embed) gets threshold 0
and never alarms on a clean index.

The 5% holds on average over deployments, not for each one. Every seed fits its detectors
on its own reference window of 100 queries, and for most detectors 30-90% of the clean
score's spread is between seeds. Calibrating on 9 seeds and testing the 10th gives 5-10%
on average but up to 45-75% on an unlucky seed, so the file also records that check
(`false_alarm_by_seed`).

Thresholds only mean something for the setup they were calibrated on: the file records
the model, revision and window size, and `load` refuses a mismatch. NFCorpus only: a
results table with any other dataset is refused, so nothing here touches SciFact or FiQA.
Clean windows depend only on (seed, trial), so any run of a config gives the same clean
scores, whatever failures it also lists.

These are the NFCorpus defaults (the monitoring guide's thresholds and the dev figures),
not the held-out run's: raw scores like qpp_max don't carry over to another dataset, so
the SciFact and FiQA runs calibrate on their own clean windows (#29). Don't copy
configs/thresholds.json into final.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import json

FALSE_ALARM = 0.05
WINDOW = ["run_id", "dataset", "seed", "failure", "severity", "trial"]
SETUP = ("model", "model_revision", "window_size")


def _one(df: pd.DataFrame, column: str):
    values = df[column].unique()
    if len(values) != 1:
        raise ValueError(f"results mix {column} values {sorted(map(str, values))}; calibrate one setup at a time")
    v = values[0]
    return None if pd.isna(v) else v.item() if hasattr(v, "item") else v


def _clean(df: pd.DataFrame, seeds) -> pd.DataFrame:
    return df[df.failure.eq("clean") & df.seed.isin(seeds)]


def calibrate(df: pd.DataFrame, seeds, false_alarm: float = FALSE_ALARM) -> dict:
    """Threshold per detector from the clean windows of `seeds`, plus the setup it holds for."""
    if not 0 < false_alarm < 1:
        raise ValueError(f"false_alarm has to be in (0, 1), got {false_alarm}")
    if set(df.dataset.unique()) != {"nfcorpus"}:
        raise ValueError(f"thresholds come from NFCorpus only, results have {sorted(df.dataset.unique())}")
    seeds = sorted(int(s) for s in seeds)
    missing = sorted(set(seeds) - set(df.seed.unique()))
    if missing:
        raise ValueError(f"results have no windows for calibration seeds {missing}")
    if not set(df.seed.unique()) - set(seeds):
        raise ValueError("calibration seeds have to leave some seeds held out")
    clean = _clean(df, seeds)
    if clean.duplicated(WINDOW + ["detector"]).any():
        raise ValueError("a detector scored the same clean window twice")
    detectors = {}
    for name, g in clean.groupby("detector"):
        scores = g.score.to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError(f"{name} has non-finite scores on clean windows")
        detectors[str(name)] = {"threshold": float(np.quantile(scores, 1 - false_alarm)), "clean_windows": len(scores)}
    return {
        "dataset": "nfcorpus",
        **{k: _one(df, k) for k in SETUP},
        "run_id": _one(df, "run_id"),
        "calibration_seeds": seeds,
        "false_alarm": false_alarm,
        "detectors": dict(sorted(detectors.items())),
    }


def alarm_rates(df: pd.DataFrame, thresholds: dict, exclude_seeds=(), alpha: float = 0.05,
                min_drop: float = 0.0) -> pd.DataFrame:
    """Share of windows above each detector's threshold, per condition, on seeds outside
    `exclude_seeds`. On clean windows and negative controls these are false alarms; on a
    harmful failure, the share of its harmful windows caught.

    `harmful` is recomputed from p_value and ndcg_drop, as in #78 and #80, so any
    min_drop can be used. A control window that did harm isn't counted as a false alarm,
    as in detector_scoring.false_alarm_rate: a harmful failure keeps its harmful windows,
    a control its harmless ones."""
    rows = df[~df.seed.isin(exclude_seeds) & df.detector.isin(thresholds)].copy()
    rows["alarm"] = rows.score > rows.detector.map(thresholds)
    harmful = (rows.p_value < alpha) & (rows.ndcg_drop >= min_drop)
    keep = rows.failure.eq("clean") | (rows.expected_harmful & harmful) | (~rows.expected_harmful & ~harmful)
    return rows[keep].pivot_table(index=["failure", "severity"], columns="detector", values="alarm", aggfunc="mean")


def leave_one_seed_out(df: pd.DataFrame, false_alarm: float = FALSE_ALARM) -> pd.DataFrame:
    """For each seed: calibrate on the clean windows of every other seed, then take the
    false-alarm rate on this seed's clean windows. Mean and worst seed per detector.

    Each seed fits its detectors on its own reference window, and much of a clean score's
    spread is between seeds, not within one. So a threshold holds the target rate on
    average over deployments but not for each one; the worst seed shows by how much."""
    clean = df[df.failure.eq("clean")]
    out = {}
    for name, g in clean.groupby("detector"):
        rates = []
        for seed in sorted(g.seed.unique()):
            line = np.quantile(g[g.seed != seed].score, 1 - false_alarm)
            rates.append(float((g[g.seed == seed].score > line).mean()))
        out[str(name)] = {"mean": float(np.mean(rates)), "worst_seed": float(max(rates))}
    return pd.DataFrame(out).T


def save(path: Path, calibrated: dict) -> None:
    out = {"generated_by": "python -m silent_rag.eval.thresholds (#23); don't edit by hand", **calibrated}
    Path(path).write_text(json.dumps(out, indent=2) + "\n")


def load(path: Path, model: str | None = None, model_revision: str | None = None,
         window_size: int | None = None) -> dict[str, float]:
    """detector name -> threshold. Pass the setup you'll use them on to check it matches."""
    saved = json.loads(Path(path).read_text())
    for key, want in zip(SETUP, (model, model_revision, window_size)):
        if want is not None and saved[key] != want:
            raise ValueError(f"thresholds were calibrated with {key} {saved[key]!r}, not {want!r}")
    return {name: d["threshold"] for name, d in saved["detectors"].items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="a results parquet from silent_rag.run")
    parser.add_argument("--seeds", required=True, help="comma-separated calibration seeds; the rest are held out")
    parser.add_argument("--false-alarm", type=float, default=FALSE_ALARM)
    parser.add_argument("--alpha", type=float, default=0.05, help="for the harmful label in the alarm-rate table")
    parser.add_argument("--min-drop", type=float, default=0.0, help="for the harmful label in the alarm-rate table")
    parser.add_argument("--out", help="write the thresholds here (e.g. configs/thresholds.json)")
    args = parser.parse_args()

    df = pd.read_parquet(args.results)
    seeds = [int(s) for s in args.seeds.split(",")]
    calibrated = calibrate(df, seeds, args.false_alarm)
    thresholds = {name: d["threshold"] for name, d in calibrated["detectors"].items()}
    rates = alarm_rates(df, thresholds, exclude_seeds=seeds, alpha=args.alpha, min_drop=args.min_drop)
    for name, rate in rates.loc[("clean", 0.0)].items():
        calibrated["detectors"][name]["held_out_false_alarm"] = round(float(rate), 4)
    for name, row in leave_one_seed_out(df, args.false_alarm).iterrows():
        calibrated["detectors"][name]["false_alarm_by_seed"] = {
            "mean": round(float(row["mean"]), 4), "worst_seed": round(float(row["worst_seed"]), 4)}
    pd.set_option("display.width", 200)
    print(pd.json_normalize(list(calibrated["detectors"].values())).set_index(
        pd.Index(list(calibrated["detectors"]))).to_string())
    print(f"\nalarm rates on held-out seeds {sorted(set(df.seed.unique()) - set(seeds))}:")
    print(rates.round(2).T.to_string())
    if args.out:
        save(args.out, calibrated)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
