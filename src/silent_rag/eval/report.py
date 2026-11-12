"""The main tables from one results file, as Markdown (#34).

    python -m silent_rag.eval.report results/dev.parquet
    python -m silent_rag.eval.report results/dev.parquet --min-drop 0 0.02 0.06 --out report.md

For each `min_drop` (harmful = p < alpha and drop >= min_drop, recomputed from the
table, so any min_drop works on one run):

1. How often each condition is harmful, by severity.
2. AUROC for every detector and harmful failure, with a 95% CI from resampling seeds
   (`bootstrap_auroc_ci`). Negatives are clean windows plus controls that stayed
   harmless, as in `compute_auroc`. A one-seed run has no CI.
3. Each detector's false-alarm rate on every negative control, at the threshold that
   gives 5% false alarms on this run's clean windows, and the earliest severity it
   reliably detects at that threshold.

The threshold is set on the clean windows of the same run, so (3) says how a detector
calibrated on its own clean traffic behaves, which needs no labels. Use one run_id per
report; a table with several is refused.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from silent_rag.eval.detector_scoring import (
    bootstrap_auroc_ci,
    compute_auroc,
    earliest_detectable_severity,
    false_alarm_rate,
)


def relabel(df: pd.DataFrame, min_drop: float, alpha: float = 0.05) -> pd.DataFrame:
    out = df.copy()
    out["harmful"] = (out.p_value < alpha) & (out.ndcg_drop >= min_drop) & out.failure.ne("clean")
    return out


def harm_table(df: pd.DataFrame) -> pd.DataFrame:
    w = df[df.detector.eq(df.detector.iloc[0])]
    t = w.groupby(["failure", "severity"]).agg(
        expected_harmful=("expected_harmful", "first"), windows=("harmful", "size"),
        mean_drop=("ndcg_drop", "mean"), harmful=("harmful", "mean"))
    return t.reset_index()


def auroc_table(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    failures = sorted(df.loc[df.expected_harmful & df.failure.ne("clean"), "failure"].unique())
    seeds = df.seed.nunique()
    rows = []
    for det in sorted(df.detector.unique()):
        for f in failures:
            try:
                if seeds > 1:
                    auc, lo, hi = bootstrap_auroc_ci(df, det, f, n_boot=n_boot)
                else:
                    auc, lo, hi = compute_auroc(df, det, f), np.nan, np.nan
            except ValueError:  # no harmful windows at this min_drop, or too few to resample
                auc = lo = hi = np.nan
            rows.append({"detector": det, "failure": f, "auroc": auc, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def alarm_table(df: pd.DataFrame, rate: float = 0.05) -> pd.DataFrame:
    controls = sorted(df.loc[~df.expected_harmful & df.failure.ne("clean"), "failure"].unique())
    failures = sorted(df.loc[df.expected_harmful & df.failure.ne("clean"), "failure"].unique())
    rows = []
    for det in sorted(df.detector.unique()):
        d = df[df.detector.eq(det)]
        t = float(np.quantile(d.loc[d.failure.eq("clean"), "score"], 1 - rate))
        row = {"detector": det, "threshold": t}
        for c in controls:
            try:
                row[f"false alarms: {c}"] = false_alarm_rate(df, det, t, failure=c)
            except ValueError:  # every window of this control turned out harmful
                row[f"false alarms: {c}"] = np.nan
        for f in failures:
            row[f"earliest: {f}"] = earliest_detectable_severity(df, det, f, t)
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def report(df: pd.DataFrame, min_drops=(0.0, 0.02, 0.06), n_boot: int = 1000) -> str:
    if df.run_id.nunique() != 1:
        raise SystemExit(f"one run per report, this table has run_ids {sorted(df.run_id.unique())}")
    meta = df.iloc[0]
    out = [f"# Results: run {meta.run_id}",
           f"{', '.join(sorted(df.dataset.unique()))}, model {meta.model}, window {meta.window_size}, "
           f"seeds {sorted(df.seed.unique().tolist())}, {df.trial.nunique()} trials per condition, "
           f"{df.detector.nunique()} detectors."]
    for md in min_drops:
        d = relabel(df, md)
        out.append(f"\n## min_drop {md}\n\n### How often each condition is harmful\n")
        out.append(markdown(harm_table(d)))
        a = auroc_table(d, n_boot)
        cell = a.apply(lambda r: "–" if np.isnan(r.auroc) else
                       f"{r.auroc:.2f}" + ("" if np.isnan(r.lo) else f" [{r.lo:.2f}, {r.hi:.2f}]"), axis=1)
        wide = a.assign(cell=cell).pivot(index="detector", columns="failure", values="cell").reset_index()
        out.append("\n### AUROC (95% seed CI)\n")
        out.append(markdown(wide))
        if (~d.expected_harmful & d.failure.ne("clean")).any():
            out.append("\n### False alarms and earliest detection, at 5% false alarms on clean windows\n")
            out.append(markdown(alarm_table(d)))
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Markdown tables from a results parquet (#34).")
    p.add_argument("results")
    p.add_argument("--min-drop", type=float, nargs="+", default=[0.0, 0.02, 0.06])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", help="write the Markdown here instead of printing it")
    args = p.parse_args()
    text = report(pd.read_parquet(args.results), args.min_drop, args.n_boot)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
