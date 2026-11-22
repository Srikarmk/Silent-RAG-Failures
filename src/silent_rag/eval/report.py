"""The main tables from one results file, as Markdown (#34).

    python -m silent_rag.eval.report results/dev.parquet
    python -m silent_rag.eval.report results/dev.parquet --min-drop 0 0.02 0.06 --out report.md

For each `min_drop` (harmful = p < alpha and drop >= min_drop, recomputed from the
table, so any min_drop works on one run):

1. How often each condition is harmful, by severity.
2. AUROC for every detector and harmful failure, with a 95% CI from resampling seeds
   (`bootstrap_auroc_ci`). Negatives are clean windows plus controls that stayed
   harmless, as in `compute_auroc`. A one-seed run has no CI.
3. Each detector's false alarms on clean windows and on every negative control, and the
   earliest severity it reliably detects, at a threshold set for 5% false alarms on
   clean windows. Leave-one-seed-out (`thresholds.out_of_seed_thresholds`): each seed is
   scored against a threshold set on the other seeds' clean windows, never its own, and
   the table gives the mean over seeds and the worst seed. Needs two or more seeds.

Before those, the cost of one check per detector (median, p90 and max of `cost_s`, and
`cost_usd` per check), which doesn't depend on min_drop. A detector that caches its work
reports what the check takes uncached; NaN means that time wasn't recorded.

Thresholds come from clean windows only, so (3) says how a detector calibrated on clean
traffic behaves, which needs no labels. Measured in-sample, the clean false-alarm rate
would be 5% by construction and the spread between seeds (#91) would vanish; the worst
seed is the number a single deployment should plan for. The `threshold` column is the
one to deploy, set on every seed's clean windows. Use one run_id per report; a table with
several is refused.
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
from silent_rag.eval.thresholds import out_of_seed_thresholds


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
    """False alarms (mean over seeds and worst seed) and earliest detection, each seed
    scored against a threshold set on the other seeds' clean windows."""
    lines = out_of_seed_thresholds(df, rate)
    # alarm = score > threshold, so shifting each row by its seed's threshold lets the
    # single-threshold helpers below apply per-seed thresholds at 0
    shift = pd.Series(list(zip(df.detector.astype(str), df.seed)), index=df.index).map(lines)
    shifted = df.assign(score=df.score - shift)
    controls = sorted(df.loc[~df.expected_harmful & df.failure.ne("clean"), "failure"].unique())
    failures = sorted(df.loc[df.expected_harmful & df.failure.ne("clean"), "failure"].unique())
    rows = []
    for det in sorted(df.detector.unique()):
        d = df[df.detector.eq(det)]
        row = {"detector": det, "threshold": float(np.quantile(d.loc[d.failure.eq("clean"), "score"], 1 - rate))}
        for c in ["clean", *controls]:
            per_seed = []
            for _, g in shifted[shifted.detector.eq(det)].groupby("seed"):
                if c == "clean":
                    w = g[g.failure.eq("clean")]
                    per_seed.append(float((w.score > 0).mean()) if len(w) else np.nan)
                    continue
                try:
                    per_seed.append(false_alarm_rate(g, det, 0.0, failure=c))
                except ValueError:  # no windows of this control, or every one did harm
                    per_seed.append(np.nan)
            known = [r for r in per_seed if not np.isnan(r)]
            row[f"false alarms: {c}"] = float(np.mean(known)) if known else np.nan
            row[f"worst seed: {c}"] = float(max(known)) if known else np.nan
        for f in failures:
            row[f"earliest: {f}"] = earliest_detectable_severity(shifted, det, f, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    def ms(x):
        return x * 1000

    g = df.groupby("detector")
    t = pd.DataFrame({
        "checks": g.size(),
        "median ms": g.cost_s.median().map(ms),
        "p90 ms": g.cost_s.quantile(0.9).map(ms),
        "max ms": g.cost_s.max().map(ms),
        "unknown": g.cost_s.apply(lambda c: int(c.isna().sum())),
        "usd per check": g.cost_usd.mean(),
    }).sort_values("median ms")
    return t.reset_index()


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
    out.append("\n## Cost per check\n")
    out.append(markdown(cost_table(df)))
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
        if d.seed.nunique() < 2:
            out.append("\n### False alarms\n\nNeeds two or more seeds: each seed is scored against "
                       "a threshold set on the other seeds' clean windows.\n")
        else:
            out.append("\n### False alarms and earliest detection, at 5% false alarms on clean windows "
                       "(leave one seed out: mean over seeds and worst seed)\n")
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
