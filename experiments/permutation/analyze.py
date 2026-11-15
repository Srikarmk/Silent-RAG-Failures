"""Percentile rule vs permutation rule on one runner table (#73). Dev only, NFCorpus.

From the repository root, after the run in README.md:
    python experiments/permutation/analyze.py
    python experiments/permutation/analyze.py --results results/permutation/dev_check.parquet

Clean-max rule: alarm when the plain statistic is at or above its highest clean-window score on
the calibration seeds, the threshold the site and progress update 2 use (reported separately).

Percentile rule: alarm when the plain statistic is above the 95th percentile of its clean-window
scores on the calibration seeds. Permutation rule: alarm when the window's permutation p-value
is below 0.05 (`<detector>_perm` score > 0.95). Both are scored on the other seeds, on the same
windows (silent_rag.eval.permutation.compare_threshold_rules). Two splits: calibrate on 0-2 and
evaluate on 3-5 (A), and the reverse (B).

Windows of distinct queries (clean, H_mixed_truncation, H_full_swap, F1) are the primary
evidence. H_query_topic_shift repeats queries, which breaks the permutation test's
exchangeability assumption, so it is always reported in its own tables.

Writes CSVs and summary.txt next to the results file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from silent_rag.detect.features import FEATURES
from silent_rag.eval.permutation import compare_threshold_rules

ROOT = Path(__file__).resolve().parents[2]
ALPHA = 0.05
SPLITS = {"A": [0, 1, 2], "B": [3, 4, 5]}
REPEATED = {"H_query_topic_shift"}  # controls whose windows repeat queries
PERM_BASES = [f"{p}_{f}" for p in ("ks", "psi") for f in FEATURES] + ["qpp_max", "qpp_gap", "qpp_nqc"]
WINDOW = ["seed", "failure", "severity", "trial"]


def check_permutation_rows(df: pd.DataFrame, permutations: int = 999) -> pd.DataFrame:
    """Every one of the 11 `_perm` detectors scored every window, with p in [1/(B+1), 1]."""
    n_windows = df.groupby(WINDOW).ngroups
    rows = []
    for base in PERM_BASES:
        perm = df[df.detector == base + "_perm"]
        p = 1 - perm.score
        rows.append({
            "detector": base + "_perm",
            "windows": len(perm),
            "all_windows": len(perm) == n_windows,
            "finite": bool(np.isfinite(perm.score).all()),
            "p_in_range": bool(((p >= 1 / (permutations + 1) - 1e-12) & (p <= 1 + 1e-12)).all()),
            "p_min": p.min(),
            "median_cost_ms": 1000 * perm.cost_s.median(),
            "plain_median_cost_ms": 1000 * df[df.detector == base].cost_s.median(),
        })
    return pd.DataFrame(rows)


def per_seed(df: pd.DataFrame, calibration) -> pd.DataFrame:
    """Alarm rates of both rules per evaluation seed, same rule as compare_threshold_rules.
    F1 rows are over all of their windows here, harmful or not (see `harmful_rate`)."""
    out = []
    for base in sorted(d for d in df.detector.unique() if not d.endswith("_perm")):
        if base + "_perm" not in set(df.detector):
            continue
        s = df[df.detector == base].set_index(WINDOW)
        p = df[df.detector == base + "_perm"].set_index(WINDOW)
        w = s[["score", "harmful"]].join(p["score"].rename("score_perm")).reset_index()
        threshold = float(np.quantile(w[(w.failure == "clean") & w.seed.isin(calibration)].score, 1 - ALPHA))
        t = w[~w.seed.isin(calibration)].assign(
            percentile_rule=lambda x: x.score > threshold, permutation_rule=lambda x: x.score_perm > 1 - ALPHA)
        g = t.groupby(["failure", "severity", "seed"]).agg(
            n=("score", "size"), percentile_rule=("percentile_rule", "mean"),
            permutation_rule=("permutation_rule", "mean"), harmful_rate=("harmful", "mean")).reset_index()
        g.insert(0, "detector", base)
        g["threshold"] = threshold
        out.append(g)
    return pd.concat(out, ignore_index=True)


def power_table(rules: pd.DataFrame) -> pd.DataFrame:
    """F1 detection on harmful windows next to each rule's clean false-alarm rate (same split)."""
    clean = rules[rules.kind == "clean"].set_index("detector")[["rate_clean_rule", "rate_permutation"]]
    f1 = rules[rules.kind == "failure"].copy()
    f1 = f1.join(clean.rename(columns=lambda c: "clean_fa_" + c.removeprefix("rate_")), on="detector")
    return f1[["detector", "failure", "severity", "n_windows", "rate_clean_rule", "clean_fa_clean_rule",
               "rate_permutation", "clean_fa_permutation", "seed_range_clean_rule", "seed_range_permutation"]
              ].rename(columns={"rate_clean_rule": "power_percentile", "clean_fa_clean_rule": "clean_fa_percentile",
                                "rate_permutation": "power_permutation", "clean_fa_permutation": "clean_fa_permutation",
                                "n_windows": "n_harmful", "seed_range_clean_rule": "seed_range_percentile"})


def clean_max_rule(df: pd.DataFrame, calibration) -> pd.DataFrame:
    """The threshold the project reports elsewhere (site, progress update 2): the highest plain
    score on the calibration seeds' clean windows, alarm at score >= threshold as in
    detector_scoring. Shown next to the permutation rule on the same evaluation windows.
    Control windows the ground truth marks harmful are dropped here, as false_alarm_rate does;
    `n_harmful_controls_dropped` says how many."""
    out = []
    bases = sorted(d for d in df.detector.unique() if not d.endswith("_perm") and d + "_perm" in set(df.detector))
    for base in bases:
        s = df[df.detector == base].set_index(WINDOW)
        p = df[df.detector == base + "_perm"].set_index(WINDOW)
        w = s[["score", "harmful", "expected_harmful"]].join(p["score"].rename("score_perm")).reset_index()
        threshold = float(w[(w.failure == "clean") & w.seed.isin(calibration)].score.max())
        t = w[~w.seed.isin(calibration)]
        for (failure, severity), g in t.groupby(["failure", "severity"]):
            kind = "clean" if failure == "clean" else ("failure" if g.expected_harmful.iloc[0] else "control")
            keep = g[g.harmful] if kind == "failure" else g[~g.harmful]
            if keep.empty:
                continue
            out.append({"detector": base, "kind": kind, "failure": failure, "severity": severity,
                        "n_windows": len(keep), "n_harmful_controls_dropped": int(g.harmful.sum()) if kind != "failure" else 0,
                        "clean_max_threshold": threshold,
                        "rate_clean_max": float((keep.score >= threshold).mean()),
                        "rate_permutation": float((keep.score_perm > 1 - ALPHA).mean())})
    order = {"clean": 0, "control": 1, "failure": 2}
    return pd.DataFrame(out).sort_values(["detector", "kind", "failure", "severity"],
                                         key=lambda c: c.map(order) if c.name == "kind" else c, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "results" / "permutation" / "dev_check.parquet"))
    args = parser.parse_args()
    results = Path(args.results)
    out_dir = results.parent
    df = pd.read_parquet(results)
    if set(df.dataset) != {"nfcorpus"}:
        sys.exit(f"dev analysis is NFCorpus only, got {sorted(set(df.dataset))}")

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 1000)
    lines = []

    def emit(title, frame):
        text = f"\n== {title} ==\n" + (frame.round(3).to_string(index=False) if isinstance(frame, pd.DataFrame)
                                     else str(frame))
        print(text)
        lines.append(text)

    emit("run", f"run_id {df.run_id.iloc[0]}, {df.groupby(WINDOW).ngroups} windows, "
                f"{df.detector.nunique()} detectors, seeds {sorted(df.seed.unique())}")
    checks = check_permutation_rows(df)
    emit("all 11 permutation detectors", checks)
    checks.to_csv(out_dir / "permutation_detectors.csv", index=False)

    # ground truth on the controls: harmless has to be measured, not assumed
    ctrl = df[(df.detector == "ks_top1") & ~df.failure.isin(["clean"]) & ~df.expected_harmful]
    emit("controls: ground-truth harmful rate (not a detector)", ctrl.groupby(["failure", "severity"]).agg(
        n=("harmful", "size"), harmful_rate=("harmful", "mean"), mean_ndcg_drop=("ndcg_drop", "mean")).reset_index())
    f1 = df[(df.detector == "ks_top1") & df.expected_harmful]
    emit("F1: ground-truth harmful rate", f1.groupby(["failure", "severity"]).agg(
        n=("harmful", "size"), harmful_rate=("harmful", "mean"), mean_ndcg_drop=("ndcg_drop", "mean")).reset_index())

    clean_perm = df[(df.failure == "clean") & df.detector.str.endswith("_perm")].assign(p=lambda x: 1 - x.score)
    emit("clean windows, all seeds: permutation p-values", clean_perm.groupby("detector").p.agg(
        n="size", frac_p_le_05=lambda p: (p <= ALPHA).mean(), frac_p_le_10=lambda p: (p <= 0.10).mean(),
        mean_p="mean").reset_index())

    cols = ["detector", "kind", "failure", "severity", "n_windows", "n_harmful_excluded", "clean_threshold", "rate_clean_rule",
            "rate_permutation", "seed_range_clean_rule", "seed_range_permutation"]
    for split, calibration in SPLITS.items():
        rules = compare_threshold_rules(df, calibration_seeds=calibration, alpha=ALPHA)
        rules.to_csv(out_dir / f"rules_split{split}.csv", index=False)
        seeds = per_seed(df, calibration)
        seeds.to_csv(out_dir / f"per_seed_split{split}.csv", index=False)
        tag = f"split {split}: calibrate on seeds {calibration}"
        distinct = rules[~rules.failure.isin(REPEATED)]
        emit(f"{tag} | distinct-query windows: false alarms (clean, control) and F1 detection (failure)",
             distinct[cols])
        emit(f"{tag} | F1 detection power next to clean false-alarm rate", power_table(rules))
        wide = seeds[~seeds.failure.str.startswith("F1")].pivot_table(
            index=["failure", "severity", "detector"], columns="seed", values=["percentile_rule", "permutation_rule"])
        emit(f"{tag} | false-alarm rate by evaluation seed (clean + controls)",
             wide.round(2).reset_index())
        emit(f"{tag} | REPEATED-QUERY control, kept separate (exchangeability does not hold)",
             rules[rules.failure.isin(REPEATED)][cols])
        cmax = clean_max_rule(df, calibration)
        cmax.to_csv(out_dir / f"clean_max_split{split}.csv", index=False)
        emit(f"{tag} | clean-max rule (max clean score, as on the site) vs permutation, distinct-query windows, "
             "harmful controls dropped", cmax[~cmax.failure.isin(REPEATED)])

    rt = df.groupby("detector").cost_s.agg(["median", "mean", "sum"]).reset_index()
    emit("runtime per window (s)", rt)
    emit("scoring time", f"all detectors {df.cost_s.sum():.1f}s, _perm {df[df.detector.str.endswith('_perm')].cost_s.sum():.1f}s")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote CSVs and summary.txt to {out_dir}")


if __name__ == "__main__":
    main()
