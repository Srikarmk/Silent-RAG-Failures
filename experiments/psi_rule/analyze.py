"""What does PSI > 0.25 mean? (#31) Dev only, NFCorpus.

From the repository root, after the run in README.md:
    python experiments/psi_rule/analyze.py
    python experiments/psi_rule/analyze.py --results results/dev_full.parquet

The rule of thumb: a PSI above 0.25 means significant drift. This checks it against the
ground truth, per failure and per score feature, with the PSI detectors as they are
(`psi_top1`, `psi_gap`, `psi_mean10`, `psi_std10`: ten quantile bins fit on the
reference window, #19).

For each condition (failure x severity):
- mean nDCG@10 drop, and the median and quartiles of PSI over its windows
- harmful failures: the share of their harmful windows with PSI > 0.25 (caught)
- clean windows and negative controls: the share of their harmless windows with
  PSI > 0.25 (false alarms)

And per failure, the nDCG@10 drop at which PSI > 0.25 in half the windows: where the
median PSI crosses 0.25, interpolated between the two severities around it.

Ground truth: a window is harmful when its bootstrap p-value for the nDCG@10 drop is
below HARM_ALPHA = 0.05 and the drop is at least MIN_DROP = 0.02 (#52). It is recomputed
from `p_value` and `ndcg_drop`, so a run stored at another min_drop (dev_full) reads the same.

Writes CSVs, summary.txt and two figures next to the results file. The figures use
greys, markers and line styles only, so they print in black and white (#30).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RULE = 0.25
HARM_ALPHA = 0.05
MIN_DROP = 0.02
DETECTORS = ["psi_top1", "psi_gap", "psi_mean10", "psi_std10"]


def kind(failure: str, expected_harmful: bool) -> str:
    return "clean" if failure == "clean" else "failure" if expected_harmful else "control"


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    if set(df.dataset.unique()) != {"nfcorpus"}:
        raise SystemExit(f"#31 is a dev analysis on NFCorpus only, got {sorted(df.dataset.unique())}")
    df = df[df.detector.isin(DETECTORS)].copy()
    missing = sorted(set(DETECTORS) - set(df.detector))
    if missing:
        raise SystemExit(f"results have no rows for {missing}")
    df["harmful"] = (df.p_value < HARM_ALPHA) & (df.ndcg_drop >= MIN_DROP)
    df["kind"] = [kind(f, e) for f, e in zip(df.failure, df.expected_harmful)]
    df["over"] = df.score > RULE
    return df


def conditions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per detector x condition."""
    rows = []
    for (det, failure, severity), g in df.groupby(["detector", "failure", "severity"]):
        k = g.kind.iloc[0]
        row = {"detector": det, "failure": failure, "severity": severity, "kind": k, "windows": len(g),
               "mean_drop": g.ndcg_drop.mean(), "harmful_share": g.harmful.mean(),
               "psi_q25": g.score.quantile(0.25), "psi_median": g.score.median(),
               "psi_q75": g.score.quantile(0.75), "over_rule": g.over.mean()}
        if k == "failure":
            h = g[g.harmful]
            row["harmful_windows"], row["caught"] = len(h), h.over.mean() if len(h) else np.nan
        else:
            ok = g[~g.harmful]
            row["harmless_windows"], row["false_alarm"] = len(ok), ok.over.mean() if len(ok) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def crossing(table: pd.DataFrame) -> pd.DataFrame:
    """Per detector x failure: the mean nDCG@10 drop where the median PSI reaches RULE,
    interpolated between the severities around it."""
    rows = []
    for (det, failure), g in table[table.kind == "failure"].groupby(["detector", "failure"]):
        g = g.sort_values("severity")
        sev, drop, med = g.severity.to_numpy(), g.mean_drop.to_numpy(), g.psi_median.to_numpy()
        above = np.flatnonzero(med >= RULE)
        if not len(above):
            note, at = f"never: median PSI at most {med.max():.2f} (drop {drop[med.argmax()]:.3f})", np.nan
        elif above[0] == 0:
            note, at = f"already at severity {sev[0]:g} (drop {drop[0]:.3f})", np.nan
        else:
            i = above[0]
            t = (RULE - med[i - 1]) / (med[i] - med[i - 1])
            at = drop[i - 1] + t * (drop[i] - drop[i - 1])
            note = f"between severity {sev[i - 1]:g} and {sev[i]:g}"
        rows.append({"detector": det, "failure": failure, "drop_at_rule": at, "note": note})
    return pd.DataFrame(rows)


def calibrated(path: Path = ROOT / "configs" / "thresholds.json") -> dict[str, float]:
    """The clean 95th-percentile thresholds from #23, for comparison with 0.25."""
    if not path.exists():
        return {}
    return {d: v["threshold"] for d, v in json.loads(path.read_text())["detectors"].items() if d in DETECTORS}


STYLE = {  # greys, markers and line styles only, so it prints in black and white
    "F1_partial_upgrade_bge": dict(color="0.0", marker="^", ls="-", label="F1, bge-small"),
    "F1_partial_upgrade_gte": dict(color="0.25", marker="v", ls="-", label="F1, gte-small"),
    "F1_partial_upgrade_minilm": dict(color="0.0", marker="o", ls="--", label="F1, all-MiniLM-L6"),
    "F1_partial_upgrade_minilm_l12": dict(color="0.25", marker="D", ls="--", label="F1, all-MiniLM-L12"),
    "F2_distractors_arguana": dict(color="0.35", marker="s", ls=":", label="F2, distractors"),
}
CONTROL = dict(color="0.55", marker="x", ls="none", label="harmless controls")
CLEAN = dict(color="0.0", marker="*", ls="none", label="clean")


def plot_conditions(table: pd.DataFrame, out: Path) -> None:
    """Median PSI (bars: quartiles) against mean nDCG@10 drop, one point per condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey=True)
    for ax, det in zip(axes.flat, DETECTORS):
        t = table[table.detector == det]
        for failure, g in t.groupby("failure"):
            g = g.sort_values("severity")
            k = g.kind.iloc[0]
            style = STYLE.get(failure, CLEAN if k == "clean" else CONTROL if k == "control"
                              else dict(color="0.4", marker=".", ls="-", label=failure))
            yerr = [g.psi_median - g.psi_q25, g.psi_q75 - g.psi_median]
            ax.errorbar(g.mean_drop, g.psi_median, yerr=yerr, color=style["color"], marker=style["marker"],
                        ls=style["ls"], ms=6, lw=1, capsize=2, label=style["label"], mfc="white"
                        if style["marker"] in "osD^v" else None)
        ax.axhline(RULE, color="0.0", lw=1.2, ls="-.")
        ax.text(ax.get_xlim()[1], RULE * 1.08, "PSI = 0.25", ha="right", va="bottom", fontsize=8)
        ax.axvline(MIN_DROP, color="0.6", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_title(det, fontsize=10)
        ax.grid(True, which="major", color="0.9", lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel("mean nDCG@10 drop (dotted: min_drop 0.02)")
    for ax in axes[:, 0]:
        ax.set_ylabel("PSI, median and quartiles")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("PSI against nDCG@10 drop per condition, NFCorpus", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_windows(df: pd.DataFrame, out: Path, detector: str = "psi_top1") -> None:
    """Every window: PSI against its own nDCG@10 drop."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df.detector == detector]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for failure, g in d.groupby("failure"):
        k = g.kind.iloc[0]
        style = STYLE.get(failure, CLEAN if k == "clean" else CONTROL)
        hollow = style["marker"] in "osD^v"  # outlined shapes; x and * are line markers, coloured directly
        colors = dict(facecolors="none", edgecolors=style["color"]) if hollow else dict(c=style["color"])
        ax.scatter(g.ndcg_drop, g.score.clip(lower=1e-3), s=6, marker=style["marker"], linewidths=0.5,
                   alpha=0.5, label=style["label"], **colors)
    ax.axhline(RULE, color="0.0", lw=1.2, ls="-.", label="PSI = 0.25")
    ax.axvline(MIN_DROP, color="0.6", lw=0.8, ls=":", label="min_drop 0.02")
    ax.set_yscale("log")
    ax.set_xlabel("nDCG@10 drop of the window")
    ax.set_ylabel(f"PSI ({detector})")
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=7, markerscale=2, frameon=False, loc="lower right")
    ax.set_title(f"{detector} on every window, NFCorpus", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "results" / "psi_rule" / "psi_check.parquet"))
    args = parser.parse_args()
    path = Path(args.results)
    out = path.parent / f"{path.stem}_psi_rule"
    out.mkdir(parents=True, exist_ok=True)

    df = labelled(pd.read_parquet(path))
    table = conditions(df)
    cross = crossing(table)
    table.to_csv(out / "conditions.csv", index=False)
    cross.to_csv(out / "crossing.csv", index=False)
    plot_conditions(table, out / "psi_vs_drop.png")
    plot_windows(df, out / "psi_top1_windows.png")

    pd.set_option("display.width", 220)
    lines = [f"results: {path} (run_id {', '.join(sorted(df.run_id.unique()))}), {df.seed.nunique()} seeds",
             f"rule: PSI > {RULE}; harmful: p < {HARM_ALPHA} and drop >= {MIN_DROP}", ""]
    clean = table[table.kind == "clean"].set_index("detector")
    cal = calibrated()
    lines.append("clean windows: share with PSI > 0.25, and the clean 95th percentile (#23)")
    for det in DETECTORS:
        lines.append(f"  {det:11s} {clean.loc[det, 'false_alarm']:6.1%} of clean windows over 0.25; "
                     f"median clean PSI {clean.loc[det, 'psi_median']:.3f}; "
                     f"95th percentile {cal.get(det, float('nan')):.3f}")
    show = ["detector", "failure", "severity", "windows", "mean_drop", "harmful_share", "psi_median", "caught",
            "false_alarm"]
    lines += ["", "per condition", table[show].round(3).to_string(index=False), "",
              "nDCG@10 drop at which the median PSI reaches 0.25", cross.round(3).to_string(index=False)]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
