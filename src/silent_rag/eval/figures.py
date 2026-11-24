"""The report's figures from one results file (#30).

    python -m silent_rag.eval.figures results/dev_full.parquet
    python -m silent_rag.eval.figures results/final.parquet --out results/figures/final

Every number comes from the report's tables (silent_rag.eval.report), so a figure and
its table can't disagree:

- auroc.png: detectors x harmful failures, AUROC in each cell with its 95% seed CI.
  Negatives are clean windows plus controls that stayed harmless, as in compute_auroc
- false_alarms.png: detectors x clean windows and each negative control, the mean
  false-alarm rate over seeds and, in brackets, the worst seed. Each seed is scored
  against a threshold set on the other seeds' clean windows (#91)
- earliness.png: detectors x harmful failures, the lowest severity from which the
  detector catches at least 80% of harmful windows at every higher severity, with the
  same per-seed thresholds
- cost_accuracy.png: median time per check (log scale) against AUROC, one panel for F1
  (mean over its partners) and one per other harmful failure

`harmful` is recomputed at --min-drop (0.02 by default, #52). Greys, numbers in the cells,
markers and line styles only, so every figure reads in black and white.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from silent_rag.eval import report

FAILURE_NAMES = {
    "F1_partial_upgrade_bge": "F1 bge-small",
    "F1_partial_upgrade_gte": "F1 gte-small",
    "F1_partial_upgrade_minilm": "F1 MiniLM-L6",
    "F1_partial_upgrade_minilm_l12": "F1 MiniLM-L12",
    "F2_distractors_arguana": "F2 distractors",
    "H_full_swap_bge": "full swap bge",
    "H_full_swap_minilm": "full swap MiniLM",
    "H_mixed_truncation_512": "mixed truncation",
    "H_query_topic_shift": "topic shift",
    "clean": "clean",
}
FAMILY_MARKERS = {"ks": "o", "psi": "s", "qpp": "^", "domain": "D", "re": "P", "canary": "*", "llm": "X"}


def name(failure: str) -> str:
    return FAILURE_NAMES.get(failure, failure)


def marker(detector: str) -> str:
    return FAMILY_MARKERS.get(detector.split("_")[0], "v")


def tables(df: pd.DataFrame, min_drop: float = 0.02, n_boot: int = 1000) -> dict[str, pd.DataFrame]:
    """The report's tables, reshaped for plotting: one row per detector."""
    if df.run_id.nunique() != 1:
        raise SystemExit(f"one run per set of figures, this table has run_ids {sorted(df.run_id.unique())}")
    d = report.relabel(df, min_drop)
    auc = report.auroc_table(d, n_boot)
    alarms = report.alarm_table(d)
    cost = report.cost_table(df).set_index("detector")
    out = {
        "auroc": auc.pivot(index="detector", columns="failure", values="auroc"),
        "auroc_lo": auc.pivot(index="detector", columns="failure", values="lo"),
        "auroc_hi": auc.pivot(index="detector", columns="failure", values="hi"),
        "cost_ms": cost["median ms"],
    }
    alarms = alarms.set_index("detector")
    for kind, prefix in (("false_alarm", "false alarms: "), ("worst_seed", "worst seed: "), ("earliest", "earliest: ")):
        cols = [c for c in alarms.columns if c.startswith(prefix)]
        out[kind] = alarms[cols].rename(columns=lambda c: c.removeprefix(prefix)).astype(float)
    # rows: best mean AUROC first, the order every heatmap uses
    order = out["auroc"].mean(axis=1).sort_values(ascending=False).index
    for key in ("auroc", "auroc_lo", "auroc_hi", "false_alarm", "worst_seed", "earliest"):
        out[key] = out[key].reindex(order)
    return out


def _heatmap(values: pd.DataFrame, text: pd.DataFrame, title: str, path: Path, vmin: float, vmax: float,
             cmap: str = "Greys", colorbar: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = values.shape
    fig, ax = plt.subplots(figsize=(1.3 * cols + 3.2, 0.42 * rows + 1.6))
    shown = np.ma.masked_invalid(values.to_numpy(dtype=float))
    im = ax.imshow(shown, cmap=matplotlib.colormaps[cmap].with_extremes(bad="white"), vmin=vmin, vmax=vmax,
                   aspect="auto")
    for i in range(rows):
        for j in range(cols):
            v = values.iat[i, j]
            dark = not np.isnan(v) and (v - vmin) / (vmax - vmin) > 0.6
            ax.text(j, i, text.iat[i, j], ha="center", va="center", fontsize=7.5,
                    color="white" if dark else "black")
    ax.set_xticks(range(cols), [name(c) for c in values.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(rows), values.index, fontsize=8)
    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="0.8", lw=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=10)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(colorbar, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_auroc(t: dict, path: Path, title: str) -> None:
    auc, lo, hi = t["auroc"], t["auroc_lo"], t["auroc_hi"]
    text = auc.copy().astype(object)
    for i in range(auc.shape[0]):
        for j in range(auc.shape[1]):
            a = auc.iat[i, j]
            text.iat[i, j] = "–" if np.isnan(a) else f"{a:.2f}" + (
                "" if np.isnan(lo.iat[i, j]) else f"\n[{lo.iat[i, j]:.2f}, {hi.iat[i, j]:.2f}]")
    _heatmap(auc, text, title, path, vmin=0.3, vmax=1.0, colorbar="AUROC (0.5 = chance)")


def plot_false_alarms(t: dict, path: Path, title: str) -> None:
    mean, worst = t["false_alarm"], t["worst_seed"]
    text = mean.copy().astype(object)
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            m, w = mean.iat[i, j], worst.iat[i, j]
            text.iat[i, j] = "–" if np.isnan(m) else f"{m:.0%}\n({w:.0%})"
    _heatmap(mean, text, title, path, vmin=0.0, vmax=1.0, colorbar="false alarms, mean over seeds")


def plot_earliness(t: dict, path: Path, title: str) -> None:
    sev = t["earliest"]
    text = sev.apply(lambda col: col.map(lambda v: "never" if np.isnan(v) else f"{v:g}"))
    # earlier is better, so the earliest severity is darkest. vmin below 0 keeps severity 1
    # light grey, so it doesn't look like "never", which stays white
    _heatmap(1.0 - sev, text, title, path, vmin=-0.25, vmax=1.0)


def plot_cost_accuracy(t: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    auc = t["auroc"]
    f1 = [c for c in auc.columns if c.startswith("F1")]
    panels = ([("F1, mean over its partners", auc[f1].mean(axis=1))] if f1 else []) + [
        (name(c), auc[c]) for c in auc.columns if c not in f1]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2), sharey=True, squeeze=False)
    # a check that took no measurable time can't go on a log axis; it's left out
    cost = t["cost_ms"].reindex(auc.index).where(lambda c: c > 0)
    for ax, (label, y) in zip(axes[0], panels):
        placed: list[tuple[float, float]] = []
        for det in y.sort_values(ascending=False).index:
            if np.isnan(y[det]) or np.isnan(cost[det]):
                continue
            ax.scatter(cost[det], y[det], marker=marker(det), s=40, facecolors="white", edgecolors="black", lw=0.9)
            # labels go top to bottom; one that would sit on a label already placed nearby
            # (within ~4x in cost, close in AUROC) moves down until it's clear
            x, label_y = float(np.log10(cost[det])), float(y[det])
            while any(abs(x - px) < 0.6 and abs(label_y - py) < 0.018 for px, py in placed):
                label_y -= 0.018
            placed.append((x, label_y))
            moved = label_y < y[det] - 1e-9
            ax.annotate(det, (cost[det], y[det]), xytext=(cost[det] * 1.25, label_y), textcoords="data",
                        fontsize=6.5, va="center",
                        arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4) if moved else None)
        ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
        if placed:
            ax.set_xscale("log")
        ax.set_xlabel("median time per check, ms (log scale)", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(True, color="0.9", lw=0.5)
    axes[0][0].set_ylabel("AUROC (dotted: chance)", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="The report's figures from a results parquet (#30).")
    p.add_argument("results")
    p.add_argument("--out", help="folder for the figures (default: results/figures/<results file name>)")
    p.add_argument("--min-drop", type=float, default=0.02)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    df = pd.read_parquet(args.results)
    out = Path(args.out or Path("results") / "figures" / Path(args.results).stem)
    out.mkdir(parents=True, exist_ok=True)
    t = tables(df, args.min_drop, args.n_boot)
    what = (f"{', '.join(sorted(df.dataset.unique()))}, run {df.run_id.iloc[0]}, {df.seed.nunique()} seeds, "
            f"min_drop {args.min_drop}")
    plot_auroc(t, out / "auroc.png", f"AUROC, harmful windows vs clean and harmless windows\n{what}")
    plot_false_alarms(t, out / "false_alarms.png",
                      f"False alarms at 5% on other seeds' clean windows: mean (worst seed)\n{what}")
    plot_earliness(t, out / "earliness.png", f"Earliest severity caught in 80% of harmful windows\n{what}")
    plot_cost_accuracy(t, out / "cost_accuracy.png", f"Cost against AUROC\n{what}")
    for key in ("auroc", "false_alarm", "worst_seed", "earliest"):
        t[key].to_csv(out / f"{key}.csv")
    print(f"wrote 4 figures and their tables to {out}")


if __name__ == "__main__":
    main()
