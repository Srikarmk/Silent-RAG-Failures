"""Figures about the replacement models: whole-test-set nDCG@10 drop by severity (table 9) and the
model differences (table 16). Spec section 5.13. Each function reads tables only, draws one
axes and returns its Figure; options default to None here, their defaults are in FIGURE_DEFAULTS.
"""
from __future__ import annotations

import numpy as np

from silent_rag.plots._common import (
    band_legend_entry,
    end_labels,
    figure_axes,
    filter_include,
    finish_axes,
    label_order,
    line_with_band,
    reference_curve,
    reference_line,
)
from silent_rag.plots.style import FIGURE_DEFAULTS, PlotStyle, color_for, display_name, style_context

DOSE_SCALES = ("drop", "ndcg", "percent_of_baseline")
FIGURE_DEFAULTS["dose_response_ndcg10_drop"] = {"scale": "drop", "annotate_ends": True}


def dose_response_ndcg10_drop(drops, *, scale=None, annotate_ends=None, models=None, style=None, ax=None, **shared):
    """Mean nDCG@10 drop over all test questions vs severity, one line per label, from table 9
    (mean_ndcg10_drop_all_test_questions_by_severity); the band is its 95% CI over questions.

    scale: "drop" (the drop, reference line at 0), "ndcg" (baseline_ndcg10 - drop, reference line
    at the clean baseline) or "percent_of_baseline" (100 * (baseline - drop) / baseline against the
    percent of document vectors replaced, reference = proportional to the share of vectors kept).
    Every line starts at severity 0 with the clean baseline (drop 0): this point is definitional
    (no vectors replaced), not estimated, so it has no band. annotate_ends names each line at its
    highest severity instead of in the legend. models (table 16) only sets the label order. On the
    ndcg and percent scales the y-axis starts at 0 unless axes.ylim is set. Raises if include
    leaves no rows.
    """
    settings = (style or PlotStyle.default()).resolve(
        "dose_response_ndcg10_drop", scale=scale, annotate_ends=annotate_ends, **shared)
    scale = settings["scale"]
    if scale not in DOSE_SCALES:
        raise ValueError(f"unknown scale {scale!r}; choose one of {list(DOSE_SCALES)}")
    if drops.empty:
        raise ValueError("dose_response_ndcg10_drop: the table has no rows")
    baselines = drops["baseline_ndcg10"].unique()
    if len(baselines) != 1:
        raise ValueError(f"baseline_ndcg10 must be the same in every row, got {sorted(baselines)}")
    baseline = float(baselines[0])
    convert = {"drop": lambda d: d, "ndcg": lambda d: baseline - d,
               "percent_of_baseline": lambda d: 100 * (baseline - d) / baseline}[scale]
    x_unit = 100 if scale == "percent_of_baseline" else 1
    table = filter_include(drops, settings)
    if table.empty:
        active = {k: v for k, v in settings["include"].items() if v is not None}
        raise ValueError(f"dose_response_ndcg10_drop: no rows to plot after include {active}")
    labels = label_order(table["failure"], settings, models)

    with style_context(settings):
        fig, ax = figure_axes(settings, ax)
        ends = []
        for label in labels:
            rows = table[table["failure"].eq(label)].sort_values("severity")
            name, color = display_name("label", label, settings), color_for(label, settings, labels)
            x = x_unit * np.r_[0.0, rows["severity"]]
            y = convert(np.r_[0.0, rows["mean_ndcg10_drop"]])
            a, b = convert(np.r_[np.nan, rows["ci_low"]]), convert(np.r_[np.nan, rows["ci_high"]])  # NaN: no band at 0
            line_with_band(ax, x, y, settings, color=color, label="_nolegend_" if settings["annotate_ends"] else name,
                           low=np.fmin(a, b), high=np.fmax(a, b))
            ends.append((y[-1], name, color))
        if scale == "drop":
            reference_line(ax, settings, label="no drop (clean baseline)", y=0)
        elif scale == "ndcg":
            reference_line(ax, settings, label=f"clean baseline nDCG@10 = {baseline:.3f}", y=baseline)
        else:
            reference_curve(ax, [0, 100], [100, 0], settings, label="proportional to the share of vectors kept")
        band_legend_entry(ax, settings, "shaded: 95% CI over questions")
        x_max = x_unit * float(table["severity"].max())
        ax.set_xlim(0, x_max * (1.25 if settings["annotate_ends"] else 1.02))  # room for end labels
        if scale != "drop":
            ax.set_ylim(bottom=0)  # nDCG@10 and its percent start at 0; the top stays as autoscaled
        severity_text = "Document vectors replaced (%)" if x_unit == 100 else "Severity (fraction of document vectors replaced)"
        finish_axes(ax, settings, xlabel=severity_text, **{
            "drop": dict(title="Mean nDCG@10 drop vs severity", ylabel="Mean nDCG@10 drop over test questions"),
            "ndcg": dict(title="Mean nDCG@10 vs severity", ylabel="Mean nDCG@10 over test questions"),
            "percent_of_baseline": dict(title="Mean nDCG@10 relative to the clean baseline vs vectors replaced",
                                        ylabel=f"Mean nDCG@10, % of clean baseline ({baseline:.3f})"),
        }[scale])
        if settings["annotate_ends"]:
            end_labels(ax, x_max, ends, settings)
    return fig
