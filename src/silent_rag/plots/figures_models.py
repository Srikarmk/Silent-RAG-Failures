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
    require_rows,
)
from silent_rag.plots.style import FIGURE_DEFAULTS, PlotStyle, color_for, display_name, point_labels, style_context

DOSE_SCALES = ("drop", "ndcg", "percent_of_baseline")
FIGURE_DEFAULTS["dose_response_ndcg10_drop"] = {"scale": "drop", "annotate_ends": True}
MODEL_Y = {  # y option -> (value, CI low, CI high) columns of table 16
    "shared": ("mean_topk_shared", "topk_shared_ci_low", "topk_shared_ci_high"),
    "switched_ndcg10": ("switched_ndcg10", "switched_ndcg10_ci_low", "switched_ndcg10_ci_high"),
}
FIGURE_DEFAULTS["replacement_model_difference"] = {"y": "shared"}
ERROR_BARS = "error bars: 95% CI over questions"


def _one_value(table, column: str) -> float:
    """The single value of a column that must be the same in every row (e.g. baseline_ndcg10)."""
    values = table[column].unique()
    if len(values) != 1:
        raise ValueError(f"{column} must be the same in every row, got {sorted(values)}")
    return float(values[0])


def replacement_model_difference(models, *, y=None, style=None, ax=None, **shared):
    """How far each replacement model is from the baseline, from table 16
    (replacement_model_difference_table): one point per model at the mean cosine between its and
    the baseline's document vectors, named next to the point (label_offsets move the names).

    y="shared": top-k documents shared with the baseline per question (mean), with reference
    lines "all k shared" and "identical vectors" (mean cosine 1). y="switched_ndcg10": nDCG@10
    when questions and documents both use the replacement model, with a line at the baseline
    nDCG@10. Error bars are 95% CIs over questions. Points are ordered by mean cosine (descending)
    unless order.labels says otherwise; both axes start at 0 unless axes.xlim / axes.ylim are set.
    """
    name = "replacement_model_difference"
    settings = (style or PlotStyle.default()).resolve(name, y=y, **shared)
    y = settings["y"]
    if y not in MODEL_Y:
        raise ValueError(f"unknown y {y!r}; choose one of {list(MODEL_Y)}")
    table = filter_include(models, settings)
    require_rows(models, table, settings, name)
    k, baseline = int(_one_value(models, "k")), _one_value(models, "baseline_ndcg10")
    value, low, high = MODEL_Y[y]
    labels = label_order(table["label"], settings, models)
    rows = table.set_index("label").loc[labels]

    with style_context(settings):
        fig, ax = figure_axes(settings, ax)
        lines = settings["lines"]
        for label, row in rows.iterrows():
            ax.errorbar(row["mean_cosine"], row[value], yerr=[[row[value] - row[low]], [row[high] - row[value]]],
                        fmt="o", color=color_for(label, settings, labels), markersize=lines["marker_size"],
                        elinewidth=lines["width"], capsize=lines["marker_size"] / 2, label="_nolegend_")
        point_labels(ax, rows["mean_cosine"], rows[value], labels, settings)
        if y == "shared":
            reference_line(ax, settings, label=f"all {k} shared", y=k)
            reference_line(ax, settings, label="identical vectors (mean cosine 1)", x=1)
        else:
            reference_line(ax, settings, label=f"baseline nDCG@10 = {baseline:.3f}", y=baseline)
        ax.errorbar([np.nan], [np.nan], yerr=[[np.nan], [np.nan]], fmt="o", color=settings["colors"]["reference_grey"],
                    markersize=lines["marker_size"], label=ERROR_BARS)  # legend entry only
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)  # a count from 0 to k, or nDCG@10; the top stays as autoscaled
        finish_axes(ax, settings, xlabel="Mean cosine similarity of document vectors, replacement vs baseline", **{
            "shared": dict(title=f"Top-{k} documents shared with the baseline vs mean cosine",
                           ylabel=f"Top-{k} documents shared with baseline"),
            "switched_ndcg10": dict(title="nDCG@10 of the fully switched system vs mean cosine",
                                    ylabel="nDCG@10, fully switched system"),
        }[y])
    return fig


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
    table = filter_include(drops, settings)
    require_rows(drops, table, settings, "dose_response_ndcg10_drop")
    baseline = _one_value(drops, "baseline_ndcg10")
    convert = {"drop": lambda d: d, "ndcg": lambda d: baseline - d,
               "percent_of_baseline": lambda d: 100 * (baseline - d) / baseline}[scale]
    x_unit = 100 if scale == "percent_of_baseline" else 1
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
