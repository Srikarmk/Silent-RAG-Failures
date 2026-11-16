"""Figures about the monitoring statistics: their values on all test questions by severity
(table 10), the share of windows with a significant drop (table 2) and the seconds per
evaluation (table 3). Spec section 5.13. Each function reads tables only, draws one axes and
returns its Figure; options default to None here, their defaults are in FIGURE_DEFAULTS.
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
    reference_line,
    require_rows,
)
from silent_rag.plots.style import FIGURE_DEFAULTS, PALETTES, PlotStyle, color_for, display_name, style_context

FIGURE_DEFAULTS["statistic_vs_severity_all_test_questions"] = {"statistic": None, "annotate_ends": True}
FIGURE_DEFAULTS["share_of_significant_windows_vs_severity"] = {"min_drop": None, "annotate_ends": True}
FIGURE_DEFAULTS["seconds_per_statistic"] = {"axes": {"log_x": True}}
RANGE_BAND = "shaded: range over seeds (not a CI)"
SEVERITY_TEXT = "Severity (fraction of document vectors replaced)"


def statistic_vs_severity_all_test_questions(stats, *, statistic=None, annotate_ends=None, models=None, style=None,
                                             ax=None, **shared):
    """One monitoring statistic computed on all test questions against the validation reference,
    vs severity, one line per label, from table 10
    (statistic_all_test_questions_vs_validation_reference_by_severity). The line is the mean over
    seeds; the band is the range over seeds (min to max), not a CI. These values are not on the
    scale of a window of 100 questions.

    statistic (required) picks the statistic. The table's clean row for it (no vectors replaced)
    is a reference line labelled with its value; it is read before include, so include.labels or
    include.severities never remove it. annotate_ends names each line at its highest severity
    instead of in the legend. models (table 16) only sets the label order. Raises if the statistic
    is missing or not in the table, if it has no single clean row, or if include leaves no rows.
    """
    name = "statistic_vs_severity_all_test_questions"
    settings = (style or PlotStyle.default()).resolve(name, statistic=statistic, annotate_ends=annotate_ends, **shared)
    statistic, available = settings["statistic"], sorted(stats["statistic"].unique())
    if statistic not in available:
        given = "no statistic given" if statistic is None else f"statistic {statistic!r} is not in the table"
        raise ValueError(f"{name}: {given}; choose one of {available}")
    one = stats[stats["statistic"].eq(statistic)]
    is_clean = one["failure"].eq("clean")
    if is_clean.sum() != 1:
        raise ValueError(f"{name}: expected one clean row for {statistic!r}, got {int(is_clean.sum())}")
    clean_value = float(one.loc[is_clean, "mean"].iloc[0])
    table = filter_include(one[~is_clean], settings)
    require_rows(one[~is_clean], table, settings, name)
    labels = label_order(table["failure"], settings, models)
    statistic_name = display_name("statistic", statistic, settings)

    with style_context(settings):
        fig, ax = figure_axes(settings, ax)
        ends = []
        for label in labels:
            rows = table[table["failure"].eq(label)].sort_values("severity")
            text, color = display_name("label", label, settings), color_for(label, settings, labels)
            line_with_band(ax, rows["severity"], rows["mean"], settings, color=color,
                           label="_nolegend_" if settings["annotate_ends"] else text, low=rows["min"], high=rows["max"])
            ends.append((float(rows["mean"].iloc[-1]), text, color))
        reference_line(ax, settings, label=f"clean baseline = {clean_value:.3f}", y=clean_value)
        band_legend_entry(ax, settings, RANGE_BAND)
        x_max = float(table["severity"].max())
        ax.set_xlim(0, x_max * (1.25 if settings["annotate_ends"] else 1.02))  # room for end labels
        finish_axes(ax, settings, title=f"{statistic_name}:\nall test questions vs validation reference, by severity",
                    xlabel=SEVERITY_TEXT, ylabel=statistic_name)
        if settings["annotate_ends"]:
            end_labels(ax, x_max, ends, settings)
    return fig


def share_of_significant_windows_vs_severity(shares, *, min_drop=None, annotate_ends=None, models=None, style=None,
                                             ax=None, **shared):
    """Percent of windows with a significant mean nDCG@10 drop vs severity, one line per label, at
    one minimum mean drop, from table 2 (share_of_windows_with_significant_ndcg10_drop_by_severity).

    min_drop (required) is matched to min_mean_ndcg10_drop with np.isclose. The clean share at that
    minimum drop is a reference line; it is read before include, so include.labels or
    include.severities never remove it. The y-axis runs from 0 to 100% unless axes.ylim is set.
    annotate_ends names each line at its highest severity instead of in the legend. models
    (table 16) only sets the label order. Raises if min_drop is missing or not in the table, if
    there is no single clean row at it, or if include leaves no rows.
    """
    name = "share_of_significant_windows_vs_severity"
    settings = (style or PlotStyle.default()).resolve(name, min_drop=min_drop, annotate_ends=annotate_ends, **shared)
    min_drop, available = settings["min_drop"], sorted(float(v) for v in shares["min_mean_ndcg10_drop"].unique())
    at_drop = (np.isclose(shares["min_mean_ndcg10_drop"], min_drop) if min_drop is not None
               else np.zeros(len(shares), dtype=bool))
    if not at_drop.any():
        given = "no min_drop given" if min_drop is None else f"min_drop {min_drop} is not in the table"
        raise ValueError(f"{name}: {given}; choose one of {available}")
    one = shares[at_drop]
    is_clean = one["failure"].eq("clean")
    if is_clean.sum() != 1:
        raise ValueError(f"{name}: expected one clean row at min_drop {min_drop}, got {int(is_clean.sum())}")
    clean_share = float(one.loc[is_clean, "share_percent"].iloc[0])
    table = filter_include(one[~is_clean], settings)
    require_rows(one[~is_clean], table, settings, name)
    labels = label_order(table["failure"], settings, models)

    with style_context(settings):
        fig, ax = figure_axes(settings, ax)
        ends = []
        for label in labels:
            rows = table[table["failure"].eq(label)].sort_values("severity")
            text, color = display_name("label", label, settings), color_for(label, settings, labels)
            line_with_band(ax, rows["severity"], rows["share_percent"], settings, color=color,
                           label="_nolegend_" if settings["annotate_ends"] else text)
            ends.append((float(rows["share_percent"].iloc[-1]), text, color))
        reference_line(ax, settings, label=f"clean baseline = {clean_share:.1f}%", y=clean_share)
        x_max = float(table["severity"].max())
        ax.set_xlim(0, x_max * (1.25 if settings["annotate_ends"] else 1.02))  # room for end labels
        ax.set_ylim(-2, 102)  # lines at 0% and 100% stay off the frame
        ax.set_yticks(range(0, 101, 20))
        finish_axes(ax, settings, xlabel=SEVERITY_TEXT, ylabel="Windows with a significant drop (%)",
                    title=f"Windows with a significant nDCG@10 drop vs severity:\nminimum mean drop {min_drop:g}")
        if settings["annotate_ends"]:
            end_labels(ax, x_max, ends, settings)
    return fig


def seconds_per_statistic(seconds, *, style=None, ax=None, **shared):
    """Median seconds per evaluation of each monitoring statistic, one dot per statistic on its own
    row, from table 3 (median_seconds_per_statistic_evaluation); each dot is labelled with its number
    of evaluations. Rows follow order.statistics, then the table's order (cheapest first), top to
    bottom. The x-axis is logarithmic by default (axes.log_x). Raises if include leaves no rows.
    """
    name = "seconds_per_statistic"
    settings = (style or PlotStyle.default()).resolve(name, **shared)
    table = filter_include(seconds, settings)
    require_rows(seconds, table, settings, name)
    present = list(table["statistic"])
    first = [s for s in settings["order"]["statistics"] if s in present]
    rows = table.set_index("statistic").loc[first + [s for s in present if s not in first]]
    palette = settings["colors"]["palette"]
    if isinstance(palette, str) and palette not in PALETTES:  # the message color_for gives
        raise ValueError(f"unknown palette {palette!r}; known: {sorted(PALETTES)}, or give a list of colours")
    color = (PALETTES[palette] if isinstance(palette, str) else palette)[0]

    with style_context(settings):
        fig, ax = figure_axes(settings, ax)
        positions = np.arange(len(rows))
        ax.plot(rows["median_seconds"], positions, "o", color=color, markersize=settings["lines"]["marker_size"])
        for x, y, n in zip(rows["median_seconds"], positions, rows["n_evaluations"], strict=True):
            ax.annotate(f"n = {n:,}", (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=settings["fonts"]["annotation"])
        ax.set_yticks(positions, [display_name("statistic", s, settings) for s in rows.index])
        ax.set_ylim(len(rows) - 0.5, -0.5)  # first statistic at the top
        ax.margins(x=0.25)  # room for the n labels right of the dots
        finish_axes(ax, settings, title="Median seconds per evaluation of each statistic",
                    xlabel="Median seconds per evaluation", ylabel="Monitoring statistic")
    return fig
