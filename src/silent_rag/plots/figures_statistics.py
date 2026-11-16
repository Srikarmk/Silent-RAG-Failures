"""Figures about the monitoring statistics: their values on all test questions by severity
(table 10). Spec section 5.13. Each function reads tables only, draws one axes and returns its
Figure; options default to None here, their defaults are in FIGURE_DEFAULTS.
"""
from __future__ import annotations

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
from silent_rag.plots.style import FIGURE_DEFAULTS, PlotStyle, color_for, display_name, style_context

FIGURE_DEFAULTS["statistic_vs_severity_all_test_questions"] = {"statistic": None, "annotate_ends": True}
RANGE_BAND = "shaded: range over seeds (not a CI)"


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
                    xlabel="Severity (fraction of document vectors replaced)", ylabel=statistic_name)
        if settings["annotate_ends"]:
            end_labels(ax, x_max, ends, settings)
    return fig
