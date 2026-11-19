"""Figures about the monitoring statistics: their values on all test questions by severity
(table 10), the share of windows with a significant drop (table 2), the seconds per
evaluation (table 3) and the AUROC by severity (table 4). Spec section 5.13. Each function reads tables only, draws one axes and
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
    nan_note,
    reference_line,
    require_rows,
)
from silent_rag.plots.style import FIGURE_DEFAULTS, PALETTES, PlotStyle, color_for, display_name, style_context

FIGURE_DEFAULTS["statistic_vs_severity_all_test_questions"] = {"statistic": None, "annotate_ends": True}
FIGURE_DEFAULTS["share_of_significant_windows_vs_severity"] = {"min_drop": None, "annotate_ends": True}
FIGURE_DEFAULTS["seconds_per_statistic"] = {"axes": {"log_x": True}}
FIGURE_DEFAULTS["auroc_vs_severity"] = {"statistic": None, "min_drop": None, "negatives": "unmodified"}
FIGURE_DEFAULTS["auroc_vs_min_ndcg10_drop"] = {"statistic": None, "negatives": "unmodified"}
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


def _seed_ci_text(rows):
    """"95% CI over N random seeds (whole seeds resampled)", N from n_seeds of the rows whose
    interval is present ("8–10" when they differ); None when no row has an interval."""
    drawn = rows.loc[rows[["ci_low", "ci_high"]].notna().all(axis=1), "n_seeds"]
    if drawn.empty:
        return None
    low, high = int(drawn.min()), int(drawn.max())
    return f"95% CI over {low if low == high else f'{low}–{high}'} random seeds (whole seeds resampled)"


def _auroc_rows(auroc, name, settings):
    """The rows of an AUROC table (4 or 5) for settings' statistic and negatives definition;
    raises naming the available values when either is missing or not in the table."""
    for key in ("statistic", "negatives"):
        value, available = settings[key], sorted(auroc[key].unique())
        if value not in available:
            given = f"no {key} given" if value is None else f"{key} {value!r} is not in the table"
            raise ValueError(f"{name}: {given}; choose one of {available}")
    return auroc[auroc["statistic"].eq(settings["statistic"]) & auroc["negatives"].eq(settings["negatives"])]


def auroc_vs_severity(auroc, *, statistic=None, min_drop=None, negatives=None, models=None, style=None, ax=None,
                      **shared):
    """AUROC of one monitoring statistic vs severity, one line per label, at one minimum mean drop,
    from table 4 (auroc_statistic_harmful_vs_negative_windows_by_severity): windows of the label
    with a significant drop against the negatives definition (default "unmodified"). The band is
    the seed-cluster 95% CI: whole seeds resampled, so it reflects variability across random
    seeds over the same test questions, not question-level sampling uncertainty.

    statistic and min_drop are required (min_drop matched with np.isclose). Undefined AUROCs and
    missing intervals are left as gaps and counted, with their reasons, in a note in the lower
    left. A reference line marks AUROC 0.5. The y-axis runs from 0 to 1 unless axes.ylim is set.
    models (table 16) only sets the label order. Raises if a selector is missing or not in the
    table, or if include leaves no rows.
    """
    name = "auroc_vs_severity"
    settings = (style or PlotStyle.default()).resolve(name, statistic=statistic, min_drop=min_drop,
                                                      negatives=negatives, **shared)
    chosen, min_drop = _auroc_rows(auroc, name, settings), settings["min_drop"]
    available = sorted(float(v) for v in auroc["min_mean_ndcg10_drop"].unique())
    at_drop = (np.isclose(chosen["min_mean_ndcg10_drop"], min_drop) if min_drop is not None
               else np.zeros(len(chosen), dtype=bool))
    if not at_drop.any():
        given = "no min_drop given" if min_drop is None else f"min_drop {min_drop} is not in the table"
        raise ValueError(f"{name}: {given}; choose one of {available}")
    one = chosen[at_drop]
    table = filter_include(one, settings)
    require_rows(one, table, settings, name)
    labels = label_order(table["failure"], settings, models)
    statistic_name = display_name("statistic", settings["statistic"], settings)
    negatives_name = display_name("negatives", settings["negatives"], settings)

    with style_context(settings):
        fig, ax = figure_axes(settings, ax)
        for label in labels:
            rows = table[table["failure"].eq(label)].sort_values("severity")
            line_with_band(ax, rows["severity"], rows["auroc"], settings, color=color_for(label, settings, labels),
                           label=display_name("label", label, settings), low=rows["ci_low"], high=rows["ci_high"])
        reference_line(ax, settings, label="AUROC 0.5", y=0.5)
        ci_text = _seed_ci_text(table)
        if ci_text is not None:
            band_legend_entry(ax, settings, f"shaded: {ci_text}")
        ax.set_xlim(0, float(table["severity"].max()) * 1.02)
        ax.set_ylim(-0.02, 1.02)  # AUROC 0 and 1 stay off the frame
        finish_axes(ax, settings, xlabel=SEVERITY_TEXT,
                    ylabel=f"AUROC (windows with a significant drop\nvs {negatives_name} windows)",
                    title=f"{statistic_name}:\nAUROC by severity, minimum mean drop {min_drop:g}")
        nan_note(ax, table, "auroc", settings)
    return fig


def auroc_vs_min_ndcg10_drop(auroc, *, statistic=None, negatives=None, models=None, style=None, ax=None, **shared):
    """AUROC of one monitoring statistic vs the minimum mean nDCG@10 drop a window needs to count
    as significant, one line per label with all severities pooled, from table 5
    (auroc_statistic_harmful_vs_negative_windows_by_min_ndcg10_drop). Pooling weights severities
    by their number of significant windows. The band is the seed-cluster 95% CI (whole seeds
    resampled: variability across random seeds over the same test questions, not question-level
    sampling uncertainty).

    statistic is required; negatives defaults to "unmodified". include.min_drops limits the
    x-axis. Undefined AUROCs and missing intervals are left as gaps and counted, with their
    reasons, in a note under the axes. A reference line marks AUROC 0.5. The y-axis runs from 0
    to 1 unless axes.ylim is set. models (table 16) only sets the label order. Raises if a
    selector is missing or not in the table, or if include leaves no rows.
    """
    name = "auroc_vs_min_ndcg10_drop"
    settings = (style or PlotStyle.default()).resolve(name, statistic=statistic, negatives=negatives, **shared)
    chosen = _auroc_rows(auroc, name, settings)
    table = filter_include(chosen, settings)
    require_rows(chosen, table, settings, name)
    labels = label_order(table["failure"], settings, models)
    statistic_name = display_name("statistic", settings["statistic"], settings)
    negatives_name = display_name("negatives", settings["negatives"], settings)

    with style_context(settings):
        fig, ax = figure_axes(settings, ax)
        for label in labels:
            rows = table[table["failure"].eq(label)].sort_values("min_mean_ndcg10_drop")
            line_with_band(ax, rows["min_mean_ndcg10_drop"], rows["auroc"], settings,
                           color=color_for(label, settings, labels), label=display_name("label", label, settings),
                           low=rows["ci_low"], high=rows["ci_high"])
        reference_line(ax, settings, label="AUROC 0.5", y=0.5)
        ci_text = _seed_ci_text(table)
        if ci_text is not None:
            band_legend_entry(ax, settings, f"shaded: {ci_text}")
        ax.set_ylim(-0.02, 1.02)  # AUROC 0 and 1 stay off the frame
        finish_axes(ax, settings, xlabel="Minimum mean nDCG@10 drop for a significant window",
                    ylabel=f"AUROC (windows with a significant drop\nvs {negatives_name} windows)",
                    title=f"{statistic_name}:\nAUROC by minimum mean drop, all severities pooled")
        nan_note(ax, table, "auroc", settings)
    return fig
