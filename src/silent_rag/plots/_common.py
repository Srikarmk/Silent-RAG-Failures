"""Drawing helpers shared by the figure modules (spec section 5.13).

Each helper takes resolved settings (PlotStyle.resolve's dict); the drawing ones draw into the
axes they are given and never save or show a figure (only figure_axes creates one, when no axes
is given). Nothing here computes a statistic: helpers filter, order and draw values that are
already in the tables (decision 12).
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from silent_rag.plots.style import new_figure, shared_legend

# include key -> the table columns it filters, first one present wins (table 16 names labels "label")
INCLUDE_COLUMNS = {"labels": ("failure", "label"), "statistics": ("statistic",),
                   "severities": ("severity",), "min_drops": ("min_mean_ndcg10_drop",)}
NUMERIC_INCLUDE = ("severities", "min_drops")


def filter_include(table: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Rows of `table` allowed by settings["include"]. null = all; a key whose column the table
    lacks is ignored. Severities and minimum drops match with np.isclose; rows with a NaN severity
    (e.g. the pooled Spearman row, which has no severity) are kept."""
    keep = pd.Series(True, index=table.index)
    for key, allowed in settings["include"].items():
        column = next((c for c in INCLUDE_COLUMNS[key] if c in table.columns), None)
        if allowed is None or column is None:
            continue
        values = table[column]
        if key in NUMERIC_INCLUDE:
            numbers = values.to_numpy(dtype=float)
            match = np.isclose(numbers[:, None], np.asarray(allowed, dtype=float)[None, :]).any(axis=1)
            keep &= match | np.isnan(numbers)
        else:
            keep &= values.isin(allowed)
    return table[keep]


def require_rows(table: pd.DataFrame, filtered: pd.DataFrame, settings: dict, figure_name: str) -> None:
    """Raise ValueError when a figure has nothing to draw: "<figure>: the table has no rows" when
    `table` (the input) is empty, or, when rows existed but `filtered` (after filter_include) is
    empty, "<figure>: no rows to plot after include {...}" naming only the include keys that are set."""
    if table.empty:
        raise ValueError(f"{figure_name}: the table has no rows")
    if filtered.empty:
        active = {k: v for k, v in settings["include"].items() if v is not None}
        raise ValueError(f"{figure_name}: no rows to plot after include {active}")


def label_order(labels, settings: dict, models: pd.DataFrame | None = None) -> list:
    """The labels in drawing order: order.labels first (those present), then the rest by mean
    cosine descending when a models table (table 16: label, mean_cosine) is given, else in order
    of first appearance. Labels missing from the models table come last, in appearance order."""
    present = list(dict.fromkeys(labels))
    rest = [label for label in present if label not in settings["order"]["labels"]]
    if models is not None:
        cosine = models.set_index("label")["mean_cosine"]
        known = sorted((label for label in rest if label in cosine.index), key=lambda label: -cosine[label])
        rest = known + [label for label in rest if label not in cosine.index]
    return [label for label in settings["order"]["labels"] if label in present] + rest


def line_with_band(ax, x, y, settings: dict, *, color, label, low=None, high=None, marker="o"):
    """A line with markers (shape `marker`, a matplotlib marker code; figures pass the label's
    marker_for) and, when low and high are given, a shaded band between them (lines.band_alpha).
    Returns (line, band or None). The band carries no legend entry; the caller says what it is
    (CI over questions, CI over seeds, range over seeds)."""
    lines = settings["lines"]
    band = None
    if low is not None and high is not None:
        band = ax.fill_between(x, low, high, color=color, alpha=lines["band_alpha"], linewidth=0)
    (line,) = ax.plot(x, y, color=color, label=label, marker=marker, linestyle="-", linewidth=lines["width"],
                      markersize=lines["marker_size"])
    return line, band


def point_with_interval(ax, x, y, low, high, settings: dict, *, color, marker, hollow=False):
    """One point at (x, y) with a vertical interval bar from low to high, drawn from its two ends
    (not as lengths from y, so an interval that does not contain y draws as it is), with caps at
    both ends and the marker above the bar (zorder 3 over 2). hollow: white fill, edge in `color`,
    edge width lines.width. A NaN low or high draws the marker only. Returns (marker, bar or None).
    The artists carry gid "interval_marker", "interval_bar" and "interval_cap"; none has a legend
    entry (the caller names the bars, e.g. "error bars: 95% CI over questions")."""
    lines = settings["lines"]
    bar = None
    if not (pd.isna(low) or pd.isna(high)):
        bar = ax.vlines(x, low, high, colors=color, linewidth=lines["width"], zorder=2, gid="interval_bar")
        ax.plot([x, x], [low, high], linestyle="none", marker="_", color=color, markersize=lines["marker_size"],
                markeredgewidth=lines["width"], zorder=2, gid="interval_cap", label="_nolegend_")
    fill = {"markerfacecolor": "white", "markeredgecolor": color, "markeredgewidth": lines["width"]} if hollow else {}
    (point,) = ax.plot([x], [y], linestyle="none", marker=marker, color=color, markersize=lines["marker_size"],
                       zorder=3, gid="interval_marker", label="_nolegend_", **fill)
    return point, bar


def reference_line(ax, settings: dict, *, label, y=None, x=None):
    """A horizontal (y=) or vertical (x=) reference line in the reference grey and style, named
    in the legend (e.g. "clean baseline", "AUROC 0.5", "alpha = 0.05")."""
    if (x is None) == (y is None):
        raise ValueError("give exactly one of x and y")
    style = _reference_style(settings, label)
    return ax.axhline(y, **style) if y is not None else ax.axvline(x, **style)


def reference_curve(ax, xs, ys, settings: dict, *, label):
    """A reference line through the points (xs, ys) that is not parallel to an axis (e.g. the
    proportional diagonal), styled and named in the legend like reference_line."""
    (line,) = ax.plot(xs, ys, **_reference_style(settings, label))
    return line


def _reference_style(settings: dict, label) -> dict:
    return dict(color=settings["colors"]["reference_grey"], linestyle=settings["lines"]["reference_line_style"],
                linewidth=settings["lines"]["reference_line_width"], label=label, zorder=1)


def end_labels(ax, x, ends, settings: dict, min_gap: float = 0.06) -> list:
    """Text labels just right of x at each series' last value, as notebook 08's label_ends: ends
    are (y, text, colour); sorted by y, each label is moved up to at least min_gap of the y-axis
    range above the one below it, so they do not overlap. If the top label then comes within half
    a gap of the upper y-limit or passes it, the stack is moved down from the top: the top label
    sits half a gap below the limit (so the whole label is inside) and each label below at most
    min_gap under the one above (labels with room below them keep their place).
    If the labels cannot fit min_gap apart between half a gap above the lower limit and half a gap
    below the upper one, they are spread evenly between those two points. Labels stay in y order. Set the y limits before calling. NaN ends are skipped."""
    low, high = ax.get_ylim()
    gap, kept = (high - low) * min_gap, sorted(e for e in ends if not np.isnan(e[0]))
    placed = []
    for y, _, _ in kept:
        placed.append(max(y, placed[-1] + gap) if placed else y)
    if len(placed) > 1 and (len(placed) - 1) * gap > high - low - gap:  # half a gap kept free at each end
        placed = list(np.linspace(low + gap / 2, high - gap / 2, len(placed)))
    elif placed and placed[-1] > high - gap / 2:
        placed[-1] = high - gap / 2
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - gap)
    return [ax.text(x, y, f" {text}", color=color, fontsize=settings["fonts"]["annotation"], va="center",
                    clip_on=False) for y, (_, text, color) in zip(placed, kept, strict=True)]


def nan_note(ax, table: pd.DataFrame, value_column: str, settings: dict, note_column: str = "note",
             interval_columns=("ci_low", "ci_high"), wrap_width: int | None = None):
    """Write under the axes (below the x-axis label, left-aligned with the axes) what the table
    has that cannot be drawn, with the reasons from the note column, one clause per kind, each
    wrapped to wrap_width characters (None = as many as fit the axes' width, see _note_wrap_width):
    - rows with a NaN value_column: "3 points not shown (value undefined): undefined: no positive windows (3)";
    - rows with a value but a NaN in any of interval_columns: "2 intervals not shown: no CI: one seed (2)".
    Interval columns the table does not have are ignored (as in filter_include), so a table without
    intervals gets no interval clause. None when there is nothing to report.
    The note is an annotation of the axes anchored to the x-axis label, so it never covers data
    or the legend, constrained layout makes room for it, and each ax= panel gets its own."""
    values_missing = table[value_column].isna()
    present = [c for c in (interval_columns or ()) if c in table.columns]
    intervals_missing = ~values_missing & table[present].isna().any(axis=1)
    clauses = [_counted(table[values_missing], note_column, "point", " (value undefined)"),
               _counted(table[intervals_missing], note_column, "interval", "")]
    if wrap_width is None:
        wrap_width = _note_wrap_width(ax, settings["fonts"]["annotation"])
    clauses = [textwrap.fill(c, wrap_width) for c in clauses if c]
    if not clauses:
        return None
    # x in axes fractions (0 = left edge), y at the bottom of the x-axis label's box, then 4 points lower
    return ax.annotate("\n".join(clauses), xy=(0, 0), xycoords=(ax.transAxes, ax.xaxis.label), xytext=(0, -4),
                       textcoords="offset points", ha="left", va="top", annotation_clip=False,
                       fontsize=settings["fonts"]["annotation"], color=settings["colors"]["reference_grey"])


NOTE_CHARACTER_WIDTH = 0.62  # a character's width as a fraction of the font size; conservative for the default font
NOTE_MIN_WRAP_WIDTH = 20


def _note_wrap_width(ax, font_size: float) -> int:
    """Characters per note line that fit the axes' current width: the width in points (its share
    of the figure times the figure width in inches times 72) over 0.62 x the font size, at least
    20. Estimated when the note is written, before constrained layout adjusts the axes."""
    width_points = ax.get_position().width * ax.figure.get_figwidth() * 72
    return max(NOTE_MIN_WRAP_WIDTH, int(width_points / (NOTE_CHARACTER_WIDTH * font_size)))


def _counted(rows: pd.DataFrame, note_column: str, noun: str, qualifier: str) -> str:
    """'<n> <noun>s not shown<qualifier>: <reason> (<count>); ...', or "" when rows is empty."""
    if rows.empty:
        return ""
    reasons = rows[note_column].fillna("no note").replace("", "no note").value_counts()
    because = "; ".join(f"{reason} ({count})" for reason, count in reasons.items())
    return f"{len(rows)} {noun}{'s' if len(rows) != 1 else ''} not shown{qualifier}: {because}"


def figure_axes(settings: dict, ax=None):
    """(fig, ax) to draw into: a new figure from settings["figure"] when ax is None, else ax and
    its own figure (the figure size settings are then not used). Call inside style_context."""
    if ax is None:
        return new_figure(settings)
    return ax.figure, ax


def finish_axes(ax, settings: dict, *, title, xlabel, ylabel):
    """Title, axis labels, limits, log scales and legend from settings["axes"] and
    settings["legend"]. title/xlabel/ylabel are the figure's built-in text, used where the axes
    setting is null. The legend goes on the axes, or beside the figure with legend.outside (via
    shared_legend); none when legend.show is false or nothing is labelled. Returns the legend or
    None. Call end_labels after this: it places labels using the final y limits. Title, axis label
    and tick label sizes are set on the axes from settings["fonts"], so an axes created outside
    style_context (an ax= panel) matches one created inside it."""
    a, fonts = settings["axes"], settings["fonts"]
    ax.set_title(title if a["title"] is None else a["title"], fontsize=fonts["title"])
    ax.set_xlabel(xlabel if a["xlabel"] is None else a["xlabel"], fontsize=fonts["axis_label"])
    ax.set_ylabel(ylabel if a["ylabel"] is None else a["ylabel"], fontsize=fonts["axis_label"])
    ax.tick_params(axis="both", which="both", labelsize=fonts["tick"])
    if a["log_x"]:
        ax.set_xscale("log")
    if a["log_y"]:
        ax.set_yscale("log")
    if a["xlim"] is not None:
        ax.set_xlim(a["xlim"])
    if a["ylim"] is not None:
        ax.set_ylim(a["ylim"])
    legend = settings["legend"]
    if not legend["show"] or not ax.get_legend_handles_labels()[1]:
        return None
    if legend["outside"]:
        return shared_legend(ax.figure, ax, settings)
    return ax.legend(loc=legend["loc"], ncols=legend["ncol"], title=legend["title"], frameon=legend["frame"],
                     fontsize=settings["fonts"]["legend"])


def band_legend_entry(ax, settings: dict, label: str):
    """A grey legend patch at lines.band_alpha naming what every band in the axes shows, e.g.
    "shaded: 95% CI over questions". It draws nothing and leaves the data limits unchanged."""
    return ax.fill_between([], [], [], color=settings["colors"]["reference_grey"], alpha=settings["lines"]["band_alpha"],
                           linewidth=0, label=label)
