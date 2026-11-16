"""Drawing helpers shared by the figure modules (spec section 5.13).

Each helper takes resolved settings (PlotStyle.resolve's dict); the drawing ones draw into the
axes they are given and never create, save or show a figure. Nothing here computes a statistic:
helpers filter, order and draw values that are already in the tables (decision 12).
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

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


def line_with_band(ax, x, y, settings: dict, *, color, label, low=None, high=None):
    """A line with markers and, when low and high are given, a shaded band between them
    (lines.band_alpha). Returns (line, band or None). The band carries no legend entry; the
    caller says what it is (CI over questions, CI over seeds, range over seeds)."""
    lines = settings["lines"]
    band = None
    if low is not None and high is not None:
        band = ax.fill_between(x, low, high, color=color, alpha=lines["band_alpha"], linewidth=0)
    (line,) = ax.plot(x, y, "o-", color=color, label=label, linewidth=lines["width"], markersize=lines["marker_size"])
    return line, band


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
    range above the one below it, so they do not overlap. Set the y limits before calling. NaN
    ends are skipped."""
    low, high = ax.get_ylim()
    gap, placed, texts = (high - low) * min_gap, [], []
    for y, text, color in sorted(e for e in ends if not np.isnan(e[0])):
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        texts.append(ax.text(x, y, f" {text}", color=color, fontsize=settings["fonts"]["annotation"],
                             va="center", clip_on=False))
    return texts


def nan_note(ax, table: pd.DataFrame, value_column: str, settings: dict, note_column: str = "note",
             interval_columns=("ci_low", "ci_high"), wrap_width: int = 80):
    """Write in the lower left of the axes what the table has that cannot be drawn, with the
    reasons from the note column, one clause per kind, each wrapped to wrap_width characters:
    - rows with a NaN value_column: "3 points not shown (value undefined): undefined: no positive windows (3)";
    - rows with a value but a NaN in any of interval_columns: "2 intervals not shown: no CI: one seed (2)".
    Interval columns the table does not have are ignored (as in filter_include), so a table without
    intervals gets no interval clause. None when there is nothing to report."""
    values_missing = table[value_column].isna()
    present = [c for c in (interval_columns or ()) if c in table.columns]
    intervals_missing = ~values_missing & table[present].isna().any(axis=1)
    clauses = [_counted(table[values_missing], note_column, "point", " (value undefined)"),
               _counted(table[intervals_missing], note_column, "interval", "")]
    clauses = [textwrap.fill(c, wrap_width) for c in clauses if c]
    if not clauses:
        return None
    return ax.text(0.01, 0.01, "\n".join(clauses), transform=ax.transAxes, ha="left", va="bottom",
                   fontsize=settings["fonts"]["annotation"], color=settings["colors"]["reference_grey"])


def _counted(rows: pd.DataFrame, note_column: str, noun: str, qualifier: str) -> str:
    """'<n> <noun>s not shown<qualifier>: <reason> (<count>); ...', or "" when rows is empty."""
    if rows.empty:
        return ""
    reasons = rows[note_column].fillna("no note").replace("", "no note").value_counts()
    because = "; ".join(f"{reason} ({count})" for reason, count in reasons.items())
    return f"{len(rows)} {noun}{'s' if len(rows) != 1 else ''} not shown{qualifier}: {because}"
