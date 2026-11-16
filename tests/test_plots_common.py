import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from silent_rag.plots._common import (  # noqa: E402
    band_legend_entry,
    end_labels,
    figure_axes,
    filter_include,
    finish_axes,
    label_order,
    line_with_band,
    nan_note,
    reference_curve,
    reference_line,
    require_rows,
)
from silent_rag.plots.style import PlotStyle, deep_merge, new_figure, style_context  # noqa: E402


def resolved(**sections):
    return PlotStyle(sections).resolve("any_figure")


def axes():
    return new_figure(resolved())[1]


AUROC = pd.DataFrame({
    "failure": ["m1", "m1", "m2", "m2", "m3"],
    "statistic": ["ks_top1", "re_embed", "ks_top1", "ks_top1", "ks_top1"],
    "severity": [0.1, 0.1, 0.3, 0.30000000000000004, np.nan],  # 0.1 * 3 as computed; NaN = pooled row
    "min_mean_ndcg10_drop": [0.0, 0.0, 0.02, 0.02, 0.0],
    "auroc": [0.6, np.nan, 0.8, np.nan, np.nan],
    "ci_low": [0.5, np.nan, np.nan, np.nan, np.nan],
    "ci_high": [0.7, np.nan, np.nan, np.nan, np.nan],
    "note": ["", "undefined: no positive windows", "no CI: one seed", "undefined: no positive windows",
             "undefined: no negative windows"],
})
MODELS = pd.DataFrame({"label": ["m1", "m2", "m3"], "mean_cosine": [0.7, 0.9, 0.8]})


def test_include_null_keeps_every_row():
    assert filter_include(AUROC, resolved()).equals(AUROC)


@pytest.mark.parametrize("include, rows", [
    ({"labels": ["m2"]}, [2, 3]),
    ({"statistics": ["ks_top1"]}, [0, 2, 3, 4]),
    ({"severities": [0.3]}, [2, 3, 4]),  # isclose match; the NaN-severity row is kept
    ({"min_drops": [0.0]}, [0, 1, 4]),
    ({"labels": ["m1", "m2"], "statistics": ["ks_top1"]}, [0, 2, 3]),
])
def test_include_filters_remove_rows(include, rows):
    assert filter_include(AUROC, resolved(include=include)).index.tolist() == rows


def test_include_uses_the_label_column_and_ignores_absent_columns():
    got = filter_include(MODELS, resolved(include={"labels": ["m3"], "statistics": ["ks_top1"]}))
    assert got["label"].tolist() == ["m3"]  # table 16 has "label", no "statistic"


def test_require_rows_passes_when_rows_remain():
    s = resolved(include={"labels": ["m2"]})
    assert require_rows(AUROC, filter_include(AUROC, s), s, "toy_figure") is None


def test_require_rows_names_the_set_include_keys_when_include_removes_every_row():
    s = resolved(include={"labels": ["m9"], "severities": [0.1]})  # statistics, min_drops stay null
    with pytest.raises(ValueError, match=r"^toy_figure: no rows to plot after include "
                                         r"\{'labels': \['m9'\], 'severities': \[0\.1\]\}$"):
        require_rows(AUROC, filter_include(AUROC, s), s, "toy_figure")


def test_require_rows_reports_an_empty_input_table_first():
    s = resolved(include={"labels": ["m9"]})
    empty = AUROC.iloc[:0]
    with pytest.raises(ValueError, match=r"^toy_figure: the table has no rows$"):
        require_rows(empty, filter_include(empty, s), s, "toy_figure")


def test_label_order_rules():
    labels = ["m1", "m3", "m2", "m1"]
    assert label_order(labels, resolved()) == ["m1", "m3", "m2"]  # first appearance
    assert label_order(labels, resolved(), MODELS) == ["m2", "m3", "m1"]  # mean cosine descending
    assert label_order(labels + ["m4"], resolved(), MODELS) == ["m2", "m3", "m1", "m4"]  # not in table 16: last
    s = resolved(order={"labels": ["m1", "m9"]})
    assert label_order(labels, s, MODELS) == ["m1", "m2", "m3"]  # order.labels first; absent m9 dropped


def test_line_with_band_draws_one_line_and_one_band():
    ax, s = axes(), resolved(lines={"band_alpha": 0.3})
    line, band = line_with_band(ax, [0.1, 0.5], [1, 2], s, color="#0072B2", label="m1", low=[0.5, 1.5], high=[1.5, 2.5])
    assert len(ax.lines) == 1 and list(ax.collections) == [band] and band.get_alpha() == 0.3
    assert (line.get_label(), line.get_color(), line.get_linewidth()) == ("m1", "#0072B2", 2.5)
    assert ax.get_legend_handles_labels()[1] == ["m1"]  # the band has no legend entry
    _, no_band = line_with_band(axes(), [0, 1], [1, 2], s, color="k", label="m2")
    assert no_band is None


def test_reference_line_is_grey_dashed_and_in_the_legend():
    ax, s = axes(), resolved()
    h = reference_line(ax, s, label="AUROC 0.5", y=0.5)
    v = reference_line(ax, s, label="alpha = 0.05", x=0.05)
    assert (h.get_color(), h.get_linestyle(), h.get_linewidth()) == ("#7f7f7f", "--", 1.5)
    assert tuple(h.get_ydata()) == (0.5, 0.5) and tuple(v.get_xdata()) == (0.05, 0.05)
    assert ax.get_legend_handles_labels()[1] == ["AUROC 0.5", "alpha = 0.05"]
    with pytest.raises(ValueError, match="exactly one"):
        reference_line(ax, s, label="x", x=1, y=1)


def test_end_labels_are_spread_and_skip_nan():
    ax, s = axes(), resolved()
    ax.set_ylim(0, 1)
    texts = end_labels(ax, 1.0, [(0.50, "b", "C1"), (0.48, "a", "C0"), (0.9, "c", "C2"), (np.nan, "d", "C3")], s)
    assert [t.get_text() for t in texts] == [" a", " b", " c"]
    ys = [t.get_position()[1] for t in texts]
    assert ys == pytest.approx([0.48, 0.54, 0.9])  # b pushed up to 0.48 + 6% of the range
    assert {t.get_position()[0] for t in texts} == {1.0} and texts[0].get_fontsize() == 16


def test_nan_note_states_points_and_intervals_with_counts_and_reasons():
    ax, s = axes(), resolved()
    note = nan_note(ax, AUROC, "auroc", s, wrap_width=200)
    assert note.get_text() == (
        "3 points not shown (value undefined): undefined: no positive windows (2); undefined: no negative windows (1)\n"
        "1 interval not shown: no CI: one seed (1)")  # the NaN-AUROC rows are not counted again as intervals
    assert list(ax.texts) == [note]
    only_intervals = nan_note(axes(), AUROC.iloc[[0, 2]], "auroc", s)
    assert only_intervals.get_text() == "1 interval not shown: no CI: one seed (1)"
    one = nan_note(axes(), AUROC.iloc[[0, 1]].assign(note=["", ""]), "auroc", s)
    assert one.get_text() == "1 point not shown (value undefined): no note (1)"
    assert nan_note(axes(), AUROC.iloc[[0]], "auroc", s) is None


def test_nan_note_without_interval_columns():
    rho = AUROC.drop(columns=["ci_low", "ci_high"]).rename(columns={"auroc": "rho"})
    for no_intervals in ({}, {"interval_columns": None}, {"interval_columns": ()}):  # {}: absent defaults ignored
        note = nan_note(axes(), rho, "rho", resolved(), wrap_width=200, **no_intervals)
        assert note.get_text().startswith("3 points not shown") and "interval" not in note.get_text()
    assert nan_note(axes(), rho.dropna(subset=["rho"]), "rho", resolved()) is None


def test_nan_note_wraps_long_text():
    reasons = [f"undefined: reason number {i} for this severity" for i in range(4)]
    table = pd.DataFrame({"auroc": [np.nan] * 4, "ci_low": np.nan, "ci_high": np.nan, "note": reasons})
    lines = nan_note(axes(), table, "auroc", resolved()).get_text().split("\n")
    assert len(lines) >= 3 and all(len(line) <= 80 for line in lines)
    assert " ".join(lines).startswith("4 points not shown (value undefined): undefined: reason number 0")
    narrow = nan_note(axes(), table, "auroc", resolved(), wrap_width=40).get_text().split("\n")
    assert len(narrow) > len(lines) and all(len(line) <= 40 for line in narrow)


def test_reference_curve_is_styled_like_reference_line_and_in_the_legend():
    ax, s = axes(), resolved(lines={"reference_line_style": ":", "reference_line_width": 2.0})
    curve = reference_curve(ax, [0, 100], [100, 0], s, label="proportional (1:1)")
    flat = reference_line(ax, s, label="clean baseline", y=0)
    assert (curve.get_color(), curve.get_linestyle(), curve.get_linewidth(), curve.get_zorder()) == (
        flat.get_color(), flat.get_linestyle(), flat.get_linewidth(), flat.get_zorder()) == ("#7f7f7f", ":", 2.0, 1)
    assert list(curve.get_xdata()) == [0, 100] and list(curve.get_ydata()) == [100, 0]
    assert ax.get_legend_handles_labels()[1] == ["proportional (1:1)", "clean baseline"]


def test_helpers_never_create_a_pyplot_figure():
    plt.close("all")
    ax, s = axes(), resolved()
    line_with_band(ax, [0, 1], [0, 1], s, color="k", label="a", low=[0, 0], high=[1, 1])
    reference_line(ax, s, label="r", y=0)
    end_labels(ax, 1, [(0.5, "a", "k")], s)
    nan_note(ax, AUROC, "auroc", s)
    assert plt.get_fignums() == []


def test_figure_axes_creates_a_figure_only_without_ax():
    s = resolved(figure={"width": 8.0, "height": 4.0})
    fig, ax = figure_axes(s)
    assert tuple(fig.get_size_inches()) == (8.0, 4.0) and ax.figure is fig
    given = axes()
    size = tuple(given.figure.get_size_inches())
    fig2, ax2 = figure_axes(s, given)
    assert ax2 is given and fig2 is given.figure and tuple(fig2.get_size_inches()) == size  # size settings unused


def test_finish_axes_uses_builtin_text_unless_the_axes_settings_say_otherwise():
    ax = axes()
    finish_axes(ax, resolved(), title="T", xlabel="Severity", ylabel="Drop")
    assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("T", "Severity", "Drop")
    ax = axes()
    finish_axes(ax, resolved(axes={"title": "Finding", "ylabel": "nDCG@10"}), title="T", xlabel="Severity", ylabel="Drop")
    assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("Finding", "Severity", "nDCG@10")


def test_finish_axes_sets_font_sizes_on_an_axes_made_outside_style_context():
    s = resolved(fonts={"title": 17, "axis_label": 13, "tick": 11})
    ax = axes()  # created with matplotlib's rcParams, before any style_context
    ax.plot([1, 10], [1, 100])
    with style_context(s):
        finish_axes(ax, s, title="T", xlabel="x", ylabel="y")
    assert (ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(), ax.yaxis.label.get_fontsize()) == (17, 13, 13)
    ticks = ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks()
    assert ticks and {t.label1.get_fontsize() for t in ticks} == {11}
    ax.set_xscale("log")  # ticks recreated later keep the size
    assert {t.label1.get_fontsize() for t in ax.xaxis.get_major_ticks() + ax.xaxis.get_minor_ticks()} == {11}


def test_finish_axes_limits_and_log_scales():
    ax = axes()
    ax.plot([1, 10], [1, 100])
    finish_axes(ax, resolved(axes={"xlim": [0.5, 20], "ylim": [1, 1000], "log_x": True, "log_y": True}),
                title="", xlabel="x", ylabel="y")
    assert ax.get_xlim() == pytest.approx((0.5, 20)) and ax.get_ylim() == pytest.approx((1, 1000))
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")


def test_finish_axes_legend_on_the_axes_outside_or_none():
    s = resolved(legend={"ncol": 2, "title": "Replacement model"})
    ax = axes()
    ax.plot([0, 1], [0, 1], label="a")
    legend = finish_axes(ax, s, title="", xlabel="x", ylabel="y")
    assert ax.get_legend() is legend and [t.get_text() for t in legend.get_texts()] == ["a"]
    assert legend.get_title().get_text() == "Replacement model"
    ax = axes()
    ax.plot([0, 1], [0, 1], label="a")
    outside = finish_axes(ax, deep_merge(s, {"legend": {"outside": True}}), title="", xlabel="x", ylabel="y")
    assert ax.get_legend() is None and ax.figure.legends == [outside]
    ax = axes()
    ax.plot([0, 1], [0, 1], label="a")
    assert finish_axes(ax, deep_merge(s, {"legend": {"show": False}}), title="", xlabel="x", ylabel="y") is None
    assert ax.get_legend() is None
    ax = axes()
    ax.plot([0, 1], [0, 1])  # nothing labelled
    assert finish_axes(ax, s, title="", xlabel="x", ylabel="y") is None and ax.get_legend() is None


def test_band_legend_entry_names_the_band_without_drawing_data():
    ax, s = axes(), resolved(lines={"band_alpha": 0.25})
    ax.plot([0, 1], [2, 3], label="m1")
    before = ax.dataLim.frozen()
    entry = band_legend_entry(ax, s, "shaded: 95% CI over questions")
    assert ax.get_legend_handles_labels()[1] == ["m1", "shaded: 95% CI over questions"]
    assert entry.get_alpha() == 0.25 and matplotlib.colors.to_hex(entry.get_facecolor()[0]) == "#7f7f7f"
    assert ax.dataLim.frozen().bounds == before.bounds
