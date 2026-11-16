import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from silent_rag.plots.figures_statistics import (  # noqa: E402
    RANGE_BAND,
    seconds_per_statistic,
    share_of_significant_windows_vs_severity,
    statistic_vs_severity_all_test_questions,
)
from silent_rag.plots.style import PALETTES, PlotStyle, color_for, new_figure  # noqa: E402

# table 10 (statistic_all_test_questions_vs_validation_reference_by_severity): two statistics, a clean row each
STATS = pd.DataFrame(
    [("clean", None, 0.0, "s1", 0.07851, 0.07851, 0.07851, 0), ("clean", None, 0.0, "s2", 0.5, 0.5, 0.5, 0),
     ("m_a", "F1", 0.5, "s1", 0.10, 0.08, 0.12, 10), ("m_a", "F1", 1.0, "s1", 0.20, 0.18, 0.23, 10),
     ("m_b", "F1", 0.5, "s1", 0.30, 0.25, 0.33, 10), ("m_b", "F1", 1.0, "s1", 0.40, 0.35, 0.44, 10),
     ("m_a", "F1", 0.5, "s2", 0.6, 0.6, 0.6, 10), ("m_a", "F1", 1.0, "s2", 0.7, 0.7, 0.7, 10),
     ("m_b", "F1", 0.5, "s2", 0.8, 0.8, 0.8, 10), ("m_b", "F1", 1.0, "s2", 0.9, 0.9, 0.9, 10)],
    columns=["failure", "injector", "severity", "statistic", "mean", "min", "max", "n_seeds"])
MODELS = pd.DataFrame({"label": ["m_a", "m_b"], "mean_cosine": [0.80, 0.95]})  # m_b drawn first
CLEAN = "clean baseline = 0.079"


def stat_fig(table=STATS, statistic="s1", **kwargs):
    return statistic_vs_severity_all_test_questions(table, statistic=statistic, **kwargs)


def series(ax):
    """{label: (x, y)} of the data lines (reference lines have zorder 1)."""
    return {line.get_label(): (list(line.get_xdata()), list(line.get_ydata()))
            for line in ax.lines if line.get_zorder() != 1}


def reference_ys(ax):
    return [line.get_ydata()[0] for line in ax.lines if line.get_zorder() == 1]


def test_returns_a_figure_without_pyplot_or_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plt.close("all")
    fig = stat_fig()
    assert isinstance(fig, Figure) and len(fig.axes) == 1
    assert plt.get_fignums() == [] and list(tmp_path.iterdir()) == []


def test_one_line_per_label_with_the_range_over_seeds_as_band_and_the_clean_line():
    ax = stat_fig(annotate_ends=False).axes[0]
    assert series(ax) == {"m_a": ([0.5, 1.0], [0.10, 0.20]), "m_b": ([0.5, 1.0], [0.30, 0.40])}  # s1 only
    bands = [c.get_paths()[0].vertices[:, 1] for c in ax.collections if len(c.get_paths())]
    assert [(b.min(), b.max()) for b in bands] == [pytest.approx((0.08, 0.23)), pytest.approx((0.25, 0.44))]
    assert reference_ys(ax) == [pytest.approx(0.07851)]
    assert ax.get_legend_handles_labels()[1] == ["m_a", "m_b", CLEAN, RANGE_BAND]  # value with 3 decimals
    assert ax.get_xlim() == (0, pytest.approx(1.02)) and ax.get_xlabel() and ax.get_ylabel() and ax.get_title()


def test_clean_line_is_read_before_include():
    ax = stat_fig(include={"labels": ["m_b"], "severities": [1.0]}, annotate_ends=False).axes[0]
    assert series(ax) == {"m_b": ([1.0], [0.40])}
    assert reference_ys(ax) == [pytest.approx(0.07851)] and CLEAN in ax.get_legend_handles_labels()[1]


def test_annotate_ends_names_lines_at_the_highest_severity_instead_of_the_legend():
    ax = stat_fig().axes[0]  # annotate_ends defaults to True
    assert sorted(t.get_text() for t in ax.texts) == [" m_a", " m_b"]
    assert {t.get_position()[0] for t in ax.texts} == {1.0} and ax.get_xlim()[1] == pytest.approx(1.25)
    assert ax.get_legend_handles_labels()[1] == [CLEAN, RANGE_BAND]


def test_statistic_from_yaml_and_display_names():
    s = PlotStyle({"figures": {"statistic_vs_severity_all_test_questions": {"statistic": "s2"}},
                   "display_names": {"statistic": {"s2": "Stat two"}, "label": {"m_a": "Model A"}}})
    ax = statistic_vs_severity_all_test_questions(STATS, style=s, annotate_ends=False).axes[0]
    assert series(ax)["Model A"][1] == [0.6, 0.7] and reference_ys(ax) == [0.5]
    assert ax.get_ylabel() == "Stat two"
    assert ax.get_title() == "Stat two:\nall test questions vs validation reference, by severity"


def test_models_order_colours_and_ax():
    okabe = PALETTES["okabe_ito"]
    lines = [line for line in stat_fig(models=MODELS, annotate_ends=False).axes[0].lines if line.get_zorder() != 1]
    assert [(line.get_label(), line.get_color()) for line in lines] == [("m_b", okabe[0]), ("m_a", okabe[1])]
    fig, axes = new_figure(PlotStyle.default().resolve("grid"), ncols=2)
    assert stat_fig(ax=axes[1], legend={"show": False}) is fig and axes[1].lines and not axes[0].lines
    assert axes[1].get_legend() is None


def test_errors():
    with pytest.raises(ValueError, match=r"no statistic given; choose one of \['s1', 's2'\]"):
        stat_fig(statistic=None)
    with pytest.raises(ValueError, match=r"statistic 'ks' is not in the table; choose one of \['s1', 's2'\]"):
        stat_fig(statistic="ks")
    with pytest.raises(ValueError, match="expected one clean row for 's1', got 0"):
        stat_fig(STATS[STATS["failure"].ne("clean")])
    with pytest.raises(ValueError, match=r"^statistic_vs_severity_all_test_questions: no rows to plot after include "
                                         r"\{'labels': \['m_z'\]\}"):
        stat_fig(include={"labels": ["m_z"]})


# table 2 (share_of_windows_with_significant_ndcg10_drop_by_severity): two minimum drops, a clean row each
SHARES = pd.DataFrame(
    [("clean", 0.0, 0.0, 200, 2, 1.0), ("clean", 0.0, 0.02, 200, 0, 0.0),
     ("m_a", 0.5, 0.0, 200, 100, 50.0), ("m_a", 1.0, 0.0, 200, 180, 90.0),
     ("m_b", 0.5, 0.0, 200, 160, 80.0), ("m_b", 1.0, 0.0, 200, 200, 100.0),
     ("m_a", 0.5, 0.02, 200, 20, 10.0), ("m_a", 1.0, 0.02, 200, 60, 30.0),
     ("m_b", 0.5, 0.02, 200, 80, 40.0), ("m_b", 1.0, 0.02, 200, 140, 70.0)],
    columns=["failure", "severity", "min_mean_ndcg10_drop", "n_windows", "n_significant", "share_percent"])


def share_fig(table=SHARES, min_drop=0.02, **kwargs):
    return share_of_significant_windows_vs_severity(table, min_drop=min_drop, **kwargs)


def test_share_returns_a_figure_without_pyplot_or_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plt.close("all")
    fig = share_fig()
    assert isinstance(fig, Figure) and len(fig.axes) == 1
    assert plt.get_fignums() == [] and list(tmp_path.iterdir()) == []


def test_share_lines_at_the_min_drop_matched_with_isclose_and_the_clean_line():
    ax = share_fig(min_drop=0.02 + 1e-12, annotate_ends=False).axes[0]
    assert series(ax) == {"m_a": ([0.5, 1.0], [10.0, 30.0]), "m_b": ([0.5, 1.0], [40.0, 70.0])}
    assert reference_ys(ax) == [0.0] and ax.get_legend_handles_labels()[1] == ["m_a", "m_b", "clean baseline = 0.0%"]
    assert ax.get_ylim() == (-2, 102) and list(ax.get_yticks()) == [0, 20, 40, 60, 80, 100]
    assert "minimum mean drop 0.02" in ax.get_title() and "alpha" not in ax.get_title()
    assert share_fig(ylim=[0, 50]).axes[0].get_ylim() == (0, 50)  # axes.ylim overrides


def test_share_clean_line_is_read_before_include_and_annotate_ends():
    ax = share_fig(min_drop=0.0, include={"labels": ["m_b"], "severities": [1.0]}).axes[0]
    assert series(ax) == {"_nolegend_": ([1.0], [100.0])} and reference_ys(ax) == [1.0]  # named by its end label
    assert [t.get_text() for t in ax.texts] == [" m_b"] and ax.get_legend_handles_labels()[1] == ["clean baseline = 1.0%"]


def test_share_yaml_option_models_and_ax():
    s = PlotStyle({"figures": {"share_of_significant_windows_vs_severity": {"min_drop": 0.0}}})
    fig = share_of_significant_windows_vs_severity(SHARES, style=s, annotate_ends=False)
    assert series(fig.axes[0])["m_a"][1] == [50.0, 90.0]
    lines = [line for line in share_fig(models=MODELS, annotate_ends=False).axes[0].lines if line.get_zorder() != 1]
    assert [line.get_label() for line in lines] == ["m_b", "m_a"]
    fig, axes = new_figure(PlotStyle.default().resolve("grid"), ncols=2)
    assert share_fig(ax=axes[0], legend={"show": False}) is fig and axes[0].lines and not axes[1].lines


def test_share_errors():
    with pytest.raises(ValueError, match=r"no min_drop given; choose one of \[0\.0, 0\.02\]"):
        share_fig(min_drop=None)
    with pytest.raises(ValueError, match=r"min_drop 0\.06 is not in the table; choose one of \[0\.0, 0\.02\]"):
        share_fig(min_drop=0.06)
    with pytest.raises(ValueError, match="expected one clean row at min_drop 0.02, got 0"):
        share_fig(SHARES[SHARES["failure"].ne("clean")])
    with pytest.raises(ValueError, match=r"^share_of_significant_windows_vs_severity: no rows to plot after include"):
        share_fig(include={"labels": ["m_z"]})


# table 3 (median_seconds_per_statistic_evaluation), cheapest first
SECONDS = pd.DataFrame({"statistic": ["s_fast", "s_mid", "s_slow"], "median_seconds": [0.0002, 0.001, 2.0],
                        "n_evaluations": [8200, 8200, 410]})


def dots(ax):
    """[(tick label, median seconds)] from top to bottom."""
    ticks = {round(t): label.get_text() for t, label in zip(ax.get_yticks(), ax.get_yticklabels())}
    (line,) = ax.lines
    return sorted(((ticks[round(y)], x) for x, y in zip(line.get_xdata(), line.get_ydata())),
                  key=lambda d: list(ticks.values()).index(d[0]))


def test_seconds_returns_a_figure_without_pyplot_or_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plt.close("all")
    fig = seconds_per_statistic(SECONDS)
    assert isinstance(fig, Figure) and len(fig.axes) == 1
    assert plt.get_fignums() == [] and list(tmp_path.iterdir()) == []


def test_seconds_dots_in_table_order_top_down_log_x_and_n_labels():
    ax = seconds_per_statistic(SECONDS).axes[0]
    assert dots(ax) == [("s_fast", 0.0002), ("s_mid", 0.001), ("s_slow", 2.0)]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]  # inverted: first row at the top
    assert ax.get_xscale() == "log" and [t.get_text() for t in ax.texts] == ["n = 8,200", "n = 8,200", "n = 410"]
    assert seconds_per_statistic(SECONDS, axes={"log_x": False}).axes[0].get_xscale() == "linear"


def test_seconds_order_display_names_include_and_ax():
    s = PlotStyle({"order": {"statistics": ["s_slow"]}, "display_names": {"statistic": {"s_mid": "Middle"}}})
    assert [d[0] for d in dots(seconds_per_statistic(SECONDS, style=s).axes[0])] == ["s_slow", "s_fast", "Middle"]
    assert [d[0] for d in dots(seconds_per_statistic(SECONDS, include={"statistics": ["s_mid"]}).axes[0])] == ["s_mid"]
    fig, axes = new_figure(PlotStyle.default().resolve("grid"), ncols=2)
    assert seconds_per_statistic(SECONDS, ax=axes[1]) is fig and axes[1].lines and not axes[0].lines
    with pytest.raises(ValueError, match=r"^seconds_per_statistic: no rows to plot after include"):
        seconds_per_statistic(SECONDS, include={"statistics": ["s_z"]})


def test_seconds_unknown_palette_raises_the_color_for_message():
    s = PlotStyle({"colors": {"palette": "viridis"}})
    with pytest.raises(ValueError) as expected:
        color_for("s_fast", s.resolve("seconds_per_statistic"), ["s_fast"])
    with pytest.raises(ValueError) as got:
        seconds_per_statistic(SECONDS, style=s)
    assert str(got.value) == str(expected.value) and str(got.value).startswith("unknown palette 'viridis'")
    assert seconds_per_statistic(SECONDS, colors={"palette": ["#123456"]}).axes[0].lines[0].get_color() == "#123456"
