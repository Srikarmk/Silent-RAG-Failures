import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from silent_rag.plots.figures_statistics import RANGE_BAND, statistic_vs_severity_all_test_questions  # noqa: E402
from silent_rag.plots.style import PALETTES, PlotStyle, new_figure  # noqa: E402

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
