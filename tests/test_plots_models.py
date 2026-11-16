import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from silent_rag.plots.figures_models import dose_response_ndcg10_drop  # noqa: E402
from silent_rag.plots.style import PALETTES, PlotStyle, new_figure  # noqa: E402

BASE = 0.3
# table 9 (mean_ndcg10_drop_all_test_questions_by_severity), alphabetical as the analysis writes it
DROPS = pd.DataFrame({
    "failure": ["m_a", "m_a", "m_b", "m_b"], "injector": "F1_partial_upgrade", "severity": [0.5, 1.0, 0.5, 1.0],
    "mean_ndcg10_drop": [0.02, 0.05, 0.06, 0.12], "ci_low": [0.01, 0.04, 0.05, 0.10],
    "ci_high": [0.03, 0.06, 0.07, 0.14], "n_questions": 323, "n_seeds": 10, "baseline_ndcg10": BASE,
})
MODELS = pd.DataFrame({"label": ["m_a", "m_b"], "mean_cosine": [0.80, 0.95]})  # m_b more similar: drawn first
BAND = "shaded: 95% CI over questions"


def dose(table=DROPS, **kwargs):
    return dose_response_ndcg10_drop(table, **kwargs)


def series(fig, name=None):
    """The data lines of the figure's axes (reference lines have zorder 1)."""
    lines = [line for line in fig.axes[0].lines if line.get_zorder() != 1]
    return lines if name is None else next(line for line in lines if line.get_label() == name)


def test_returns_a_figure_without_pyplot_or_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plt.close("all")
    fig = dose()
    assert isinstance(fig, Figure) and len(fig.axes) == 1
    assert plt.get_fignums() == [] and list(tmp_path.iterdir()) == []


def test_drop_scale_lines_start_at_the_definitional_severity_0_point():
    fig = dose(annotate_ends=False)
    ax = fig.axes[0]
    assert ax.get_legend_handles_labels()[1] == ["m_a", "m_b", "no drop (clean baseline)", BAND]
    a = series(fig, "m_a")
    assert list(a.get_xdata()) == [0.0, 0.5, 1.0] and list(a.get_ydata()) == pytest.approx([0.0, 0.02, 0.05])
    assert ax.get_xlabel() and ax.get_ylabel() and ax.get_title()


def band_bounds(fig, i):
    vertices = fig.axes[0].collections[i].get_paths()[0].vertices
    return vertices[:, 0].min(), vertices[:, 1].min(), vertices[:, 1].max()


@pytest.mark.parametrize("scale, y0, x_end, band_at_end", [
    ("drop", 0.0, 1.0, (0.04, 0.06)),
    ("ndcg", BASE, 1.0, (BASE - 0.06, BASE - 0.04)),  # bounds swap: low = baseline - ci_high
    ("percent_of_baseline", 100.0, 100.0, (100 * (BASE - 0.06) / BASE, 100 * (BASE - 0.04) / BASE)),
])
def test_scales_convert_line_and_band_and_leave_severity_0_without_a_band(scale, y0, x_end, band_at_end):
    one = DROPS[DROPS["failure"].eq("m_a") & DROPS["severity"].eq(1.0)]
    fig = dose(one, scale=scale, annotate_ends=False)
    line = series(fig, "m_a")
    assert line.get_xdata()[0] == 0.0 and line.get_ydata()[0] == pytest.approx(y0) and line.get_xdata()[-1] == x_end
    x_min, low, high = band_bounds(fig, 0)
    assert x_min == x_end and (low, high) == pytest.approx(band_at_end)  # the band starts at the first severity


def test_reference_line_per_scale():
    labels = {s: dose(scale=s, annotate_ends=False).axes[0].get_legend_handles_labels()[1][2] for s in
              ("drop", "ndcg", "percent_of_baseline")}
    assert labels == {"drop": "no drop (clean baseline)", "ndcg": "clean baseline nDCG@10 = 0.300",
                      "percent_of_baseline": "proportional to the share of vectors kept"}
    curve = next(line for line in dose(scale="percent_of_baseline").axes[0].lines if line.get_zorder() == 1)
    assert list(curve.get_xdata()) == [0, 100] and list(curve.get_ydata()) == [100, 0]


def test_annotate_ends_names_lines_at_the_highest_severity_instead_of_the_legend():
    ax = dose().axes[0]  # annotate_ends defaults to True
    assert sorted(t.get_text() for t in ax.texts) == [" m_a", " m_b"]
    assert {t.get_position()[0] for t in ax.texts} == {1.0} and ax.get_xlim()[1] == pytest.approx(1.25)
    assert ax.get_legend_handles_labels()[1] == ["no drop (clean baseline)", BAND]


def test_yaml_sets_the_options_when_the_call_does_not():
    s = PlotStyle({"figures": {"dose_response_ndcg10_drop": {"scale": "ndcg", "annotate_ends": False}}})
    assert series(dose(style=s), "m_a").get_ydata()[0] == pytest.approx(BASE)
    assert series(dose(style=s, scale="drop"), "m_a").get_ydata()[0] == 0.0  # the keyword argument wins


def test_include_removes_a_series_and_display_names_apply():
    fig = dose(include={"labels": ["m_b"]}, annotate_ends=False,
               style=PlotStyle({"display_names": {"label": {"m_b": "Model B"}}}))
    assert [line.get_label() for line in series(fig)] == ["Model B"]


def test_models_table_sets_label_order_and_colours():
    okabe = PALETTES["okabe_ito"]
    plain = series(dose(annotate_ends=False))
    assert [(line.get_label(), line.get_color()) for line in plain] == [("m_a", okabe[0]), ("m_b", okabe[1])]
    ordered = series(dose(models=MODELS, annotate_ends=False))
    assert [(line.get_label(), line.get_color()) for line in ordered] == [("m_b", okabe[0]), ("m_a", okabe[1])]
    overridden = series(dose(models=MODELS, annotate_ends=False, style=PlotStyle({"order": {"labels": ["m_a"]}})))
    assert [line.get_label() for line in overridden] == ["m_a", "m_b"]  # order.labels wins over mean cosine


def test_ax_draws_into_the_given_axes():
    fig, axes = new_figure(PlotStyle.default().resolve("grid"), ncols=2)
    got = dose(ax=axes[1], legend={"show": False})
    assert got is fig and len(axes[1].lines) > 0 and not axes[0].lines and axes[1].get_legend() is None


@pytest.mark.parametrize("scale, bottom_is_0", [("drop", False), ("ndcg", True), ("percent_of_baseline", True)])
def test_y_axis_starts_at_0_on_the_ndcg_scales(scale, bottom_is_0):
    ax = dose(scale=scale).axes[0]
    low, high = ax.get_ylim()
    assert bool(low == 0) is bottom_is_0
    top_data = max(np.max(line.get_ydata()) for line in ax.lines)
    assert high > top_data  # the top is still autoscaled above the data
    assert dose(scale=scale, ylim=[0.1, 50]).axes[0].get_ylim() == (0.1, 50)  # axes.ylim overrides


def test_no_rows_after_include_raises_naming_the_settings():
    with pytest.raises(ValueError, match=r"no rows to plot after include \{'labels': \['m_z'\], 'severities': \[0\.5\]\}"):
        dose(include={"labels": ["m_z"], "severities": [0.5]})
    with pytest.raises(ValueError, match="the table has no rows"):
        dose(DROPS.iloc[:0])


def test_unknown_scale_and_varying_baseline_raise():
    with pytest.raises(ValueError, match="unknown scale 'log'"):
        dose(scale="log")
    varying = DROPS.assign(baseline_ndcg10=np.where(DROPS["failure"].eq("m_a"), 0.3, 0.31))
    with pytest.raises(ValueError, match="baseline_ndcg10 must be the same in every row"):
        dose(varying)
