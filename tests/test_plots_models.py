import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from silent_rag.plots.figures_models import (  # noqa: E402
    dose_response_ndcg10_drop,
    ndcg10_drop_at_severity_vs_mean_cosine,
    replacement_model_difference,
)
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


# table 16 (replacement_model_difference_table): m_b is closer to the baseline than m_a
TABLE16 = pd.DataFrame({
    "label": ["m_a", "m_b"], "replacement_model": ["org/a", "org/b"], "replacement_revision": ["r1", "r2"], "k": 10,
    "mean_cosine": [0.80, 0.95], "mean_topk_shared": [4.0, 8.0], "topk_shared_ci_low": [3.5, 7.6],
    "topk_shared_ci_high": [4.4, 8.3], "baseline_ndcg10": BASE, "switched_ndcg10": [0.25, 0.31],
    "switched_ndcg10_ci_low": [0.22, 0.29], "switched_ndcg10_ci_high": [0.28, 0.33],
})
ERROR_BARS = "error bars: 95% CI over questions"


def models_fig(table=TABLE16, **kwargs):
    return replacement_model_difference(table, **kwargs)


def points(ax):
    """(x, y, colour, error bar low, high) per drawn model, in drawing order (the legend-only entry is NaN)."""
    out = []
    for container in ax.containers:
        x, y = container[0].get_data()
        if np.isnan(x[0]):
            continue
        (segment,) = container[2][0].get_segments()
        out.append((x[0], y[0], container[0].get_color(), segment[0][1], segment[1][1]))
    return out


def test_model_difference_returns_a_figure_without_pyplot_or_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plt.close("all")
    fig = models_fig()
    assert isinstance(fig, Figure) and len(fig.axes) == 1
    assert plt.get_fignums() == [] and list(tmp_path.iterdir()) == []


def test_shared_points_error_bars_references_and_mean_cosine_order():
    ax = models_fig().axes[0]
    okabe = PALETTES["okabe_ito"]
    assert points(ax) == [(0.95, 8.0, okabe[0], 7.6, 8.3), (0.80, 4.0, okabe[1], 3.5, 4.4)]  # m_b first: higher cosine
    assert ax.get_legend_handles_labels()[1] == ["all 10 shared", "identical vectors (mean cosine 1)", ERROR_BARS]
    assert ax.get_xlim()[0] == 0 and ax.get_xlim()[1] >= 1
    assert ax.get_xlabel() and ax.get_ylabel().startswith("Top-10 documents shared") and ax.get_title()


def test_switched_ndcg10_points_and_baseline_line():
    ax = models_fig(y="switched_ndcg10").axes[0]
    assert [p[:2] for p in points(ax)] == [(0.95, 0.31), (0.80, 0.25)]
    assert [p[3:] for p in points(ax)] == [pytest.approx((0.29, 0.33)), pytest.approx((0.22, 0.28))]
    assert ax.get_legend_handles_labels()[1] == ["baseline nDCG@10 = 0.300", ERROR_BARS]


def test_points_are_named_with_display_names_and_offsets():
    s = PlotStyle({"display_names": {"label": {"m_a": "Model A"}}, "label_offsets": {"m_a": [-14, -18]}})
    texts = {t.get_text(): tuple(t.xyann) for t in models_fig(style=s).axes[0].texts}
    assert texts == {"Model A": (-14, -18), "m_b": (6, 6)}


def test_model_difference_yaml_option_include_order_and_ax():
    s = PlotStyle({"figures": {"replacement_model_difference": {"y": "switched_ndcg10"}}})
    assert points(models_fig(style=s).axes[0])[0][1] == 0.31
    assert [p[0] for p in points(models_fig(include={"labels": ["m_a"]}).axes[0])] == [0.80]
    assert [p[0] for p in points(models_fig(style=PlotStyle({"order": {"labels": ["m_a"]}})).axes[0])] == [0.80, 0.95]
    fig, axes = new_figure(PlotStyle.default().resolve("grid"), ncols=2)
    assert models_fig(ax=axes[0], legend={"show": False}) is fig and points(axes[0]) and not axes[1].containers


@pytest.mark.parametrize("y, top_value", [("shared", 10), ("switched_ndcg10", 0.33)])  # k line / highest CI end
def test_model_difference_y_axis_starts_at_0(y, top_value):
    low, high = models_fig(y=y).axes[0].get_ylim()
    assert low == 0 and high > top_value  # the top is still autoscaled above the data and reference line
    assert models_fig(y=y, ylim=[0.2, 12]).axes[0].get_ylim() == (0.2, 12)  # axes.ylim overrides


def test_model_difference_errors():
    with pytest.raises(ValueError, match="unknown y 'cosine'"):
        models_fig(y="cosine")
    with pytest.raises(ValueError, match=r"^replacement_model_difference: no rows to plot after include \{'labels'"):
        models_fig(include={"labels": ["m_z"]})
    with pytest.raises(ValueError, match="k must be the same in every row"):
        models_fig(TABLE16.assign(k=[10, 5]))
    with pytest.raises(ValueError, match="baseline_ndcg10 must be the same in every row"):
        models_fig(TABLE16.assign(baseline_ndcg10=[0.3, 0.31]))


def at_severity(drops=DROPS, models=TABLE16, **kwargs):
    return ndcg10_drop_at_severity_vs_mean_cosine(drops, models, **kwargs)


def test_drop_at_severity_returns_a_figure_without_pyplot_or_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plt.close("all")
    fig = at_severity()
    assert isinstance(fig, Figure) and len(fig.axes) == 1
    assert plt.get_fignums() == [] and list(tmp_path.iterdir()) == []


def test_drop_at_severity_joins_table_9_and_table_16_on_label():
    ax = at_severity().axes[0]  # severity defaults to 1.0
    okabe = PALETTES["okabe_ito"]
    assert points(ax) == [(0.95, 0.12, okabe[0], 0.10, 0.14), (0.80, 0.05, okabe[1], 0.04, 0.06)]  # m_b first
    assert ax.get_legend_handles_labels()[1] == ["no drop (clean baseline)", ERROR_BARS]
    assert ax.get_xlim()[0] == 0 and ax.get_ylabel() == "Mean nDCG@10 drop at severity 1" and ax.get_title()
    assert {t.get_text() for t in ax.texts} == {"m_a", "m_b"}


def test_drop_at_severity_option_from_call_yaml_and_isclose():
    assert [p[1] for p in points(at_severity(severity=0.5).axes[0])] == [0.06, 0.02]
    s = PlotStyle({"figures": {"ndcg10_drop_at_severity_vs_mean_cosine": {"severity": 0.5}}})
    assert [p[1] for p in points(at_severity(style=s).axes[0])] == [0.06, 0.02]
    noisy = DROPS.assign(severity=DROPS["severity"] * 3 / 3 + 1e-12)  # stored value not exactly 1.0
    assert [p[1] for p in points(at_severity(noisy).axes[0])] == [0.12, 0.05]


def test_drop_at_severity_include_order_and_ax():
    assert [p[0] for p in points(at_severity(include={"labels": ["m_a"]}).axes[0])] == [0.80]
    assert [p[0] for p in points(at_severity(style=PlotStyle({"order": {"labels": ["m_a"]}})).axes[0])] == [0.80, 0.95]
    fig, axes = new_figure(PlotStyle.default().resolve("grid"), ncols=2)
    assert at_severity(ax=axes[1], legend={"show": False}) is fig and points(axes[1]) and not axes[0].containers


def test_drop_at_severity_lists_labels_missing_a_severity_or_a_mean_cosine():
    no_b_at_1 = DROPS[~(DROPS["failure"].eq("m_b") & DROPS["severity"].eq(1.0))]
    with pytest.raises(ValueError, match=r"without a row at severity 1: \['m_b'\]; .*mean cosine in table 16: \[\]"):
        at_severity(no_b_at_1)
    with pytest.raises(ValueError, match=r"severity 0\.7: \['m_b', 'm_a'\]; .*table 16: \['m_a'\]$"):
        at_severity(models=TABLE16[TABLE16["label"].eq("m_b")], severity=0.7)
    with pytest.raises(ValueError, match=r"^ndcg10_drop_at_severity_vs_mean_cosine: no rows to plot after include"):
        at_severity(include={"labels": ["m_z"]})
