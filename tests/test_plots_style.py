import subprocess
import sys

import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib.axes import Axes  # noqa: E402

from silent_rag.plots import style  # noqa: E402
from silent_rag.plots.style import (  # noqa: E402
    SCHEMA_DEFAULTS,
    PlotStyle,
    color_for,
    deep_merge,
    display_name,
    marker_for,
    new_figure,
    point_labels,
    save_figure,
    shared_legend,
    style_context,
)

FIG = "toy_figure"


@pytest.fixture(autouse=True)
def toy_defaults(monkeypatch):
    """Built-in defaults (layer 2) for a toy figure with one option, `scale`."""
    monkeypatch.setitem(style.FIGURE_DEFAULTS, FIG, {"fonts": {"title": 22}, "scale": "drop"})


def write_yaml(tmp_path, text):
    path = tmp_path / "figures.yaml"
    path.write_text(text)
    return path


def test_default_is_the_schema_plus_the_figure_defaults():
    got = PlotStyle.default().resolve(FIG)
    assert got == deep_merge(SCHEMA_DEFAULTS, {"fonts": {"title": 22}, "scale": "drop"})
    assert PlotStyle.default().resolve("no_builtin_defaults") == SCHEMA_DEFAULTS


def test_each_layer_wins_over_the_one_before(tmp_path):
    def title(yaml_text="", **kwargs):
        return PlotStyle.from_yaml(write_yaml(tmp_path, yaml_text)).resolve(FIG, **kwargs)["fonts"]["title"]

    assert PlotStyle.default().resolve("other")["fonts"]["title"] == 24  # 1 built-in global
    assert title() == 22  # 2 built-in per figure
    assert title("fonts: {title: 21}") == 21  # 3 YAML global
    assert title(f"fonts: {{title: 21}}\nfigures: {{{FIG}: {{fonts: {{title: 20}}}}}}") == 20  # 4 YAML figures.f
    assert title(f"fonts: {{title: 21}}\nfigures: {{{FIG}: {{fonts: {{title: 20}}}}}}", fonts={"title": 19}) == 19


def test_nested_mappings_merge_one_key_at_a_time():
    got = PlotStyle({"fonts": {"tick": 12}, "legend": {"ncol": 2}}).resolve(FIG, legend={"frame": True})
    assert got["fonts"] == {**SCHEMA_DEFAULTS["fonts"], "title": 22, "tick": 12}
    assert got["legend"] == {**SCHEMA_DEFAULTS["legend"], "ncol": 2, "frame": True}


def test_lists_replace():
    got = PlotStyle({"figure": {"formats": ["png"]}, "include": {"labels": ["a", "b"]}}).resolve(FIG, include={"labels": ["c"]})
    assert got["figure"]["formats"] == ["png"] and got["include"]["labels"] == ["c"]


def test_none_keyword_argument_is_not_given():
    s = PlotStyle({"axes": {"title": "from yaml"}, "fonts": {"tick": 12}})
    got = s.resolve(FIG, title=None, fonts=None, scale=None)
    assert (got["axes"]["title"], got["fonts"]["tick"], got["scale"]) == ("from yaml", 12, "drop")


def test_yaml_null_overrides(tmp_path):
    text = f"include: {{labels: [a]}}\nfigures: {{{FIG}: {{include: {{labels: null}}, axes: {{title: null}}}}}}"
    got = PlotStyle.from_yaml(write_yaml(tmp_path, text)).resolve(FIG, title=None)
    assert got["include"]["labels"] is None and got["axes"]["title"] is None


def test_keyword_shorthands_go_into_axes_and_figure_options_are_allowed():
    got = PlotStyle.default().resolve(FIG, title="T", xlabel="x", ylabel="y", xlim=[0, 1], ylim=[0, 2], scale="ndcg")
    assert {k: got["axes"][k] for k in ("title", "xlabel", "ylabel", "xlim", "ylim")} == {
        "title": "T", "xlabel": "x", "ylabel": "y", "xlim": [0, 1], "ylim": [0, 2]}
    assert got["scale"] == "ndcg"


@pytest.mark.parametrize("text, match", [
    ("fonts: {titel: 20}", r"'fonts\.titel' in the figure-style YAML \(global section\)"),
    ("scale: ndcg", r"'scale' in the figure-style YAML \(global section\)"),  # figure options are per figure
    ("fonts: 20", r"'fonts' .* must be a mapping"),
    (f"figures: {{{FIG}: {{legend: {{colour: red}}}}}}", rf"'legend\.colour' in the figure-style YAML section figures\.{FIG}"),
])
def test_unknown_yaml_keys_raise_naming_key_and_place(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        PlotStyle.from_yaml(write_yaml(tmp_path, text)).resolve(FIG)


@pytest.mark.parametrize("kwargs, match", [
    ({"colour": "red"}, rf"unknown keyword argument 'colour' for figure '{FIG}'"),
    ({"colour": None}, "unknown keyword argument 'colour'"),  # unknown even when None
    ({"fonts": {"size": 3}}, rf"'fonts\.size' in the keyword arguments of figure '{FIG}'"),
    ({"axes": {"gird": False}}, rf"'axes\.gird' in the keyword arguments of figure '{FIG}'"),
    ({"display_names": {"labels": {}}}, r"'display_names\.labels' in the keyword arguments"),
])
def test_unknown_keyword_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PlotStyle.default().resolve(FIG, **kwargs)


def test_every_schema_section_is_a_keyword_argument():
    got = PlotStyle({"display_names": {"label": {"a": "A"}}}).resolve(
        FIG, axes={"grid": False, "title": "from axes"}, order={"labels": ["b", "a"]},
        display_names={"label": {"b": "B"}}, label_offsets={"a": [1, 2]}, colors={"by_label": {"a": "#000000"}})
    assert (got["axes"]["grid"], got["axes"]["log_x"], got["axes"]["title"]) == (False, False, "from axes")
    assert got["order"]["labels"] == ["b", "a"] and got["label_offsets"] == {"a": [1, 2]}
    assert got["display_names"]["label"] == {"a": "A", "b": "B"} and got["colors"]["by_label"] == {"a": "#000000"}


def test_shorthand_wins_over_an_axes_dict():
    got = PlotStyle.default().resolve(FIG, title="shorthand", axes={"title": "dict", "xlabel": "x"})
    assert (got["axes"]["title"], got["axes"]["xlabel"]) == ("shorthand", "x")


def test_check_figure_names_lists_every_unknown_name():
    s = PlotStyle({"figures": {"auroc_vs_severity": {}, "auroc_vs_severiy": {}, "spearman": {}}})
    s.check_figure_names(["auroc_vs_severity", "auroc_vs_severiy", "spearman", "unused"])  # all known: no error
    with pytest.raises(ValueError, match=r"\['auroc_vs_severiy', 'spearman'\]"):
        s.check_figure_names(["auroc_vs_severity"])


def test_repeated_yaml_keys_are_refused(tmp_path):
    path = write_yaml(tmp_path, "fonts: {title: 1}\nfonts: {tick: 2}")
    with pytest.raises(ValueError, match=rf"{path}: line 2: repeated keys \['fonts'\]"):
        PlotStyle.from_yaml(path)


def test_worked_example_from_the_spec(monkeypatch):
    monkeypatch.delitem(style.FIGURE_DEFAULTS, FIG)
    s = PlotStyle({"fonts": {"title": 22, "tick": 14}, "figures": {"auroc_vs_severity": {"fonts": {"title": 18}}}})
    fonts = s.resolve("auroc_vs_severity", fonts={"tick": 12})["fonts"]
    assert (fonts["title"], fonts["tick"], fonts["axis_label"]) == (18, 12, 20)


def test_resolve_never_changes_the_defaults_and_imports_no_matplotlib():
    before = deep_merge(SCHEMA_DEFAULTS, {})
    PlotStyle.default().resolve(FIG)["fonts"]["title"] = 99
    assert SCHEMA_DEFAULTS == before and style.FIGURE_DEFAULTS[FIG]["fonts"]["title"] == 22
    code = "import sys, silent_rag.plots.style as s; s.PlotStyle.default().resolve('x'); print('matplotlib' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"


@pytest.fixture
def settings():
    return PlotStyle({"fonts": {"title": 30}, "label_offsets": {"a": [-14, -18]},
                      "display_names": {"label": {"a": "Model A"}}}).resolve(FIG)


def test_style_context_applies_settings_and_restores_every_rcparam(settings):
    before = dict(matplotlib.rcParams)
    with style_context(settings):
        assert matplotlib.rcParams["axes.titlesize"] == 30 and matplotlib.rcParams["lines.linewidth"] == 2.5
    assert dict(matplotlib.rcParams) == before


def test_new_figure_size_dpi_layout_and_grid(settings):
    fig, ax = new_figure(settings)
    assert tuple(fig.get_size_inches()) == (12.0, 6.75) and fig.dpi == 200
    assert fig.get_layout_engine().__class__.__name__ == "ConstrainedLayoutEngine" and isinstance(ax, Axes)
    assert new_figure(settings, nrows=2, ncols=3)[1].shape == (2, 3)


def test_shared_legend_names_each_label_once(settings):
    fig, axes = new_figure(settings, ncols=2)
    for ax in axes:
        ax.plot([0, 1], [0, 1], label="a")
    axes[1].plot([0, 1], [1, 0], label="b")
    assert [t.get_text() for t in shared_legend(fig, axes, settings).get_texts()] == ["a", "b"]
    assert shared_legend(fig, axes, deep_merge(settings, {"legend": {"show": False}})) is None


def drawn(settings):
    fig, ax = new_figure(settings)
    ax.plot([0, 1, 2], [0, 1, 0], label="a")
    ax.fill_between([0, 1, 2], [0, 0, 0], [1, 1, 1], alpha=0.2)  # clip paths and ids in the SVG
    ax.set_title("t")
    return fig


@pytest.mark.parametrize("formats", [["png", "svg"], ["png"], ["svg"]])
def test_save_writes_each_requested_format(tmp_path, settings, formats):
    paths = save_figure(drawn(settings), tmp_path / "sub" / "fig", deep_merge(settings, {"figure": {"formats": formats}}))
    assert paths == [tmp_path / "sub" / f"fig.{f}" for f in formats] and all(p.stat().st_size > 0 for p in paths)
    assert sorted(p.name for p in (tmp_path / "sub").iterdir()) == sorted(f"fig.{f}" for f in formats)


def test_saves_are_byte_identical(tmp_path, settings):
    settings = deep_merge(settings, {"figure": {"formats": ["png", "svg", "pdf"]}})
    first = save_figure(drawn(settings), tmp_path / "one", settings)
    second = save_figure(drawn(settings), tmp_path / "two", settings)  # a new figure object
    for a, b in zip(first, second):
        assert a.read_bytes() == b.read_bytes(), a.suffix
    png, svg, pdf = (p.read_bytes() for p in first)
    assert b"Matplotlib" not in png and b"<dc:date>" not in svg
    assert b"/CreationDate" not in pdf and b"Matplotlib" not in pdf


def test_colours_follow_order_labels_whatever_subset_is_drawn(settings):
    s = deep_merge(settings, {"order": {"labels": ["a", "b", "c"]}})
    okabe = style.PALETTES["okabe_ito"]
    assert okabe[:4] == ["#0072B2", "#009E73", "#E69F00", "#D55E00"]  # notebook 08's four models: blue, green, orange, vermillion
    assert color_for("c", s, ["c"]) == color_for("c", s, ["b", "c"]) == okabe[2]
    assert color_for("c", settings, ["c", "a"]) == okabe[0]  # no order.labels: the given order
    assert color_for("c", deep_merge(s, {"colors": {"by_label": {"c": "#123456"}}}), ["c"]) == "#123456"


def test_palette_by_name_or_list(settings):
    listed = deep_merge(settings, {"colors": {"palette": ["#111111", "#222222"]}})
    assert [color_for(x, listed, ["a", "b"]) for x in "ab"] == ["#111111", "#222222"]
    with pytest.raises(ValueError, match="3 labels but 2 palette colours"):
        color_for("c", listed, ["a", "b", "c"])
    with pytest.raises(ValueError, match="unknown palette 'viridis'"):
        color_for("a", deep_merge(settings, {"colors": {"palette": "viridis"}}), ["a"])


def test_markers_follow_the_colour_order_whatever_subset_is_drawn(settings):
    s = deep_merge(settings, {"order": {"labels": ["a", "b", "c"]}})
    shapes = SCHEMA_DEFAULTS["markers"]["sequence"]
    assert shapes[:4] == ["o", "s", "^", "X"] and "D" not in shapes and len(set(shapes)) == len(shapes) == 8
    assert marker_for("c", s, ["c"]) == marker_for("c", s, ["b", "c"]) == "^"  # position 2, as its colour
    palette = style.PALETTES["okabe_ito"]
    for given in (["c"], ["c", "a"], ["b", "a", "c"]):  # shape and colour always come from the same position
        for subset in (s, settings):
            for label in given:
                assert shapes.index(marker_for(label, subset, given)) == palette.index(color_for(label, subset, given))
    assert marker_for("c", settings, ["c", "a"]) == "o"  # no order.labels: the given order
    assert marker_for("c", deep_merge(s, {"markers": {"by_label": {"c": "*"}}}), ["c"]) == "*"


def test_marker_sequence_from_yaml_too_few_shapes_and_unknown_keys(tmp_path):
    s = PlotStyle.from_yaml(write_yaml(tmp_path, "markers:\n  sequence: [o, s]\n  by_label: {b: d}\n")).resolve(FIG)
    assert s["markers"] == {"sequence": ["o", "s"], "by_label": {"b": "d"}}  # the list replaces; by_label merges
    assert [marker_for(x, s, ["a", "b", "c"]) for x in "ab"] == ["o", "d"]
    with pytest.raises(ValueError, match="3 labels but 2 marker shapes"):
        marker_for("c", s, ["a", "b", "c"])
    with pytest.raises(ValueError, match="is not in order.labels or the given labels"):
        marker_for("z", s, ["a"])
    with pytest.raises(ValueError, match="unknown style key 'markers.shape'"):
        PlotStyle({"markers": {"shape": "o"}})
    assert PlotStyle.default().resolve(FIG, markers={"by_label": {"a": "s"}})["markers"]["by_label"] == {"a": "s"}


def test_display_name_falls_back_to_the_key(settings):
    assert display_name("label", "a", settings) == "Model A" and display_name("label", "b", settings) == "b"
    assert display_name("statistic", "ks_top1", settings) == "ks_top1"
    with pytest.raises(ValueError, match="unknown display-name kind 'labels'"):
        display_name("labels", "a", settings)


def test_point_labels_use_configured_offsets(settings):
    fig, ax = new_figure(settings)
    a, b = point_labels(ax, [0.1, 0.2], [1, 2], ["a", "b"], settings)
    assert (a.get_text(), tuple(a.xyann), a.get_ha(), a.get_va()) == ("Model A", (-14, -18), "right", "top")
    assert (b.get_text(), tuple(b.xyann), b.get_ha(), b.get_va()) == ("b", style.DEFAULT_LABEL_OFFSET, "left", "bottom")
    assert a.get_fontsize() == settings["fonts"]["annotation"]
