from pathlib import Path

import pytest

from silent_rag.plots import style
from silent_rag.plots.style import SCHEMA_DEFAULTS, PlotStyle, deep_merge

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
    assert "matplotlib" not in Path(style.__file__).read_text()  # this step is pure dicts
