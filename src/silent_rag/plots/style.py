"""Figure style settings: schema defaults and the five-layer merge (spec section 5.13).

For figure f, later layers win: built-in global defaults (SCHEMA_DEFAULTS) < built-in defaults
for f (FIGURE_DEFAULTS[f], set by f's module) < the YAML global section < the YAML figures.f
section < keyword arguments. Mappings merge key by key; lists and scalars replace. A keyword
argument of None means "not given"; an explicit null in YAML overrides. Unknown keys raise.

The drawing helpers below take resolved settings (PlotStyle.resolve's dict) and import matplotlib
only when called, so loading and resolving styles never imports it.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

SCHEMA_DEFAULTS = {
    "figure": {"width": 12.0, "height": 6.75, "dpi": 200, "formats": ["png", "svg"], "layout": "constrained"},
    "fonts": {"family": None, "title": 24, "axis_label": 20, "tick": 16, "legend": 16, "annotation": 16},
    "lines": {"width": 2.5, "marker_size": 8, "band_alpha": 0.18, "reference_line_style": "--",
              "reference_line_width": 1.5},
    "colors": {"palette": "okabe_ito", "reference_grey": "#7f7f7f", "by_label": {}},
    # one marker shape per label, beside its colour; no "D": the diamond is not used for a label
    "markers": {"sequence": ["o", "s", "^", "X", "v", "P", "h", "<"], "by_label": {}},
    "order": {"labels": [], "statistics": [], "severities": []},
    "display_names": {"label": {}, "statistic": {}, "questions": {}, "negatives": {}},
    "include": {"labels": None, "statistics": None, "severities": None, "min_drops": None},
    "label_offsets": {},
    "legend": {"show": True, "loc": "best", "title": None, "outside": False, "ncol": 1, "frame": False},
    "axes": {"title": None, "xlabel": None, "ylabel": None, "xlim": None, "ylim": None,
             "log_x": False, "log_y": False, "grid": True},
}

# Built-in per-figure defaults (layer 2), filled by each figure's module. Keys outside the schema
# are that figure's options (e.g. "scale"); they are the only extra keys its YAML section and
# keyword arguments may use.
FIGURE_DEFAULTS: dict[str, dict] = {}

AXES_SHORTHANDS = ("title", "xlabel", "ylabel", "xlim", "ylim")


def deep_merge(base: dict, override: dict) -> dict:
    """A new dict: base with override applied. Where both values are mappings they merge key by
    key, recursively; anything else in override (lists, scalars, None) replaces base's value."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def check_keys(settings: dict, where: str, schema: dict = SCHEMA_DEFAULTS, prefix: str = "") -> None:
    """Raise ValueError naming the first key of `settings` not in `schema` (recursively) and
    `where` it was found. A schema mapping that is empty ({}) accepts any key (e.g. by_label)."""
    for key, value in settings.items():
        name = f"{prefix}{key}"
        if key not in schema:
            raise ValueError(f"unknown style key {name!r} in {where}")
        if isinstance(schema[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"style key {name!r} in {where} must be a mapping, got {value!r}")
            if schema[key]:
                check_keys(value, where, schema[key], f"{name}.")


class PlotStyle:
    """The figure-style YAML (global section plus `figures`), resolved per figure by resolve()."""

    def __init__(self, settings: dict | None = None):
        settings = copy.deepcopy(settings or {})
        figures = settings.pop("figures", None) or {}
        if not isinstance(figures, dict):
            raise ValueError(f"style key 'figures' must be a mapping, got {figures!r}")
        check_keys(settings, "the figure-style YAML (global section)")
        self.global_settings, self.figures = settings, figures

    @classmethod
    def default(cls) -> PlotStyle:
        return cls()

    @classmethod
    def from_yaml(cls, path) -> PlotStyle:
        from silent_rag.run import UniqueKeyLoader  # refuses repeated keys, like the run configs

        with open(Path(path)) as f:
            try:
                settings = yaml.load(f, Loader=UniqueKeyLoader)
            except SystemExit as e:  # UniqueKeyLoader's error for a repeated key
                raise ValueError(f"{path}: {e}") from None
        if settings is not None and not isinstance(settings, dict):
            raise ValueError(f"{path}: the figure-style YAML must be a mapping")
        return cls(settings)

    def check_figure_names(self, known) -> None:
        """Raise ValueError listing every name under `figures` that is not in `known`."""
        unknown = sorted(set(self.figures) - set(known))
        if unknown:
            raise ValueError(f"unknown figure names under 'figures' in the figure-style YAML: {unknown}")

    def resolve(self, figure_name: str, **kwargs) -> dict:
        """The settings for one figure: the five layers merged, as one dict (no `figures` key)."""
        builtin = FIGURE_DEFAULTS.get(figure_name, {})
        schema = {**SCHEMA_DEFAULTS, **{k: v for k, v in builtin.items() if k not in SCHEMA_DEFAULTS}}
        check_keys(builtin, f"the built-in defaults of figure {figure_name!r}", schema)
        yaml_figure = self.figures.get(figure_name) or {}
        check_keys(yaml_figure, f"the figure-style YAML section figures.{figure_name}", schema)
        unknown = [k for k in kwargs if k not in schema and k not in AXES_SHORTHANDS]
        if unknown:
            raise ValueError(f"unknown keyword argument {unknown[0]!r} for figure {figure_name!r}")
        given = {k: v for k, v in kwargs.items() if v is not None and k not in AXES_SHORTHANDS}
        shorthands = {k: v for k, v in kwargs.items() if v is not None and k in AXES_SHORTHANDS}
        if shorthands:  # applied after an axes= dict, so title= wins over axes={"title": ...}
            given = deep_merge(given, {"axes": shorthands})
        check_keys(given, f"the keyword arguments of figure {figure_name!r}", schema)
        resolved = SCHEMA_DEFAULTS
        for layer in (builtin, self.global_settings, yaml_figure, given):
            resolved = deep_merge(resolved, layer)
        return resolved


# Okabe & Ito (2008) colour-blind-safe palette, ordered as notebook 08's figures (blue, green,
# orange, vermillion for the four models in mean-cosine order); yellow and black last.
PALETTES = {"okabe_ito": ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]}
DEFAULT_LABEL_OFFSET = (6, 6)  # points, for labels without an entry in label_offsets
SVG_HASHSALT = "silent-rag"  # fixes the ids matplotlib writes into SVG files
SAVE_METADATA = {  # no date or version stamp
    "svg": {"Date": None},
    "png": {"Software": None},
    "pdf": {"CreationDate": None, "Creator": None, "Producer": None},
}


def style_context(settings: dict):
    """A matplotlib.rc_context with the fonts, line widths, legend frame and grid of `settings`.
    Draw inside it; every rcParam is restored on exit."""
    import matplotlib

    fonts, lines = settings["fonts"], settings["lines"]
    rc = {"axes.titlesize": fonts["title"], "figure.titlesize": fonts["title"], "axes.labelsize": fonts["axis_label"],
          "xtick.labelsize": fonts["tick"], "ytick.labelsize": fonts["tick"], "legend.fontsize": fonts["legend"],
          "legend.title_fontsize": fonts["legend"], "lines.linewidth": lines["width"],
          "lines.markersize": lines["marker_size"], "legend.frameon": settings["legend"]["frame"],
          "axes.grid": settings["axes"]["grid"]}
    if fonts["family"] is not None:
        rc["font.family"] = fonts["family"]
    return matplotlib.rc_context(rc)


def new_figure(settings: dict, nrows: int = 1, ncols: int = 1):
    """(fig, axes) sized from settings["figure"]; axes is one Axes, or an array for a grid. The
    Figure is not registered with pyplot, so it is never shown or kept open by accident."""
    from matplotlib.figure import Figure

    f = settings["figure"]
    fig = Figure(figsize=(f["width"], f["height"]), dpi=f["dpi"], layout=f["layout"])
    return fig, fig.subplots(nrows, ncols, squeeze=True)


def shared_legend(fig, axes, settings: dict):
    """One figure legend for all `axes`, each label once (first occurrence). None if legend.show is
    false. A figure legend can't place itself, so loc "best" becomes upper right (right upper when
    outside, which needs the constrained layout)."""
    import numpy as np

    legend = settings["legend"]
    if not legend["show"]:
        return None
    handles = {}
    for ax in np.atleast_1d(axes).ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    loc = legend["loc"]
    if legend["outside"]:
        loc = "outside " + ("right upper" if loc == "best" else loc)
    elif loc == "best":
        loc = "upper right"
    return fig.legend(list(handles.values()), list(handles), loc=loc, ncols=legend["ncol"], title=legend["title"],
                      frameon=legend["frame"], fontsize=settings["fonts"]["legend"])


def save_figure(fig, path_stem, settings: dict) -> list[Path]:
    """Write `fig` as <path_stem>.<format> for every format in settings["figure"]["formats"] and
    return the paths. SVG, PNG and PDF carry no date or version stamp and SVG ids use a fixed salt, so
    saving the same figure twice gives identical bytes."""
    import matplotlib

    paths = []
    for fmt in settings["figure"]["formats"]:
        path = Path(f"{path_stem}.{fmt}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with matplotlib.rc_context({"svg.hashsalt": SVG_HASHSALT}):
            fig.savefig(path, format=fmt, dpi=settings["figure"]["dpi"], metadata=SAVE_METADATA.get(fmt))
        paths.append(path)
    return paths


def color_for(label, settings: dict, labels) -> str:
    """The colour of `label`: colors.by_label if set there, else the palette colour at the label's
    position in order.labels followed by the rest of `labels` in their given order. With
    order.labels set, a label keeps its colour whichever subset a figure draws."""
    colors = settings["colors"]
    if label in colors["by_label"]:
        return colors["by_label"][label]
    palette = colors["palette"]
    if isinstance(palette, str):
        if palette not in PALETTES:
            raise ValueError(f"unknown palette {palette!r}; known: {sorted(PALETTES)}, or give a list of colours")
        palette = PALETTES[palette]
    position, n_labels = _label_position(label, settings, labels)
    if position >= len(palette):
        raise ValueError(f"{n_labels} labels but {len(palette)} palette colours; add colours or colors.by_label")
    return palette[position]


def marker_for(label, settings: dict, labels) -> str:
    """The marker shape of `label` (a matplotlib marker code): markers.by_label if set there, else
    markers.sequence at the label's position, by the same order rules as color_for, so a label's
    shape and colour go together and stay fixed whichever subset a figure draws."""
    markers = settings["markers"]
    if label in markers["by_label"]:
        return markers["by_label"][label]
    position, n_labels = _label_position(label, settings, labels)
    if position >= len(markers["sequence"]):
        raise ValueError(f"{n_labels} labels but {len(markers['sequence'])} marker shapes; "
                         "add shapes to markers.sequence or markers.by_label")
    return markers["sequence"][position]


def _label_position(label, settings: dict, labels) -> tuple[int, int]:
    """(position of `label`, number of labels) in order.labels followed by the rest of `labels` in
    their given order; shared by color_for and marker_for."""
    order = list(dict.fromkeys([*settings["order"]["labels"], *labels]))
    if label not in order:
        raise ValueError(f"label {label!r} is not in order.labels or the given labels")
    return order.index(label), len(order)


def display_name(kind: str, key, settings: dict):
    """display_names.<kind>[key], or the key itself when no name is set."""
    if kind not in settings["display_names"]:
        raise ValueError(f"unknown display-name kind {kind!r}; known: {sorted(settings['display_names'])}")
    return settings["display_names"][kind].get(key, key)


def point_labels(ax, xs, ys, labels, settings: dict) -> list:
    """Write each label's display name next to its point, shifted by label_offsets[label] points
    (DEFAULT_LABEL_OFFSET otherwise), aligned away from the point. Returns the annotations."""
    out = []
    for x, y, label in zip(xs, ys, labels, strict=True):
        dx, dy = settings["label_offsets"].get(label, DEFAULT_LABEL_OFFSET)
        out.append(ax.annotate(display_name("label", label, settings), (x, y), xytext=(dx, dy),
                               textcoords="offset points", fontsize=settings["fonts"]["annotation"],
                               ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top"))
    return out
