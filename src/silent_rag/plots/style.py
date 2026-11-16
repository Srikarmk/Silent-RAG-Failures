"""Figure style settings: schema defaults and the five-layer merge (spec section 5.13).

For figure f, later layers win: built-in global defaults (SCHEMA_DEFAULTS) < built-in defaults
for f (FIGURE_DEFAULTS[f], set by f's module) < the YAML global section < the YAML figures.f
section < keyword arguments. Mappings merge key by key; lists and scalars replace. A keyword
argument of None means "not given"; an explicit null in YAML overrides. Unknown keys raise.
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
