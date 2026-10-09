"""One row per (condition, trial, detector). Every plot and table, including the
heatmap, should be a query over this table."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class ResultRow:
    dataset: str
    failure: str  # "clean", "F1".."F5", "H1".."H3"
    severity: float  # 0.0 for clean
    trial: int
    window_size: int
    detector: str
    score: float
    ndcg_drop: float
    harmful: bool  # significant drop under the paired bootstrap
    cost_s: float  # wall-clock seconds for this detector check
    cost_usd: float = 0.0  # only non-zero for API-based detectors (LLM judge)


def to_frame(rows: list[ResultRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])
