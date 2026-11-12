"""two_stage (#74): a cheap detector raises the alarm, and only then the LLM judge
(#26) checks whether quality really dropped.

On every window it scores `stage1` (combined_scores by default, #75). If that score is
above `stage1_threshold` the window is flagged and the judge scores it; otherwise the
judge never runs. The score is the judge's share of irrelevant pairs on flagged windows
and 0 on the rest, so one threshold on it (the judge's own) gives the pair's alarms, and
`cost_s` is what a team running it would pay: stage 1 on every window, plus the judge
(uncached) on the flagged ones.

Both thresholds come from clean windows of calibration seeds, never from failures or
controls: `stage1_threshold` here, the judge's alarm threshold at evaluation time. In a
pair, stage 1 should be looser than when used alone, since the judge removes most of its
false alarms (#74: at 20% clean false alarms the pair detects more than at 5%, with the
same false-alarm rate).
"""

from __future__ import annotations

import math
import time

from silent_rag import judge
from silent_rag.detect.base import REGISTRY, Detector, register
from silent_rag.detect.llm_judge import LLMJudge
from silent_rag.types import Observation


@register
class TwoStage(Detector):
    name = "two_stage"

    def __init__(self, stage1_threshold: float, stage1: str = "combined_scores", model: str = judge.DEFAULT_MODEL,
                 queries: int = 100, depth: int = 5, effort: str = judge.DEFAULT_EFFORT, live: bool = False,
                 seed: int = 0):
        self.stage1_name, self.stage1_threshold = stage1, float(stage1_threshold)
        self.judge = LLMJudge(model=model, queries=queries, depth=depth, effort=effort, live=live, seed=seed)

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        t = params.get("stage1_threshold")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError(f"stage1_threshold must be a finite number set on clean windows, got {t!r}")
        stage1 = params.get("stage1", "combined_scores")
        if stage1 in (cls.name, LLMJudge.name):
            raise ValueError(f"stage1 must be a cheap detector, not {stage1!r}")
        if stage1 not in REGISTRY:
            raise ValueError(f"stage1 {stage1!r} isn't a registered detector")
        if hasattr(REGISTRY[stage1], "check_config"):
            REGISTRY[stage1].check_config({}, run_cfg)
        LLMJudge.check_config({k: v for k, v in params.items() if k not in ("stage1", "stage1_threshold")}, run_cfg)

    def attach(self, retriever) -> None:
        self.stage1 = REGISTRY[self.stage1_name]()
        self.stage1.attach(retriever)
        self.judge.attach(retriever)

    def fit(self, reference: Observation) -> None:
        if not hasattr(self, "stage1"):
            self.stage1 = REGISTRY[self.stage1_name]()
        self.stage1.fit(reference)
        self.judge.fit(reference)

    def score(self, window: Observation) -> float:
        start = time.perf_counter()
        first = self.stage1.score(window)
        stage1_s = time.perf_counter() - start
        self.flagged = first > self.stage1_threshold
        if not self.flagged:
            self.last_cost_s, self.last_cost_usd = stage1_s, 0.0
            return 0.0
        share = self.judge.score(window)
        self.last_cost_s = stage1_s + self.judge.last_cost_s
        self.last_cost_usd = self.judge.last_cost_usd
        return share
