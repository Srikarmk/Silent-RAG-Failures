"""llm_judge (#26): an LLM decides whether retrieved documents are relevant, for a
sample of each window's queries. The score is the share of judged pairs that come
back irrelevant, so higher means more likely broken, and the clean reference
window's share is the baseline the score is compared against.

Unlike the drift detectors this one measures quality, not change, which is what
the two-stage monitor (#74) needs from its second stage. It is also the only
detector that costs money: each check records the dollar cost of its judgements
as `cost_usd` (what the check costs uncached; reruns hit the cache and pay nothing).

Judgements are cached by silent_rag.judge. With `live: false` (the default) the
detector never calls the API: pairs must already be judged, e.g. by a batch run,
and a missing pair is an error. `live: true` judges missing pairs as it goes.
"""

from __future__ import annotations

import zlib

from silent_rag import judge
from silent_rag.detect.base import Detector, register
from silent_rag.types import Observation


@register
class LLMJudge(Detector):
    name = "llm_judge"
    client = None  # tests set a fake; otherwise built on first live call

    def __init__(self, model: str = judge.DEFAULT_MODEL, queries: int = 10, depth: int = 5,
                 effort: str = judge.DEFAULT_EFFORT, live: bool = False, seed: int = 0):
        self.model, self.queries, self.depth = model, queries, depth
        self.effort, self.live, self.seed = effort, live, seed
        self.cache = None

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        model = params.get("model", judge.DEFAULT_MODEL)
        if model not in judge.PRICES:
            raise ValueError(f"model must be one of {sorted(judge.PRICES)} (priced), got {model!r}")
        for name in ("queries", "depth"):
            v = params.get(name, 1)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer, got {v!r}")
        k = run_cfg.get("k", 10)
        if params.get("depth", 5) > k:
            raise ValueError(f"depth {params['depth']} is deeper than the k={k} results a window has")

    def pairs(self, window: Observation) -> list[tuple[str, str]]:
        if window.query_texts is None or window.doc_texts is None or window.doc_ids is None:
            raise ValueError("llm_judge needs query_texts, doc_ids and doc_texts in the Observation")
        if window.parent_ids is not None:
            raise ValueError("llm_judge judges whole documents and doesn't handle a chunked index yet")
        text = dict(zip(window.doc_ids, window.doc_texts))
        # the same sample of positions under every condition: a hash of the window's ids
        key = zlib.crc32("\n".join(window.query_ids).encode()) ^ self.seed
        order = sorted(range(len(window.query_ids)),
                       key=lambda i: zlib.crc32(f"{i}:{window.query_ids[i]}".encode(), key))[: self.queries]
        return [(window.query_texts[i], text[d]) for i in sorted(order) for d in window.topk_ids[i][: self.depth]]

    def _judge(self, window: Observation) -> tuple[float, float]:
        if self.cache is None:
            self.cache = judge.Cache()
        pairs = self.pairs(window)
        keys = [judge.key(self.model, q, d) for q, d in pairs]
        if self.live:
            got = judge.judge_live(pairs, self.model, self.cache, client=self.client, effort=self.effort)
        else:
            got = self.cache.get(keys)
            missing = len(set(keys) - set(got))
            if missing:
                raise RuntimeError(
                    f"llm_judge: {missing} of {len(set(keys))} pairs aren't judged yet; judge them first "
                    "(silent_rag.judge, batch) or set live: true"
                )
        verdicts = [got[k].relevant for k in keys if got[k].relevant is not None]
        if not verdicts:
            raise RuntimeError("llm_judge: the model gave no verdict for any pair in this window")
        return 1.0 - sum(verdicts) / len(verdicts), sum(got[k].cost_usd for k in set(keys))

    def fit(self, reference: Observation) -> None:
        self.reference_score_, _ = self._judge(reference)

    def score(self, window: Observation) -> float:
        share, self.last_cost_usd = self._judge(window)
        return share
