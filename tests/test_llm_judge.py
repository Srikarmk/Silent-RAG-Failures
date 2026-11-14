import json
from types import SimpleNamespace

import numpy as np
import pytest

from silent_rag import judge
from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.detect.llm_judge import LLMJudge
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.pipeline import Retriever
from silent_rag.run import load_plugins
from test_pipeline import HashEncoder, ScrambleDocs, toy_dataset


def message(relevant, stop="end_turn", tin=500, tout=100):
    text = json.dumps({"relevant": relevant}) if relevant is not None else "not json"
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=tin, output_tokens=tout))


class FakeClient:
    """Relevant when the question's first word is in the document, like the toy data."""

    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **params):
        self.calls.append(params)
        body = params["messages"][0]["content"]
        q = body.split("<question>\n")[1].split("\n</question>")[0]
        d = body.split("<document>\n")[1].split("\n</document>")[0]
        return message(q.split()[0] in d.split())


def test_params_use_structured_output_low_effort_and_a_fixed_system_prompt():
    p = judge.params("claude-haiku-5-5", "q text", "d text")
    assert p["system"] == judge.SYSTEM and p["output_config"]["effort"] == "low"
    assert p["output_config"]["format"]["schema"]["required"] == ["relevant"]
    assert "<question>\nq text\n</question>" in p["messages"][0]["content"]


def test_cost_uses_the_price_list_and_batch_halves_it():
    assert judge.cost("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert judge.cost("claude-haiku-5-5", 0, 1_000_000) == pytest.approx(0.5)
    assert judge.cost("claude-opus-5-5", 1_000_000, 1_000_000, batch=True) == pytest.approx(12.0)
    with pytest.raises(ValueError, match="no price"):
        judge.cost("other-model", 1, 1)


@pytest.mark.parametrize("msg, relevant", [
    (message(True), True), (message(False), False),
    (message(True, stop="refusal"), None), (message(None), None),
])
def test_parse_reads_the_verdict_and_bills_even_without_one(msg, relevant):
    j = judge.parse(msg, "claude-haiku-5-5")
    assert j.relevant is relevant and j.cost_usd == pytest.approx(judge.cost("claude-haiku-5-5", 500, 100))


def test_live_judging_caches_so_a_rerun_makes_no_calls(tmp_path):
    cache, client = judge.Cache(tmp_path / "j.sqlite"), FakeClient()
    pairs = [("apple pie", "apple tart recipe"), ("pear", "plum jam")]
    first = judge.judge_live(pairs, "claude-haiku-5-5", cache, client)
    assert len(client.calls) == 2 and [j.relevant for j in first.values()] == [True, False]
    judge.judge_live(pairs, "claude-haiku-5-5", judge.Cache(tmp_path / "j.sqlite"), client)
    assert len(client.calls) == 2
    judge.judge_live(pairs, "claude-opus-5-5", cache, client)  # another model is another judgement
    assert len(client.calls) == 4


def test_agreement_counts_any_positive_grade_and_skips_unjudged():
    qrels = {"q1": {"a": 2, "b": 0}, "q2": {"c": 1}}
    rows = [("q1", "a"), ("q1", "b"), ("q2", "c"), ("q2", "x")]
    out = judge.agreement(rows, [True, False, None, True], qrels)
    assert out["unjudged"] == 1 and out["accuracy"] == pytest.approx(2 / 3)
    assert out["precision"] == pytest.approx(0.5) and out["recall"] == pytest.approx(1.0)


def test_estimate_needs_no_api_and_scales_with_text():
    small = judge.estimate([("q", "d" * 400)], "claude-opus-5-5")
    big = judge.estimate([("q", "d" * 4000)], "claude-opus-5-5")
    assert big["cost_usd"] >= small["cost_usd"] and small["pairs"] == 1


# the detector


@pytest.fixture
def fake_judge(monkeypatch):
    load_plugins()
    client = FakeClient()
    monkeypatch.setattr(LLMJudge, "client", client)
    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)
    return client


def run(detector_params, **cfg):
    base = {"datasets": ["nfcorpus"], "window_size": 10, "trials": 2, "failures": {ScrambleDocs.name: [1.0]},
            "detectors": [{"llm_judge": {"model": "claude-haiku-5-5", **detector_params}}]}
    return runner.run({**base, **cfg}, loader=lambda name: toy_dataset(),
                      make_retriever=lambda: Retriever("fake", encoder=HashEncoder()))


def test_detector_scores_damage_higher_and_records_dollars(fake_judge):
    df = run({"live": True, "queries": 5, "depth": 3})
    clean, broken = df[df.failure == "clean"], df[df.failure != "clean"]
    assert broken.score.mean() > clean.score.mean()
    assert (df.cost_usd > 0).all()
    # 5 queries x 3 docs, at most 15 distinct pairs per check
    assert (df.cost_usd <= 15 * judge.cost("claude-haiku-5-5", 500, 100) + 1e-12).all()


def test_rerun_hits_the_cache_and_offline_mode_needs_no_client(fake_judge, monkeypatch):
    run({"live": True})
    calls = len(fake_judge.calls)
    monkeypatch.setattr(LLMJudge, "client", None)
    again = run({"live": False})  # same windows, all judged already
    assert len(fake_judge.calls) == calls and len(again) > 0


def test_offline_mode_refuses_unjudged_pairs(fake_judge):
    with pytest.raises(RuntimeError, match="aren't judged yet"):
        run({"live": False})


@pytest.mark.parametrize("params, message", [
    ({"model": "gpt-x"}, "must be one of"),
    ({"queries": 0}, "queries must be a positive integer"),
    ({"depth": 20}, "deeper than the k=10"),
])
def test_check_config_rejects_bad_params(params, message):
    with pytest.raises(ValueError, match=message):
        LLMJudge.check_config(params, {"k": 10})


def test_registered():
    load_plugins()
    assert DETECTORS["llm_judge"] is LLMJudge


# local models through Ollama


def test_local_models_are_free_and_refuse_the_batch_api(tmp_path):
    assert judge.cost("ollama:qwen3:8b", 10_000, 10_000) == 0.0
    assert judge.known("ollama:qwen3:8b") and not judge.known("ollama:")
    with pytest.raises(ValueError, match="judge local models live"):
        judge.judge_batch([("q", "d")], "ollama:qwen3:8b", judge.Cache(tmp_path / "j.sqlite"))
    assert LLMJudge.check_config({"model": "ollama:gemma3:12b"}, {"k": 10}) is None


def test_ollama_client_sends_the_same_prompt_and_schema_and_reads_the_reply(monkeypatch):
    import urllib.request

    sent = {}

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"message": {"content": '{"relevant": true}'}, "done_reason": "stop",
                               "prompt_eval_count": 420, "eval_count": 7}).encode()

    def fake_urlopen(req, timeout):
        sent.update(json.loads(req.data))
        return Reply()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    msg = judge.OllamaClient().messages.create(**judge.params("ollama:qwen3:8b", "q", "d"))
    assert sent["model"] == "qwen3:8b" and sent["format"] == judge.SCHEMA and sent["think"] is False
    assert sent["messages"][0] == {"role": "system", "content": judge.SYSTEM}
    j = judge.parse(msg, "ollama:qwen3:8b")
    assert j.relevant is True and j.input_tokens == 420 and j.cost_usd == 0.0


# cost in seconds (#27)


class SlowClient(FakeClient):
    """Each request takes a fixed, measurable time."""

    def create(self, **params):
        import time
        time.sleep(0.01)
        return super().create(**params)


def test_a_cached_rerun_reports_the_same_seconds_as_judging(monkeypatch):
    load_plugins()
    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)
    monkeypatch.setattr(LLMJudge, "client", SlowClient())
    first = run({"live": True, "queries": 3, "depth": 2})
    assert (first.cost_s >= 0.01).all()  # at least one uncached request's worth per check
    monkeypatch.setattr(LLMJudge, "client", None)
    again = run({"live": False, "queries": 3, "depth": 2})  # every pair cached
    assert np.allclose(again.cost_s.to_numpy(), first.cost_s.to_numpy())


def test_pairs_without_a_recorded_time_give_nan_not_zero(fake_judge):
    run({"live": True})
    cache = judge.Cache()
    cache.db.execute("UPDATE judgements SET seconds = NULL")
    cache.db.commit()
    again = run({"live": False})
    assert again.cost_s.isna().all()


def test_an_old_cache_gets_the_seconds_column(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE judgements (key TEXT PRIMARY KEY, model TEXT, relevant INTEGER, "
               "input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, created REAL)")
    db.execute("INSERT INTO judgements VALUES ('k', 'm', 1, 10, 2, 0.0, 0.0)")
    db.commit()
    db.close()
    got = judge.Cache(path).get(["k"])["k"]
    assert got.relevant is True and got.seconds is None
    judge.Cache(path).put("m", {"k2": judge.Judgement(False, 1, 1, 0.0, 0.5)})
    assert judge.Cache(path).get(["k2"])["k2"].seconds == 0.5


def test_other_detectors_keep_the_wall_clock_time():
    from silent_rag.detect.base import Detector
    assert Detector.last_cost_s is None
