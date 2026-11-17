"""The runner's query-shift rules: an injector may ask another query of the dataset at a
position, ground truth pairs the question asked with its own clean result, and the
reference window is never asked."""

import dataclasses

import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.inject.base import REGISTRY, Injector
from silent_rag.inject.controls import QueryTopicShift
from silent_rag.pipeline import Retriever
from silent_rag.run import load_plugins
from test_inject_f1 import CLEAN
from test_pipeline import HashEncoder, toy_dataset


def fake_retriever():
    return Retriever("fake", encoder=HashEncoder())


def run(failures, detectors=("ks_top1",), **extra):
    cfg = {"datasets": ["nfcorpus"], "window_size": 10, "trials": 2, "failures": failures,
           "detectors": list(detectors), **extra}
    return runner.run(cfg, loader=lambda name: toy_dataset(), make_retriever=fake_retriever)


class _Shift(Injector):
    """Moves every pool position to the first pool query, keeping or breaking the rules."""

    mode = "ok"

    def apply(self, state, severity, seed):
        ids, texts = list(state.query_ids), list(state.query_texts)
        pool = set(self.query_pool)
        target = self.query_pool[0]
        text = state.query_texts[state.query_ids.index(target)]
        if self.mode == "outside":
            target, text = "new-question", "something the dataset never asked"
        if self.mode == "reference":
            target = next(q for q in state.query_ids if q not in pool)
            text = state.query_texts[state.query_ids.index(target)]
        for i, q in enumerate(state.query_ids):
            if q in pool:
                ids[i] = target
                texts[i] = "wrong text" if self.mode == "text" else text
        return dataclasses.replace(state, query_ids=ids, query_texts=texts, query_embeddings=None)


def shift_injector(monkeypatch, mode):
    name = f"test_shift_{mode}"
    monkeypatch.setitem(REGISTRY, name, type(f"Shift_{mode}", (_Shift,), {"name": name, "mode": mode}))
    return name


def test_query_pool_is_set_before_attach_and_excludes_the_reference(monkeypatch):
    seen = {}

    class Spy(Injector):
        name = "test_shift_spy"

        def attach(self, retriever, state):
            seen["pool"] = list(self.query_pool)
            seen["all"] = list(state.query_ids)

        def apply(self, state, severity, seed):
            return state

    monkeypatch.setitem(REGISTRY, Spy.name, Spy)
    run({Spy.name: [0.5]})
    assert len(seen["pool"]) == 30 and set(seen["pool"]) < set(seen["all"])  # 40 queries, reference of 10


def test_shifted_window_is_scored_against_the_question_asked(monkeypatch):
    df = run({shift_injector(monkeypatch, "ok"): [0.5]})
    shifted = df[df.failure != "clean"]
    # the index didn't change, so every question asked gets exactly its clean result
    assert (shifted.ndcg_drop == 0).all() and not shifted.harmful.any()


@pytest.mark.parametrize("mode, message", [
    ("outside", "aren't in the dataset .* no ground truth for them yet"),
    ("reference", "reference-window queries"),
    ("text", "another query's text"),
])
def test_breaking_the_query_shift_rules_is_an_error(monkeypatch, mode, message):
    with pytest.raises(RuntimeError, match=message):
        run({shift_injector(monkeypatch, mode): [0.5]})


# H_query_topic_shift


def attached(n_topics=3, topic=0, pool=None):
    clean = fake_retriever()
    state = clean.embed_corpus(toy_dataset())
    inj = QueryTopicShift(n_topics=n_topics, topic=topic)
    inj.query_pool = pool or list(state.query_ids[10:])
    inj.attach(clean, state)
    return inj, state


def test_registered_as_harmless():
    load_plugins()
    assert REGISTRY["H_query_topic_shift"] is QueryTopicShift and not QueryTopicShift.harmful


@pytest.mark.parametrize("params, message", [
    ({"n_topics": 1}, "n_topics must be an integer >= 2"),
    ({"n_topics": 3.0}, "n_topics must be an integer >= 2"),
    ({"topic": 5}, r"topic must be an integer in \[0, n_topics\)"),
    ({"topic": -1}, r"topic must be an integer in \[0, n_topics\)"),
])
def test_topic_shift_rejects_bad_params(params, message):
    with pytest.raises(ValueError, match=message):
        QueryTopicShift.check_config(params, CLEAN)


def test_topic_shift_needs_the_query_pool():
    clean = fake_retriever()
    with pytest.raises(RuntimeError, match="needs query_pool"):
        QueryTopicShift().attach(clean, clean.embed_corpus(toy_dataset()))


def test_topic_shift_moves_pool_positions_to_one_topic_only():
    inj, state = attached()
    topic_ids = {state.query_ids[j] for j in inj.topic_positions}
    assert topic_ids <= set(inj.query_pool)
    shifted = inj.apply(state, 1.0, seed=0)
    pool = set(inj.query_pool)
    for i, q in enumerate(state.query_ids):
        if q in pool:
            assert shifted.query_ids[i] in topic_ids
        else:
            assert shifted.query_ids[i] == q  # the reference window is untouched
    # ids, texts and vectors move together
    for i, q in enumerate(shifted.query_ids):
        j = state.query_ids.index(q)
        assert shifted.query_texts[i] == state.query_texts[j]
        np.testing.assert_array_equal(shifted.query_embeddings[i], state.query_embeddings[j])
    assert shifted.doc_embeddings is state.doc_embeddings


def test_topic_shift_grows_a_superset_with_severity_and_follows_the_seed():
    inj, state = attached()

    def moved(severity, seed):
        out = inj.apply(state, severity, seed)
        return {i for i, (a, b) in enumerate(zip(out.query_ids, state.query_ids)) if a != b}

    assert moved(0.2, 0) <= moved(0.5, 0) <= moved(0.9, 0)
    assert moved(0.5, 0) == moved(0.5, 0)
    assert len(moved(0.9, 0)) > len(moved(0.2, 0))


def test_topic_shift_runs_through_the_runner_with_no_drop_and_moves_the_query_classifier():
    df = run({"shift": {"injector": "H_query_topic_shift", "severities": [0.9], "n_topics": 3}},
             detectors=["ks_top1", "domain_classifier_query"], window_size=10, trials=3)
    shifted = df[df.failure == "shift"]
    assert (shifted.ndcg_drop == 0).all() and not shifted.expected_harmful.any()
    q = df[df.detector == "domain_classifier_query"]
    assert q[q.failure == "shift"].score.mean() > q[q.failure == "clean"].score.mean()
