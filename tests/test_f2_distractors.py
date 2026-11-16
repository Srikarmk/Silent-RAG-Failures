import dataclasses

import numpy as np
import pytest

from silent_rag import data
from silent_rag import run as runner
from silent_rag.inject.base import REGISTRY
from silent_rag.inject.f2_distractors import Distractors
from silent_rag.pipeline import Retriever
from silent_rag.run import load_plugins
from test_inject_f1 import CLEAN
from test_pipeline import HashEncoder, toy_dataset


def clean_state(**settings):
    clean = Retriever("clean/model", encoder=HashEncoder(), **settings)
    return clean, clean.embed_corpus(toy_dataset())


def test_registered_as_a_harmful_failure():
    load_plugins()
    assert REGISTRY["F2_distractors"] is Distractors and Distractors.harmful






def distractors(monkeypatch, n=40, **settings):
    corpus = {f"a{i}": f"unrelated{i} debate{i} filler" for i in range(n)}
    monkeypatch.setattr(Distractors, "load", staticmethod(lambda source: corpus))
    clean, state = clean_state(**settings)
    inj = Distractors()
    inj.attach(clean, state)
    return inj, state


def test_distractors_sources_are_never_an_evaluation_dataset():
    assert not set(data.DISTRACTOR_SOURCES) & (runner.DEV_DATASETS | runner.HELD_OUT)
    assert not set(data.DISTRACTOR_SOURCES) & set(data.SOURCES)


def test_distractors_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="source must be one of"):
        Distractors.check_config({"source": "scifact"}, CLEAN)
    assert Distractors.check_config({}, CLEAN) is None


@pytest.mark.parametrize("severity, count", [(0.1, 4), (0.5, 20), (0.9, 36)])
def test_distractors_adds_a_severity_fraction_of_the_corpus_and_keeps_every_original_row(monkeypatch, severity, count):
    inj, state = distractors(monkeypatch)
    grown = inj.apply(state, severity, seed=0)
    n = len(state.doc_ids)
    assert len(grown.doc_ids) == len(grown.doc_texts) == len(grown.doc_embeddings) == n + count
    assert grown.doc_ids[:n] == list(state.doc_ids)
    np.testing.assert_array_equal(grown.doc_embeddings[:n], state.doc_embeddings)
    assert all(d.startswith("arguana:") for d in grown.doc_ids[n:])
    assert grown.query_embeddings is state.query_embeddings


def test_distractors_grows_a_superset_with_severity_and_follows_the_seed(monkeypatch):
    inj, state = distractors(monkeypatch)

    def added(severity, seed):
        return set(inj.apply(state, severity, seed).doc_ids[len(state.doc_ids):])

    assert added(0.1, 0) < added(0.5, 0) < added(0.9, 0)
    assert added(0.5, 0) == added(0.5, 0) != added(0.5, 1)


def test_distractors_embeds_with_the_clean_model_so_rows_match_their_text(monkeypatch):
    inj, state = distractors(monkeypatch)
    grown = inj.apply(state, 0.5, seed=0)
    fresh = Retriever("clean/model", encoder=HashEncoder()).encode(grown.doc_texts)
    np.testing.assert_allclose(grown.doc_embeddings, fresh, atol=1e-6)


def test_distractors_with_too_few_source_docs_is_an_error(monkeypatch):
    inj, state = distractors(monkeypatch, n=10)
    with pytest.raises(ValueError, match="adds 20 docs but arguana only has 10"):
        inj.apply(state, 0.5, seed=0)


def test_distractors_extends_parent_ids_on_a_chunked_index(monkeypatch):
    inj, state = distractors(monkeypatch)
    chunked = dataclasses.replace(state, parent_ids=list(state.doc_ids))
    grown = inj.apply(chunked, 0.5, seed=0)
    assert grown.parent_ids[: len(state.doc_ids)] == list(state.doc_ids)
    assert grown.parent_ids[len(state.doc_ids):] == grown.doc_ids[len(state.doc_ids):]


def test_distractors_run_through_the_runner_as_a_failure_re_embed_cannot_see(monkeypatch):
    corpus = {f"a{i}": f"unrelated{i} debate{i} filler" for i in range(40)}
    monkeypatch.setattr(Distractors, "load", staticmethod(lambda source: corpus))
    cfg = {
        "datasets": ["nfcorpus"], "model": "clean/model", "window_size": 10, "trials": 2,
        "failures": {"distractors": {"injector": "F2_distractors", "severities": [0.5, 0.9]}},
        "detectors": ["ks_top1", "re_embed", "domain_classifier_retrieved_docs"],
    }
    df = runner.run(cfg, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("clean/model", encoder=HashEncoder()))
    off = df[df.failure == "distractors"]
    assert off.expected_harmful.all()
    assert (off.ndcg_drop == 0).all()  # no toy query shares a word with these distractors
    # every added row is what the clean model makes of its text, so re_embed sees nothing
    assert (off[off.detector == "re_embed"].score == 0).all()
