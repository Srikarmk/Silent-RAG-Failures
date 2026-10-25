import dataclasses
import importlib

import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.detect import re_embed
from silent_rag.detect.re_embed import ReEmbed
from silent_rag.pipeline import Retriever
from test_pipeline import HashEncoder, ScrambleDocs, toy_dataset


@pytest.fixture
def setup(tmp_path):
    r = Retriever("fake", encoder=HashEncoder())
    state = r.embed_corpus(toy_dataset(200), cache_dir=tmp_path)
    det = ReEmbed(sample=100)
    det.attach(r)
    return r, state, det


def window(state, idx=np.arange(20)):
    return runner.observe(state, idx, [[]] * len(idx), np.zeros((len(idx), 1)), state.query_embeddings[idx])


def test_misaligned_index_is_an_error_not_a_score(setup):
    _, state, det = setup
    rows = np.vstack([state.doc_embeddings, state.doc_embeddings[:50]])
    bad = dataclasses.replace(state, doc_embeddings=rows, doc_ids=state.doc_ids + [f"x{i}" for i in range(50)])
    with pytest.raises(ValueError, match="row-aligned"):
        det.score(window(bad))


def test_l2_sees_unnormalized_vectors_cosine_doesnt(setup):
    r, state, _ = setup
    scaled = state.doc_embeddings * np.linspace(0.5, 3, len(state.doc_embeddings))[:, None]
    damaged = window(dataclasses.replace(state, doc_embeddings=scaled.astype(np.float32)))
    cos, l2 = ReEmbed(sample=100), ReEmbed(sample=100, metric="l2")
    for d in (cos, l2):
        d.attach(r)
    assert cos.score(damaged) < 1e-6
    assert l2.score(damaged) > 0.3
    assert l2.score(window(state)) < 1e-6


def test_correctly_embedded_extra_docs_dont_fire_and_sample_stays_paired(setup):
    r, state, det = setup
    extra = [f"stale doc {i} filler" for i in range(100)]
    grown = dataclasses.replace(
        state,
        doc_ids=state.doc_ids + [f"s{i}" for i in range(100)],
        doc_texts=state.doc_texts + extra,
        doc_embeddings=np.vstack([state.doc_embeddings, r.encode(extra)]),
    )
    assert det.score(window(grown)) < 1e-6
    picked = lambda s: {s.doc_ids[i] for i in det.rows(window(s))}  # noqa: E731
    # appended rows can displace some picks, but the rest are the same docs
    assert len(picked(state) & picked(grown)) >= 50


@pytest.mark.parametrize("kw", [{"sample": 0}, {"sample": -1}, {"sample": 2.5}, {"metric": "dot"}])
def test_bad_parameters_are_rejected(kw):
    with pytest.raises(ValueError):
        ReEmbed(**kw)


def test_config_mistakes_fail_before_any_work():
    def loader(name):
        raise AssertionError("should have failed before loading data")

    for detectors in ([{"re_embed": {"samples": 50}}], [{"re_embed": 50}], ["re_embed", {"re_embed": {"sample": 5}}]):
        cfg = {"datasets": ["nfcorpus"], "detectors": detectors, "window_size": 10, "trials": 1}
        with pytest.raises(SystemExit):
            runner.run(cfg, loader=loader)


def test_module_reload_does_not_break_the_registry():
    importlib.reload(re_embed)
    importlib.reload(re_embed)


def test_floor_makes_clean_exactly_zero_but_keeps_real_damage(setup):
    r, state, det = setup
    # ~3e-6 in cosine: visible without a floor, below the 1e-5 floor, like batch / device noise
    noisy = (state.doc_embeddings + np.random.default_rng(0).normal(0, 3e-4, state.doc_embeddings.shape)).astype(np.float32)
    assert det.score(window(dataclasses.replace(state, doc_embeddings=noisy))) == 0.0
    no_floor = ReEmbed(sample=100, floor=0)
    no_floor.attach(r)
    assert no_floor.score(window(dataclasses.replace(state, doc_embeddings=noisy))) > 0
    assert det.score(window(ScrambleDocs().apply(state, 0.2, seed=0))) > 0.01


@pytest.mark.parametrize("floor", [-1e-3, 1.0, "0.1", True])
def test_bad_floor_is_rejected(floor):
    with pytest.raises(ValueError):
        ReEmbed(floor=floor)
