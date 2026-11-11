import dataclasses

import numpy as np
import pytest

from silent_rag import run as runner
from silent_rag.inject.base import REGISTRY
from silent_rag.inject.controls import FullSwap, MixedTruncation
from silent_rag.pipeline import Retriever
from silent_rag.run import load_plugins
from test_inject_f1 import CLEAN, PINNED, ShiftedEncoder, SmallEncoder
from test_pipeline import HashEncoder, toy_dataset


def fake(encoder, seen=None):
    class Fake(Retriever):
        def __init__(self, model, **kw):
            if seen is not None:
                seen.append((model, kw))
            super().__init__(model, encoder=encoder, **kw)

    return Fake


def clean_state(**settings):
    clean = Retriever("clean/model", encoder=HashEncoder(), **settings)
    return clean, clean.embed_corpus(toy_dataset())


def attach(monkeypatch, cls, encoder, params, seen=None, **settings):
    monkeypatch.setattr(cls, "retriever_cls", fake(encoder, seen))
    clean, state = clean_state(**settings)
    inj = cls(**params)
    inj.attach(clean, state)
    return inj, state


TRUNC = {"max_seq_length": 512}


def test_both_are_registered_as_harmless():
    load_plugins()
    assert REGISTRY["H_mixed_truncation"] is MixedTruncation
    assert REGISTRY["H_full_swap"] is FullSwap
    assert not MixedTruncation.harmful and not FullSwap.harmful


# mixed truncation


@pytest.mark.parametrize("n", [0, -1, 256.0, "512", True, None])
def test_truncation_rejects_a_length_that_is_not_a_positive_integer(n):
    with pytest.raises(ValueError, match="max_seq_length must be a positive integer"):
        MixedTruncation.check_config({"max_seq_length": n}, CLEAN)


def test_truncation_at_the_clean_length_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="512 is the clean setting"):
        attach(monkeypatch, MixedTruncation, ShiftedEncoder(), TRUNC, max_seq_length=512)


def test_truncation_keeps_the_clean_model_and_settings_except_the_length(monkeypatch):
    seen = []
    settings = dict(normalize=False, query_prefix="query: ", max_seq_length=256, revision="cleanrev", batch_size=8)
    attach(monkeypatch, MixedTruncation, ShiftedEncoder(), TRUNC, seen, **settings)
    assert seen == [("clean/model", {**settings, "max_seq_length": 512, "device": "cpu"})]


@pytest.mark.parametrize("severity, count", [(0.1, 4), (0.5, 20), (0.9, 36)])
def test_truncation_replaces_a_severity_fraction_of_rows_and_grows_with_it(monkeypatch, severity, count):
    inj, state = attach(monkeypatch, MixedTruncation, ShiftedEncoder(), TRUNC, max_seq_length=256)
    damaged = inj.apply(state, severity, seed=0)
    rows = np.flatnonzero((damaged.doc_embeddings != state.doc_embeddings).any(axis=1))
    assert len(rows) == count
    np.testing.assert_array_equal(damaged.doc_embeddings[rows], inj.alt_embeddings[rows])
    smaller = inj.apply(state, 0.05, seed=0)
    assert set(np.flatnonzero((smaller.doc_embeddings != state.doc_embeddings).any(axis=1))) <= set(rows)
    assert damaged.query_embeddings is state.query_embeddings


def test_truncation_with_another_dimension_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="32-dim vectors but the index is 64-dim"):
        attach(monkeypatch, MixedTruncation, SmallEncoder(), TRUNC, max_seq_length=256)


# full swap


@pytest.mark.parametrize("params", PINNED, ids=["minilm", "bge"])
def test_full_swap_accepts_the_pinned_f1_partners(params):
    assert FullSwap.check_config(params, CLEAN) is None


def test_full_swap_rejects_the_clean_model():
    with pytest.raises(ValueError, match="is the clean model; H_full_swap needs a different model"):
        FullSwap.check_config({"alt_model": CLEAN["model"], "alt_revision": PINNED[0]["alt_revision"]}, CLEAN)


def test_full_swap_moves_documents_and_queries_together(monkeypatch):
    inj, state = attach(monkeypatch, FullSwap, ShiftedEncoder(), PINNED[0])
    swapped = inj.apply(state, 1.0, seed=0)
    np.testing.assert_array_equal(swapped.doc_embeddings, inj.doc_embeddings)
    np.testing.assert_array_equal(swapped.query_embeddings, inj.query_embeddings)
    assert not np.array_equal(swapped.query_embeddings, state.query_embeddings)
    for field in ("doc_ids", "doc_texts", "query_ids", "query_texts"):
        assert getattr(swapped, field) is getattr(state, field), field


def test_full_swap_ranks_like_a_clean_system_on_the_new_model(monkeypatch):
    """Consistent, just different: a rolled encoder ranks exactly like the clean one."""
    inj, state = attach(monkeypatch, FullSwap, ShiftedEncoder(), PINNED[0])
    clean = Retriever("clean/model", encoder=HashEncoder())
    before, _, _ = clean.search(state, k=3, query_embeddings=state.query_embeddings)
    swapped = inj.apply(state, 1.0, seed=0)
    after, _, _ = clean.search(swapped, k=3, query_embeddings=swapped.query_embeddings)
    assert [r[0] for r in after] == [r[0] for r in before]


@pytest.mark.parametrize("severity", [0.5, 0.9])
def test_full_swap_only_runs_at_full_severity(monkeypatch, severity):
    inj, state = attach(monkeypatch, FullSwap, ShiftedEncoder(), PINNED[0])
    with pytest.raises(ValueError, match="only runs at severity 1.0"):
        inj.apply(state, severity, seed=0)


def test_config_with_a_partial_full_swap_is_rejected_before_anything_runs():
    load_plugins()
    with pytest.raises(SystemExit, match=r"H_full_swap only runs at severities \[1.0\]"):
        runner.failure_params({"swap": {"injector": "H_full_swap", "severities": [0.5, 1.0], **PINNED[0]}})


def test_full_swap_with_another_dimension_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="32-dim vectors but the index is 64-dim"):
        attach(monkeypatch, FullSwap, SmallEncoder(), PINNED[0])


# both


@pytest.mark.parametrize("cls, params, severity", [(MixedTruncation, TRUNC, 0.5), (FullSwap, PINNED[0], 1.0)])
def test_apply_before_attach_is_an_error(cls, params, severity):
    _, state = clean_state()
    with pytest.raises(RuntimeError, match=r"call attach\(\) before apply\(\)"):
        cls(**params).apply(state, severity, seed=0)


@pytest.mark.parametrize("cls, params, severity", [(MixedTruncation, TRUNC, 0.5), (FullSwap, PINNED[0], 1.0)])
def test_apply_on_another_state_is_an_error(monkeypatch, cls, params, severity):
    inj, state = attach(monkeypatch, cls, ShiftedEncoder(), params, max_seq_length=256)
    with pytest.raises(ValueError, match="doc_ids don't match"):
        inj.apply(dataclasses.replace(state, doc_ids=state.doc_ids[::-1]), severity, seed=0)


def test_both_run_through_the_runner_as_harmless_controls(monkeypatch):
    for cls in (MixedTruncation, FullSwap):
        monkeypatch.setattr(cls, "retriever_cls", fake(ShiftedEncoder()))
    cfg = {
        "datasets": ["nfcorpus"], "model": "clean/model", "window_size": 10, "trials": 2,
        "failures": {
            "trunc512": {"injector": "H_mixed_truncation", "severities": [0.5], "max_seq_length": 512},
            "swap": {"injector": "H_full_swap", "severities": [1.0], **PINNED[0]},
        },
        "detectors": ["ks_top1", "re_embed"],
    }
    df = runner.run(cfg, loader=lambda name: toy_dataset(),
                    make_retriever=lambda: Retriever("clean/model", encoder=HashEncoder(), max_seq_length=256))
    controls = df[df.failure != "clean"]
    assert set(controls.failure) == {"trunc512", "swap"}
    assert not controls.expected_harmful.any()
    # a consistent swap ranks like clean here, so no window loses nDCG
    assert (df[df.failure == "swap"].ndcg_drop == 0).all()
    # but it does move what re_embed sees: the stored vectors aren't the clean model's
    re = df[df.detector == "re_embed"]
    assert (re[re.failure == "swap"].score > re[re.failure == "clean"].score.max()).all()
