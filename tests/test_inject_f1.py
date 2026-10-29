import inspect

import numpy as np
import pytest

from silent_rag import pipeline
from silent_rag.inject.f1_partial_upgrade import PartialUpgrade
from silent_rag.pipeline import Retriever
from test_pipeline import HashEncoder, toy_dataset

ALT = {"alt_model": "alt/model", "alt_revision": "abc123"}


@pytest.mark.parametrize("given", [{}, {"alt_model": ALT["alt_model"]}, {"alt_revision": ALT["alt_revision"]}])
def test_alt_model_and_revision_are_required(given):
    # bind, not construct: apply() is still abstract, so construction fails for any args
    with pytest.raises(TypeError, match="missing a required argument"):
        inspect.signature(PartialUpgrade).bind(**given)
    inspect.signature(PartialUpgrade).bind(**ALT)


class Attachable(PartialUpgrade):
    def apply(self, state, severity, seed):  # apply() is the next step, attach() is tested on its own
        raise NotImplementedError


class SmallEncoder:
    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True):
        return np.ones((len(texts), 32), dtype=np.float32)


def attached(monkeypatch, encoder, seen=None, **clean_settings):
    class Alt(Retriever):
        def __init__(self, model, **kw):
            if seen is not None:
                seen.append((model, kw))
            super().__init__(model, encoder=encoder, **kw)

    monkeypatch.setattr(Attachable, "retriever_cls", Alt)
    clean = Retriever("clean/model", encoder=HashEncoder(), **clean_settings)
    state = clean.embed_corpus(toy_dataset())
    inj = Attachable(**ALT)
    inj.attach(clean, state)
    return inj, state


def test_alt_retriever_keeps_the_clean_settings_except_model_and_revision(monkeypatch):
    seen = []
    clean = dict(normalize=False, query_prefix="query: ", max_seq_length=128, revision="cleanrev", batch_size=8)
    inj, state = attached(monkeypatch, HashEncoder(), seen, **clean)
    assert seen == [(ALT["alt_model"], {**clean, "revision": ALT["alt_revision"], "device": "cpu"})]
    assert inj.alt_embeddings.shape == state.doc_embeddings.shape


def test_alt_corpus_is_encoded_once_then_loaded_from_the_cache(monkeypatch):
    first, second = HashEncoder(), HashEncoder()
    a, _ = attached(monkeypatch, first)
    b, _ = attached(monkeypatch, second)
    assert (first.calls, second.calls) == (1, 0)
    np.testing.assert_array_equal(a.alt_embeddings, b.alt_embeddings)
    assert len(list(pipeline.CACHE_DIR.glob("f1_alt-*.npy"))) == 1


def test_alt_model_with_another_dimension_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="alt/model gives 32-dim vectors but the index is 64-dim"):
        attached(monkeypatch, SmallEncoder())
