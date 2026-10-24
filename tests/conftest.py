import pytest

from silent_rag import pipeline


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Keep test runs out of the real embedding cache."""
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "embeddings")
