from __future__ import annotations

from abc import ABC, abstractmethod

from silent_rag.types import RetrievalState


class Injector(ABC):
    """A failure (or a harmless change) applied at some severity.

    Rules:
    - never modify `state` in place, return a new one
    - same (severity, seed) must give the same result
    - severity is the fraction of the corpus or query traffic affected
    - query positions stay fixed. See RetrievalState for when to reset
      query_embeddings
    """

    name: str
    harmful: bool = True  # False for the H1-H3 harmless changes

    def attach(self, retriever, state: RetrievalState) -> None:
        """Called once per run with the clean retriever, before any apply(). The same
        instance then handles every severity and trial, so expensive setup (a second
        model's embeddings, a re-chunked corpus) belongs here or in a cache on self."""

    @abstractmethod
    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState: ...


REGISTRY: dict[str, type[Injector]] = {}


def register(cls: type[Injector]) -> type[Injector]:
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} needs a `name` class attribute to be registered")
    if name in REGISTRY and REGISTRY[name] is not cls:
        raise ValueError(f"two injectors registered as {name!r}: {REGISTRY[name].__name__} and {cls.__name__}")
    REGISTRY[name] = cls
    return cls
