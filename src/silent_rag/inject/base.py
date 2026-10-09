from __future__ import annotations

from abc import ABC, abstractmethod

from silent_rag.types import RetrievalState


class Injector(ABC):
    """A failure (or a harmless change) applied at some severity.

    Rules:
    - never modify `state` in place, return a new one
    - same (severity, seed) must give the same result
    - severity is the fraction of the corpus or query traffic affected
    """

    name: str
    harmful: bool = True  # False for the H1-H3 harmless changes

    @abstractmethod
    def apply(self, state: RetrievalState, severity: float, seed: int) -> RetrievalState: ...


REGISTRY: dict[str, type[Injector]] = {}


def register(cls: type[Injector]) -> type[Injector]:
    REGISTRY[cls.name] = cls
    return cls
