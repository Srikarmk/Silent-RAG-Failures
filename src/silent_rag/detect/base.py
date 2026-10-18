from __future__ import annotations

from abc import ABC, abstractmethod

from silent_rag.types import Observation


class Detector(ABC):
    """Label-free detector. Higher score = more likely something is broken.

    fit() gets a window from the clean system, score() gets a window that may
    or may not be damaged. Neither gets qrels.
    """

    name: str

    def attach(self, retriever) -> None:
        """Called once before fit() with the clean retriever, i.e. the model the
        system is supposed to be running. Only detectors that re-embed need it."""

    @abstractmethod
    def fit(self, reference: Observation) -> None: ...

    @abstractmethod
    def score(self, window: Observation) -> float: ...


REGISTRY: dict[str, type[Detector]] = {}


def register(cls: type[Detector]) -> type[Detector]:
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} needs a `name` class attribute to be registered")
    if name in REGISTRY and REGISTRY[name] is not cls:
        raise ValueError(f"two detectors registered as {name!r}: {REGISTRY[name].__name__} and {cls.__name__}")
    REGISTRY[name] = cls
    return cls
