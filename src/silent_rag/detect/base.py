from __future__ import annotations

from abc import ABC, abstractmethod

from silent_rag.types import Observation


class Detector(ABC):
    """Label-free detector. Higher score = more likely something is broken.

    fit() gets a window from the clean system, score() gets a window that may
    or may not be damaged. Neither gets qrels.
    """

    name: str

    @abstractmethod
    def fit(self, reference: Observation) -> None: ...

    @abstractmethod
    def score(self, window: Observation) -> float: ...


REGISTRY: dict[str, type[Detector]] = {}


def register(cls: type[Detector]) -> type[Detector]:
    REGISTRY[cls.name] = cls
    return cls
