from __future__ import annotations

from abc import ABC, abstractmethod

from silent_rag.types import Observation


class Detector(ABC):
    """Label-free detector. Higher score = more likely something is broken.

    fit() gets a window from the clean system, score() gets a window that may
    or may not be damaged. Neither gets qrels.
    """

    name: str
    # API-based detectors set this in score(): the dollar cost of that check, recorded as cost_usd
    last_cost_usd: float = 0.0

    def attach(self, retriever) -> None:
        """Called with the clean retriever, i.e. the model the system is supposed
        to be running. The runner makes a fresh detector for each dataset and seed
        and calls this once, just before fit(). Only detectors that re-embed need it."""

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        """Detector-specific config checks, run at bind time before anything is loaded
        or embedded. `params` are this detector's constructor params, already accepted
        by the constructor; `run_cfg` is the whole run config. Raise ValueError to
        reject. Does nothing by default."""

    @abstractmethod
    def fit(self, reference: Observation) -> None: ...

    @abstractmethod
    def score(self, window: Observation) -> float: ...


REGISTRY: dict[str, type[Detector]] = {}


def register(cls: type[Detector]) -> type[Detector]:
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} needs a `name` class attribute to be registered")
    old = REGISTRY.get(name)
    # re-importing the same class (importlib.reload, notebook autoreload) is fine, two different classes aren't
    if old is not None and old is not cls and (old.__module__, old.__qualname__) != (cls.__module__, cls.__qualname__):
        raise ValueError(f"two detectors registered as {name!r}: {REGISTRY[name].__name__} and {cls.__name__}")
    REGISTRY[name] = cls
    return cls
