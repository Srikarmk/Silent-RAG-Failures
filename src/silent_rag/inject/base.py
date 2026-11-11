from __future__ import annotations

from abc import ABC, abstractmethod

from silent_rag.types import RetrievalState


class Injector(ABC):
    """A failure (or a harmless change) applied at some severity.

    Rules:
    - never modify `state` in place, return a new one
    - same (severity, seed) must give the same result
    - severity is the fraction of the corpus or query traffic affected
    - the number of queries stays fixed. An injector may put another query
      of the dataset at a position (a query shift), changing query_ids and
      query_texts there together, using only ids from `query_pool`. See
      RetrievalState for when to reset query_embeddings
    """

    name: str
    harmful: bool = True  # False for the harmless changes (negative controls)
    severities: tuple[float, ...] | None = None  # the only severities that make sense, if not all of (0, 1]
    # Set by the runner before attach(): the query ids outside the reference window. An
    # injector that moves traffic to other queries of the dataset may only use these.
    query_pool: list[str] | None = None

    def attach(self, retriever, state: RetrievalState) -> None:
        """Called with the clean retriever and the clean, read-only `state` before any
        apply(); `state` is the same object every apply() later receives. The runner
        makes a fresh instance and calls this once per dataset and seed, not once per
        run. That instance then handles every severity and trial, so expensive setup
        (a second model's corpus embeddings, a re-chunked corpus) belongs here, kept
        on self."""

    @classmethod
    def check_config(cls, params: dict, run_cfg: dict) -> None:
        """Injector-specific config checks, run at bind time before anything is
        embedded and before any instance exists. `params` are this label's constructor
        params; `run_cfg` holds the clean `model` and `revision`. Raise ValueError to
        reject. Does nothing by default."""

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
