"""F1 partial embedding upgrade (#8): a `severity` fraction of index rows is
re-embedded with a second model while queries stay on the clean one.

The replacement model has no default in code: `alt_model` and `alt_revision`
come from the config, since they decide how much damage F1 does.
"""

from __future__ import annotations

from silent_rag.inject.base import Injector


class PartialUpgrade(Injector):
    name = "F1_partial_upgrade"

    def __init__(self, alt_model: str, alt_revision: str | None):
        self.alt_model = alt_model
        self.alt_revision = alt_revision
