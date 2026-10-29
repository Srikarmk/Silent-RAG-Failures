import inspect

import pytest

from silent_rag.inject.f1_partial_upgrade import PartialUpgrade

ALT = {"alt_model": "alt/model", "alt_revision": "abc123"}


@pytest.mark.parametrize("given", [{}, {"alt_model": ALT["alt_model"]}, {"alt_revision": ALT["alt_revision"]}])
def test_alt_model_and_revision_are_required(given):
    # bind, not construct: apply() is still abstract, so construction fails for any args
    with pytest.raises(TypeError, match="missing a required argument"):
        inspect.signature(PartialUpgrade).bind(**given)
    inspect.signature(PartialUpgrade).bind(**ALT)
