import doctest

import matplotlib
import pytest

matplotlib.use("Agg")

import cabaret.camera
import cabaret.focuser
import cabaret.observatory
import cabaret.queries
import cabaret.site
import cabaret.sources
import cabaret.telescope

pytestmark = pytest.mark.filterwarnings("ignore:.*FigureCanvasAgg.*:UserWarning")


@pytest.mark.parametrize(
    "mod",
    [
        cabaret.sources,
        cabaret.queries,
        cabaret.camera,
        cabaret.site,
        cabaret.focuser,
        cabaret.telescope,
        cabaret.observatory,
    ],
    ids=[
        cabaret.sources.__name__,
        cabaret.queries.__name__,
        cabaret.camera.__name__,
        cabaret.site.__name__,
        cabaret.focuser.__name__,
        cabaret.telescope.__name__,
        cabaret.observatory.__name__,
    ],
)
def test_doctests(mod):
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    result = doctest.testmod(mod, optionflags=flags)
    assert result.failed == 0, f"{result.failed} doctest failures in {mod.__name__}"
