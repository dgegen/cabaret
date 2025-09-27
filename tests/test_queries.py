import pytest
from astropy.coordinates import SkyCoord

from cabaret.queries import GaiaQuery

from .utils import has_internet


@pytest.mark.skipif(not has_internet(), reason="Requires internet")
def test_get_sources_basic():
    center = SkyCoord(ra=10.68458, dec=41.26917, unit="deg")
    radius = 0.05
    sources = GaiaQuery.get_sources(center, radius, limit=10, timeout=30)
    assert len(sources) <= 10
    assert sources is not None


@pytest.mark.skipif(not has_internet(), reason="Requires internet")
def test_get_sources_timeout():
    center = SkyCoord(ra=10.68458, dec=41.26917, unit="deg")
    radius = 0.05
    with pytest.raises(TimeoutError):
        GaiaQuery.get_sources(center, radius, limit=10, timeout=0.0001)
