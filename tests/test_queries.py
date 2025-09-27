import pytest
from astropy.coordinates import SkyCoord

from cabaret.queries import get_gaia_sources


def test_get_gaia_sources_basic():
    center = SkyCoord(ra=10.68458, dec=41.26917, unit="deg")
    radius = 0.05
    sources = get_gaia_sources(center, radius, limit=10, timeout=30)
    assert len(sources) <= 10
    assert sources is not None


def test_get_gaia_sources_timeout():
    center = SkyCoord(ra=10.68458, dec=41.26917, unit="deg")
    radius = 0.05
    with pytest.raises(TimeoutError):
        get_gaia_sources(center, radius, limit=10, timeout=0.0001)
