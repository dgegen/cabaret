import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from cabaret.sources import Sources

COORDS = np.array([[10.64, 41.26], [10.68, 41.22]])
FLUXES = np.array([169435.6, 52203.9])


@pytest.fixture
def example_sources():
    return Sources(SkyCoord(COORDS, unit="deg"), FLUXES)


def test_sources_init_and_len(example_sources):
    sources = example_sources
    assert len(sources) == 2
    assert np.allclose(sources.fluxes, FLUXES)
    assert np.allclose(sources.coords.ra.deg, COORDS[:, 0])
    assert np.allclose(sources.coords.dec.deg, COORDS[:, 1])


def test_sources_ra_dec_properties(example_sources):
    sources = example_sources
    assert np.allclose(sources.ra.deg, COORDS[:, 0])
    assert np.allclose(sources.dec.deg, COORDS[:, 1])


def test_sources_to_pixel(example_sources):
    sources = example_sources
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [10.65, 41.25]
    wcs.wcs.crpix = [100, 100]
    wcs.wcs.cdelt = [-0.0002777778, 0.0002777778]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    pixels = sources.to_pixel(wcs)
    assert pixels.shape == (2, 2)


def test_sources_invalid_init():
    # coords is not of type SkyCoord
    with pytest.raises(ValueError):
        Sources(COORDS, FLUXES)

    # Length mismatch
    with pytest.raises(ValueError):
        Sources(SkyCoord(COORDS, unit="deg"), FLUXES[:-1])


def test_sources_from_array():
    sources = Sources.from_arrays(COORDS[:, 0], COORDS[:, 1], FLUXES)
    assert np.allclose(sources.ra.deg, COORDS[:, 0])
    assert np.allclose(sources.dec.deg, COORDS[:, 1])
    assert np.allclose(sources.fluxes, FLUXES)


def test_center():
    coords = SkyCoord(ra=[179.999, 180.001], dec=[10.68, 10.689], unit="deg")
    sources = Sources(coords, fluxes=np.array([169_435.6, 92_203.9]))

    ra_center, dec_center = sources.center
    assert np.isclose(ra_center, 180.0)
    assert np.isclose(dec_center, 10.6845)


def test_center_with_wrap():
    coords = SkyCoord(ra=[359.999, 0.001], dec=[10.68, 10.689], unit="deg")
    sources = Sources(coords, fluxes=np.array([169_435.6, 92_203.9]))

    ra_center, _ = sources.center
    assert np.isclose(ra_center, 0.0)
