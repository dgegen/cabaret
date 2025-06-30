from dataclasses import dataclass


@dataclass
class Site:
    sky_background: float = 150  # for I+z band in Paranal, e-/m2/arcsec2/s
    seeing: float = 1.3  # arcsec
