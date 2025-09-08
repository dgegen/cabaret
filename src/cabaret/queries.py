from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astroquery.gaia import Gaia


def gaia_launch_job_with_timeout(query, timeout=None, **kwargs):
    """
    Launch a Gaia job and return its results, optionally enforcing a timeout.

    Parameters
    ----------
    query : str
        The query string passed to Gaia.launch_job.
    timeout : float or None, optional
        Maximum number of seconds to wait for Gaia.launch_job to complete.
        If None, the job is run on the main thread (no thread overhead).
    **kwargs
        Additional keyword arguments forwarded to Gaia.launch_job.

    Returns
    -------
    object
        The result returned by job.get_results().

    Raises
    ------
    TimeoutError
        If `timeout` is not None and the call does not complete within `timeout`.
    """
    # Run directly on the main thread when no timeout is requested to avoid
    # unnecessary thread creation and to preserve original callstacks/tracebacks.
    if timeout is None:
        job = Gaia.launch_job(query, **kwargs)
        return job.get_results()

    with ThreadPoolExecutor() as executor:
        future = executor.submit(Gaia.launch_job, query, **kwargs)
        try:
            job = future.result(timeout=timeout)
            return job.get_results()
        except TimeoutError:
            raise TimeoutError("Gaia query timed out.")


def gaia_radecs(
    center: tuple[float, float] | SkyCoord,
    fov: float | Quantity,
    limit: int = 100000,
    circular: bool = True,
    tmass: bool = False,
    dateobs: datetime | None = None,
    timeout: float | None = None,
) -> np.ndarray:
    """
    Query the Gaia archive to retrieve the RA-DEC coordinates of stars
    within a given field-of-view (FOV) centered on a given sky position.

    Parameters
    ----------
    center : tuple or astropy.coordinates.SkyCoord
        The sky coordinates of the center of the FOV.
        If a tuple is given, it should contain the RA and DEC in degrees.
    fov : float or astropy.units.Quantity
        The field-of-view of the FOV in degrees. If a float is given,
        it is assumed to be in degrees.
    limit : int, optional
        The maximum number of sources to retrieve from the Gaia archive.
        By default, it is set to 10000.
    circular : bool, optional
        Whether to perform a circular or a rectangular query.
        By default, it is set to True.
    tmass : bool, optional
        Whether to retrieve the 2MASS J magnitudes catelog.
        By default, it is set to False.
    dateobs : datetime.datetime, optional
        The date of the observation. If given, the proper motions of the sources
        will be taken into account. By default, it is set to None.
    timeout : float, optional
        The maximum time to wait for the Gaia query to complete, in seconds.
        If None, there is no timeout. By default, it is set to None.

    Returns
    -------
    np.ndarray
        An array of shape (n, 2) containing the RA-DEC coordinates
        of the retrieved sources in degrees.

    Raises
    ------
    ImportError
        If the astroquery package is not installed.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> from twirl import gaia_radecs
    >>> center = SkyCoord(ra=10.68458, dec=41.26917, unit='deg')
    >>> fov = 0.1
    >>> radecs = gaia_radecs(center, fov)
    """

    if isinstance(center, SkyCoord):
        ra = center.ra.deg
        dec = center.dec.deg
    else:
        ra, dec = center

    if not isinstance(fov, u.Quantity):
        fov = fov * u.deg

    if fov.ndim == 1:
        ra_fov, dec_fov = fov.to(u.deg).value
    else:
        ra_fov = dec_fov = fov.to(u.deg).value

    radius = np.max([ra_fov, dec_fov]) / 2

    select_cols = [
        "gaia.ra",
        "gaia.dec",
        "gaia.pmra",
        "gaia.pmdec",
        "gaia.phot_rp_mean_flux",
    ]
    joins = []
    where = []
    order_by = "gaia.phot_rp_mean_flux DESC"

    # add TMASS columns/joins/order if requested
    if tmass:
        select_cols.append("tmass.j_m")
        joins.extend(
            [
                "INNER JOIN gaiadr2.tmass_best_neighbour AS tmass_match "
                + "ON tmass_match.source_id = gaia.source_id",
                "INNER JOIN gaiadr1.tmass_original_valid AS tmass "
                + "ON tmass.tmass_oid = tmass_match.tmass_oid",
            ]
        )
        order_by = "tmass.j_m"

    # spatial filter (circular or rectangular)
    if circular:
        where.append(
            f"1=CONTAINS(POINT('ICRS', {ra}, {dec}), "
            f"CIRCLE('ICRS', gaia.ra, gaia.dec, {radius}))"
        )
    else:
        where.append(
            f"gaia.ra BETWEEN {ra - ra_fov / 2} AND {ra + ra_fov / 2} "
            f"AND gaia.dec BETWEEN {dec - dec_fov / 2} AND {dec + dec_fov / 2}"
        )

    select_clause = ", ".join(select_cols)
    joins_clause = "\n".join(joins)
    where_clause = " AND ".join(where)

    query = f"""
    SELECT TOP {limit} {select_clause}
    FROM gaiadr2.gaia_source AS gaia
    {joins_clause}
    WHERE {where_clause}
    ORDER BY {order_by}
    """

    table = gaia_launch_job_with_timeout(query, timeout=timeout)

    # add proper motion to ra and dec
    if dateobs is not None:
        # calculate fractional year
        dateobs = dateobs.year + (dateobs.timetuple().tm_yday - 1) / 365.25  # type: ignore

        years = dateobs - 2015.5  # type: ignore
        table["ra"] += years * table["pmra"] / 1000 / 3600
        table["dec"] += years * table["pmdec"] / 1000 / 3600

    if tmass:
        table.remove_rows(np.isnan(table["j_m"]))
        return (
            np.array([table["ra"].value.data, table["dec"].value.data]).T,
            table["j_m"].value.data,
        )
    else:
        table.remove_rows(np.isnan(table["phot_rp_mean_flux"]))
        return (
            np.array([table["ra"].value.data, table["dec"].value.data]).T,
            table["phot_rp_mean_flux"].value.data,
        )
