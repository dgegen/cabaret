from datetime import datetime
from pathlib import Path
from typing import Any

from astropy.io import fits


class FITSManager:
    """A class to manage FITS file operations, including creating headers with
    observatory metadata.

    Examples
    --------
    >>> from cabaret.fits_manager import FITSManager
    >>> from cabaret import Observatory, Sources
    >>> observatory = Observatory(name="My Observatory")
    >>> sources = Sources.get_test_source()
    >>> ra, dec = sources.ra.deg.mean(), sources.dec.deg.mean()
    >>> image = observatory.generate_image(
    ...     ra=ra, dec=dec, exp_time=30, sources=sources, seed=0
    ... )
    >>> hdu_list = FITSManager.get_fits_from_array(
    ...     image=image, observatory=observatory, exp_time=30, ra=ra, dec=dec
    ... )
    >>> hdu_list[0].header['DEC']

    """

    @staticmethod
    def save(
        image,
        file_path: str | Path,
        observatory,
        exp_time: float | None = None,
        ra: float | None = None,
        dec: float | None = None,
        dateobs: datetime | None = None,
        light: int | None = None,
        user_header: dict[str, Any] | fits.Header | None = None,
        overwrite: bool = True,
    ):
        """
        Save a numpy array image to a FITS file using observatory metadata.

        Parameters
        ----------
        image : numpy.ndarray
            The image data to save.
        file_path : str
            The path to the FITS file to write.
        observatory : Observatory
            The observatory instance for header metadata.
        exp_time, ra, dec, dateobs, light, user_header : optional
            Passed to get_fits_from_array.
        overwrite : bool, optional
            Whether to overwrite existing file (default: True).
        """
        hdul = FITSManager.to_hdu_list(
            image=image,
            observatory=observatory,
            exp_time=exp_time,
            ra=ra,
            dec=dec,
            dateobs=dateobs,
            light=light,
            user_header=user_header,
        )
        hdul.writeto(file_path, overwrite=overwrite)

    @staticmethod
    def to_hdu_list(
        image,
        observatory,
        exp_time: float | None = None,
        ra: float | None = None,
        dec: float | None = None,
        dateobs: datetime | None = None,
        light: int | None = None,
        user_header: dict[str, Any] | fits.Header | None = None,
    ) -> fits.HDUList:
        """
        Create a FITS HDUList from a numpy array, optionally with a header.

        Parameters
        ----------
        image : numpy.ndarray
            The image data to convert to FITS.
        observatory : Observatory
            The observatory instance for header metadata.
        exp_time : float, optional
            Exposure time in seconds.
        ra : float, optional
            Right ascension of the image center in degrees.
        dec : float, optional
            Declination of the image center in degrees.
        dateobs : datetime, optional
            The observation date and time.
        light : int, optional
            Light pollution level (1-5).
        user_header : dict or fits.Header, optional
            Additional header keywords to add.

        Returns
        -------
        fits.HDUList
            The created FITS HDUList.
        """

        header = FITSManager.get_header_from_observatory(
            observatory,
            user_header=user_header,
        )
        FITSManager.add_image_info_to_header(
            header, exp_time=exp_time, ra=ra, dec=dec, dateobs=dateobs, light=light
        )
        hdu = fits.PrimaryHDU(data=image, header=header)
        hdul = fits.HDUList([hdu])
        return hdul

    @staticmethod
    def add_image_info_to_header(
        header: fits.Header,
        exp_time: float | None = None,
        ra: float | None = None,
        dec: float | None = None,
        dateobs: datetime | None = None,
        light: int | None = None,
    ):
        """Add image-specific info to a FITS header."""
        if exp_time is not None:
            header["EXPTIME"] = (float(exp_time), "Exposure time in seconds")
        if ra is not None:
            header["RA"] = (float(ra), "Right ascension of image center [deg]")
        if dec is not None:
            header["DEC"] = (float(dec), "Declination of image center [deg]")
        if light is not None:
            header["LIGHTLVL"] = (float(light), "Light pollution level (1-5)")
        if dateobs is None:
            dateobs = datetime.now()
        header["DATE-OBS"] = (dateobs.isoformat(), "UTC datetime start of exposure")

    @staticmethod
    def get_header_from_observatory(
        observatory,
        user_header: dict[str, Any] | None = None,
    ) -> "fits.Header":
        """
        Create a FITS header populated with metadata from an Observatory instance.

        Parameters
        ----------
        observatory : Observatory
            The observatory instance to extract metadata from.
        extra : dict, optional
            Additional header keywords to add.

        Returns
        -------
        fits.Header
            The populated FITS header.
        """
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError("Please install astropy to use FITSHeaderBuilder.")

        header = fits.Header()
        header["OBSNAME"] = (observatory.name, "Observatory name")

        # Camera-related fields
        camera = observatory.camera
        header["INSTRUME"] = (getattr(camera, "name", "Camera"), "Instrument used")
        header["XBINNING"] = (
            getattr(camera, "bin_x", 1),
            "Binning level along the X-axis",
        )
        header["YBINNING"] = (
            getattr(camera, "bin_y", 1),
            "Binning level along the Y-axis",
        )
        header["XPIXSZ"] = (
            getattr(camera, "pitch", None),
            "Pixel Width in microns (after binning)",
        )
        header["YPIXSZ"] = (
            getattr(camera, "pitch", None),
            "Pixel Height in microns (after binning)",
        )
        header["CAM-DNAM"] = (
            getattr(camera, "name", "cabaret_camera"),
            "Short name of Camera driver",
        )

        header["FOCUSER"] = (
            getattr(observatory.focuser, "name", "cabaret_focuser"),
            "Focuser name",
        )
        header["FOCUSPOS"] = (observatory.focuser.position, "Focuser position")
        header["TELESCOP"] = (
            getattr(observatory.telescope, "name", "cabaret_telescope"),
            "Telescope name",
        )
        telescope = observatory.telescope
        header["FOCALLEN"] = (
            telescope.focal_length * 1000,
            "[mm] Focal length of telescope",
        )
        header["APTDIA"] = (
            telescope.diameter * 1000,
            "[mm] Aperture diameter of telescope",
        )
        header["APTAREA"] = (
            telescope.collecting_area * 1e6,
            "[mm^2] Aperture area of telescope",
        )
        header["SITE"] = (
            getattr(observatory.site, "name", "cabaret_site"),
            "Site name",
        )
        # Add more fields as needed from the observatory/camera/telescope/site

        if user_header:
            # Accept both dict and fits.Header
            if isinstance(user_header, dict):
                for k, v in user_header.items():
                    header[k] = v
            elif isinstance(user_header, fits.Header):
                for card in user_header.cards:
                    header[card.keyword] = (card.value, card.comment)
            else:
                raise TypeError("user_header must be a dict or fits.Header")

        try:
            from importlib.metadata import version

            header["CABARET"] = (version("cabaret"), "Version of Cabaret")
        except Exception:
            header["CABARET"] = ("unknown", "Version of Cabaret")
        return header
