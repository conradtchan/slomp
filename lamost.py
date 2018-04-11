
""" Utility functions to deal with LAMOST observed and model spectra. """

import numpy as np
from astropy.io import fits
from scipy import interpolate


def read_dr3_spectrum(path, common_dispersion=None, bounds_error=False):
    r"""
    Read a spectrum produced as part of the third data release of LAMOST.

    :param path:
        The local path to the spectrum.

    :returns:
        A four-length tuple containing the rest-frame dispersion array (vacuum 
        wavelengths), the flux array, the inverse variance array, and a metadata
        dictionary.
    """

    header_keys = ("helio_rv", "z", "z_err")

    with fits.open(path) as image:
        # data array indices:
        # flux, inverse variance, wavelength, andmask, ormask.
        flux, ivar, dispersion, and_mask, or_mask = image[0].data

        # Create a meta dictionary that contains things we will probably care 
        # about later on, and the path so that we can trace provenance of other
        # things as needed.
        meta = dict(path=path)
        for header_key in header_keys:
            meta[header_key] = image[0].header[header_key.upper()]

    # Use the OR mask to set the inverse variances to zero for any pixels with
    # indications of being bad. For example, the bit mask meanings are:
    # 1 : BADCCD : bad pixel on CCD
    # 2 : BADPROFILE : bad profile in extraction
    # 3 : NOSKY : no sky information at this wavelength
    # 4 : BRIGHTSKY : sky level too high
    # 5 : BADCENTER : fibre trace out of the CCD
    # 6 : NODATA : no good data.

    # From http://dr3.lamost.org/doc/data-production-description

    # These are all bad things. And the LAMOST pipeline people are more familiar
    # with the data than we are. So let's believe them.

    rest_dispersion = dispersion * (1 - meta["z"])
    ivar[or_mask > 0] = 0.0

    if common_dispersion is not None:
        flux = (interpolate.interp1d(rest_dispersion, flux,
            bounds_error=bounds_error, fill_value=1))(common_dispersion)
        ivar = (interpolate.interp1d(rest_dispersion, ivar,
            bounds_error=bounds_error, fill_value=0))(common_dispersion)

        rest_dispersion = common_dispersion
        ivar[ivar < 0] = 0

    assert np.all(ivar >= 0), "negative inverse variances"
    assert np.all(np.isfinite(flux)), "non-finite fluxes"

    return (rest_dispersion, flux, ivar, meta)


def gaussian_weight_matrix(dispersion, L):
    r""" 
    Design matrix of weights for the Gaussian-smoothed spectrum.

    :param dispersion:
        An array of the dispersion (wavelengths).

    :param L:
        The width of the Gaussian in pixels.

    :returns:
        The design matrix of weights for the smoothed spectrum.
    """
    return np.exp(-0.5*(dispersion[:,None]-dispersion[None,:])**2/L**2)


def smooth_spec(dispersion, flux, ivar, L):
    r"""
    Smooth a spectrum with a running Gaussian.

    :param dispersion:
        An array of the dispersion (wavelengths).

    :param flux:
        The observed flux array.

    :param ivar:
        The inverse variances of the fluxes.

    :param L:
        The width of the Gaussian in pixels.
    
    :returns:
        An array of smoothed fluxes.
    """
    w = gaussian_weight_matrix(dispersion, L)
    denominator = np.dot(ivar, w.T)
    numerator = np.dot(flux*ivar, w.T)
    bad_pixel = denominator == 0
    smoothed = np.zeros(numerator.shape)
    smoothed[~bad_pixel] = numerator[~bad_pixel] / denominator[~bad_pixel]
    return smoothed


def continuum_normalize(dispersion, flux, ivar, L=50):
    r""" 

    Pseudo-continuum-normalise a spectrum by dividing by a Gaussian-weighted
    smoothed spectrum.

    :param dispersion:
        An array of the dispersion (wavelengths).

    :param flux:
        The observed flux array.

    :param ivar:
        The inverse variances of the fluxes.

    :param L: [optional]
        The width of the Gaussian in pixels.

    :returns:
        A two-length tuple containing the pseudo-continuum-normalised fluxes 
        and the associated inverse variances.
    """

    smoothed_spec = smooth_spec(dispersion, flux, ivar, L)
    norm_flux = flux / smoothed_spec
    norm_ivar = smoothed_spec * ivar * smoothed_spec

    bad_pixel = ~np.isfinite(norm_flux)
    norm_flux[bad_pixel] = 1.0
    norm_ivar[bad_pixel] = 0.0

    return (norm_flux, norm_ivar)