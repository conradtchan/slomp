
"""
An example script to pseudo-continuum-normalise a single LAMOST spectrum and 
save the result to disk.
"""

import matplotlib.pyplot as plt
import pickle

import lamost

# When loading spectra, let's resample them onto a common wavelength scale.
# This makes it easier for any data-driven model or classifier.

with open("common_vac_wavelengths.pkl", "rb") as fp:
    common_dispersion = pickle.load(fp)


# wget http://dr3.lamost.org/sas/fits/B5591606/spec-55916-B5591606_sp03-051.fits.gz
# gunzip spec-55916-B5591606_sp03-051.fits.gz
input_path = "spec-55916-B5591606_sp03-051.fits"
dispersion, flux, ivar, meta = lamost.read_dr3_spectrum(input_path,
    common_dispersion=common_dispersion)

norm_flux, norm_ivar = lamost.continuum_normalize(dispersion, flux, ivar)

fig, ax = plt.subplots(2)
ax[0].plot(dispersion, flux, c="k")
ax[1].plot(dispersion, norm_flux, c="k")

output_path = "{}.pkl".format(input_path[:-5])
with open(output_path, "wb") as fp:
    # We don't save the dispersion array because it is already stored in
    # common_vac_wavelengths.pkl
    pickle.dump((norm_flux, norm_ivar, meta), fp)
