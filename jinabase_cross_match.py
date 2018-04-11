
"""
SAGA sucks.
JINAbase seems a little more useful, even if it is not as complete as SAGA.
(Completeness might be by design: they may have excluded old and dodgy results.)
"""

# I went to http://jinabase.pythonanywhere.com/plot and selected position 
# information, CEMP signature, and literature data (Teff, logg, [M/H], vmic)
# and saved the resulting table to jinabase-retrieved-20180411.txt

# There are 2124 rows (many duplicates).

import numpy as np
from astropy.table import Table

input_path = "jinabase-retrieved-20180411.txt"
data = Table.read(input_path, format="ascii")

# Clean up positions.
data["RA"] = [ra.replace("_", ":") for ra in data["RA"]]
data["DEC"] = [ra.replace("_", ":") for ra in data["DEC"]]


def remove_duplicates(data, label_names=("Teff", "logg", "M/H", "Vmic", "Fe/H", 
    "Mg/H")):
    """
    Remove duplicates by taking median of parameters.
    
    :param data:
        The data table from JINAbase.

    :param label_names:
        Label names to take medians from when grouping. All other label names
        will be taken from the first instance of each star.
    """

    # There are probably better ways to do this (e.g. with some aggregator)
    keep = np.ones(len(data), dtype=bool)
    unique_names = np.unique(data["Name"])

    cleaned_data = data.copy()
    for unique_name in unique_names:
        indices = np.where(data["Name"] == unique_name)[0]
        if len(indices) == 1: continue

        keep[indices[1:]] = False
        for ln in label_names:
            values = []
            for value in data[ln][indices]:
                if str(value).startswith("<"):
                    print("Ignoring upper limit on {} for {}: {} (all data: {})"\
                          .format(ln, unique_name, value, data[ln][indices]))
                    continue

                if value != "*":
                    values.append(value)

            print(unique_name, ln)
            cleaned_data[ln][indices[0]] \
                = np.nanmedian(np.array(values).astype(float))

    cleaned_data = cleaned_data[keep]

    assert len(cleaned_data) == len(unique_names)

    return cleaned_data


cleaned_data = remove_duplicates(data)

output_path = "{}_cleaned.fits".format(input_path[:-4])
cleaned_data.write(output_path)

# Cross-matching the cleaned_data file with LAMOST DR3 stellar file
# (dr3_stellar.fits) with a 5" cone revealed 37 pairs. Damn.

saga_input_path = "saga-retrieve-20180411.tsv"
saga = Table.read(saga_input_path, format="ascii.tab", guess=False, 
    names=("object", "reference", "ra", "dec", "teff", "logg", "mh", "feh"))




