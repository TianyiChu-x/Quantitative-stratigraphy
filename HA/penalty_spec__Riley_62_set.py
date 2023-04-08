import numpy as np


def produce_list():
    n_biostrat = 4522
    # n_biostrat = 62
    biostrat = np.arange(1, n_biostrat + 1)

    n_pmag = 0
    pmag = 63

    n_dates = 0
    dates = np.matrix([[109, 2, 110, 1, 100], [111, 2, 112, 1, 100], [113, 2, 114, 1, 100]])

    n_ashes = 0
    ashes = np.matrix([[68, 100], [69, 100]])
    n_continuous = 0
    continuous = np.matrix([[70, 5], [71, 5]])
    penalty_spec_62 = {
        "n_biostrat": n_biostrat,
        "biostrat": biostrat,
        "n_pmag": n_pmag,
        "pmag": pmag,
        "n_dates": n_dates,
        "dates": dates,
        "n_ashes": n_ashes,
        "ashes": ashes,
        "n_continuous": n_continuous,
        "continuous": continuous
    }

    return penalty_spec_62



"""
n_biostrat = 62
biostrat = range(1, 63)

n_pmag = 0
pmag = 63

n_dates = 0
dates = [[109, 0, 110, 1, 100], [111, 2, 112, 1, 100], [113, 2, 114, 1, 100]]

n_ashes = 0
ashes = [[68, 100], [69, 100]]

n_continuous = 0
continuous = [[70, 5], [71, 5]]

penalty_spec_62 = {"n_biostrat": n_biostrat, "biostrat": biostrat, "n_pmag": n_pmag, "pmag": pmag, "n_dates": n_dates, "dates": dates, "n_ashes": n_ashes, "ashes": ashes, "n_continuous": n_continuous, "continuous": continuous}
"""
