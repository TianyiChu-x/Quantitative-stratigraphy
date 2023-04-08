import numpy as np


def ColumnRangeExtension(y):
    lp = np.where(y == 1)
    lpv = np.arange(lp[0][0], lp[0][-1] + 1)
    return sum(y[lpv] == 0)


def NetRangeExtension(y):
    result = []
    for i in range(y.shape[1]):
        total = ColumnRangeExtension(y[:, i])
        result.append(total)
    return sum(result)


def ftransitions2(x):
    # ???
    y = np.diff(x)
    y = np.sign(y)
    yalt = np.diff(y)
    reversals = sum(abs(yalt) / 2)
    return reversals


"""import numpy as np


def ColumnRangeExtension(y):
    lp = np.where(y > 0)[0]
    lpv = np.arange(lp[0], lp[len(lp) - 1] + 1)
    return np.sum(np.where(y[lpv] == 0, np.NaN, 0), axis=0)


def NetRangeExtension(y):
    return np.sum(np.apply_along_axis(ColumnRangeExtension, 1, y), axis=0)


def NetRangeExtension2(y):
    rx = np.sum(np.apply_along_axis(ColumnRangeExtension, 1, y), axis=0)
    return rx[0]


def fdistance(x):
    nd = np.sum(np.sqrt(np.diff(x[:, 1]) ** 2))
    return nd


def ftransitions(x):
    y = np.diff(x[:, 1])
    y = np.sign(y)
    yalt = y[:-1] - y[-len(y)]
    reversals = np.sum(yalt != 0)
    return reversals


def ftransitions2(x):
    y = np.diff(x[np.logical_not(np.isnan(x))])
    y = np.sign(y)
    yalt = np.diff(y)
    reversals = np.sum(np.abs(yalt) / 2)
    return reversals"""
