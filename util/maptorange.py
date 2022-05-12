import numpy as np


def do(X, min_expected, max_expected):
    X = np.asarray(X)
    X = (X - min_expected) / (max_expected - min_expected)
    return X