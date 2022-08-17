import numpy as np


def do(X, min_expected, max_expected):
    X = np.asarray(X)
    X = (X - min_expected) / (max_expected - min_expected)
    return X


def undo(X, min_expected, max_expected):
    X = np.asarray(X)
    X = X * (max_expected - min_expected) + min_expected
    return X
