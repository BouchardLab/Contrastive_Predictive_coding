import numpy as np


def compute_R2(X_inf, X_true):
    """
    compute R2 value
    :param X_inf: inferred X dim T x N
    :param X_true: true X dim T x N
    :return:
    """
    return 1 - np.sum((X_inf - X_true)**2) / np.sum(
        (X_true - np.mean(X_true, axis=0)) ** 2)