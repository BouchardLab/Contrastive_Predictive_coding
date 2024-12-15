import numpy as np


def linear_alignment(X_inf, X_true):
    beta = np.linalg.lstsq(X_inf, X_true, rcond=None)[0]
    X_trans = np.dot(X_inf, beta)
    return X_trans
