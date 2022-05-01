import numpy as np
from sklearn.linear_model import LinearRegression as LR
from numpy.lib.stride_tricks import as_strided
from sklearn.utils import check_random_state


def form_lag_matrix(X, T, stride=1, stride_tricks=True, rng=None, writeable=False):
    """Form the data matrix with `T` lags.

    Parameters
    ----------
    X : ndarray (n_time, N)
        Timeseries with no lags.
    T : int
        Number of lags.
    stride : int or float
        If stride is an `int`, it defines the stride between lagged samples used
        to estimate the cross covariance matrix. Setting stride > 1 can speed up the
        calculation, but may lead to a loss in accuracy. Setting stride to a `float`
        greater than 0 and less than 1 will random subselect samples.
    rng : NumPy random state
        Only used if `stride` is a float.
    stride_tricks : bool
        Whether to use numpy stride tricks to form the lagged matrix or create
        a new array. Using numpy stride tricks can can lower memory usage, especially for
        large `T`. If `False`, a new array is created.
    writeable : bool
        For testing. You should not need to set this to True. This function uses stride tricks
        to form the lag matrix which means writing to the array will have confusing behavior.
        If `stride_tricks` is `False`, this flag does nothing.

    Returns
    -------
    X_with_lags : ndarray (n_lagged_time, N * T)
        Timeseries with lags.
    """
    if not isinstance(stride, int) or stride < 1:
        if not isinstance(stride, float) or stride <= 0. or stride >= 1.:
            raise ValueError('stride should be an int and greater than or equal to 1 or a float ' +
                             'between 0 and 1.')
    N = X.shape[1]
    frac = None
    if isinstance(stride, float):
        frac = stride
        stride = 1
    n_lagged_samples = (len(X) - T) // stride + 1
    if n_lagged_samples < 1:
        raise ValueError('T is too long for a timeseries of length {}.'.format(len(X)))
    if stride_tricks:
        X = np.asarray(X, dtype=float, order='C')
        shape = (n_lagged_samples, N * T)
        strides = (X.strides[0] * stride,) + (X.strides[-1],)
        X_with_lags = as_strided(X, shape=shape, strides=strides, writeable=writeable)
    else:
        X_with_lags = np.zeros((n_lagged_samples, T * N))
        for i in range(n_lagged_samples):
            X_with_lags[i, :] = X[i * stride:i * stride + T, :].flatten()
    if frac is not None:
        rng = check_random_state(rng)
        idxs = np.sort(rng.choice(n_lagged_samples, size=int(np.ceil(n_lagged_samples * frac)),
                                  replace=False))
        X_with_lags = X_with_lags[idxs]

    return X_with_lags


def linear_decode_r2(X_train, Y_train, X_test, Y_test, decoding_window=1, offset=0):
    """Train a linear model on the training set and test on the test set.
    This will work with batched training data and/or batched test data.
    X_train : ndarray (time, channels) or (batches, time, channels)
        Feature training data for regression.
    Y_train : ndarray (time, channels) or (batches, time, channels)
        Target training data for regression.
    X_test : ndarray (time, channels) or (batches, time, channels)
        Feature test data for regression.
    Y_test : ndarray (time, channels) or (batches, time, channels)
        Target test data for regression.
    decoding_window : int
        Number of time samples of X to use for predicting Y (should be odd). Centered around
        offset value.
    offset : int
        Temporal offset for prediction (0 is same-time prediction).
    """

    if isinstance(X_train, np.ndarray) and X_train.ndim == 2:
        X_train = [X_train]
    if isinstance(Y_train, np.ndarray) and Y_train.ndim == 2:
        Y_train = [Y_train]

    if isinstance(X_test, np.ndarray) and X_test.ndim == 2:
        X_test = [X_test]
    if isinstance(Y_test, np.ndarray) and Y_test.ndim == 2:
        Y_test = [Y_test]

    X_train_lags = [form_lag_matrix(Xi, decoding_window) for Xi in X_train]
    X_test_lags = [form_lag_matrix(Xi, decoding_window) for Xi in X_test]

    Y_train = [Yi[decoding_window // 2:] for Yi in Y_train]
    Y_train = [Yi[:len(Xi)] for Yi, Xi in zip(Y_train, X_train_lags)]
    if offset >= 0:
        Y_train = [Yi[offset:] for Yi in Y_train]
    else:
        Y_train = [Yi[:Yi.shape[0] + offset] for Yi in Y_train]

    Y_test = [Yi[decoding_window // 2:] for Yi in Y_test]
    Y_test = [Yi[:len(Xi)] for Yi, Xi in zip(Y_test, X_test_lags)]
    if offset >= 0:
        Y_test = [Yi[offset:] for Yi in Y_test]
    else:
        Y_test = [Yi[:Yi.shape[0] + offset] for Yi in Y_test]

    if offset >= 0:
        X_train_lags = [Xi[:Xi.shape[0] - offset] for Xi in X_train_lags]
        X_test_lags = [Xi[:Xi.shape[0] - offset] for Xi in X_test_lags]
    else:
        X_train_lags = [Xi[-offset:] for Xi in X_train_lags]
        X_test_lags = [Xi[-offset:] for Xi in X_test_lags]

    if len(X_train_lags) == 1:
        X_train_lags = X_train_lags[0]
    else:
        X_train_lags = np.concatenate(X_train_lags)

    if len(Y_train) == 1:
        Y_train = Y_train[0]
    else:
        Y_train = np.concatenate(Y_train)

    if len(X_test_lags) == 1:
        X_test_lags = X_test_lags[0]
    else:
        X_test_lags = np.concatenate(X_test_lags)

    if len(Y_test) == 1:
        Y_test = Y_test[0]
    else:
        Y_test = np.concatenate(Y_test)

    model = LR().fit(X_train_lags, Y_train)
    r2 = model.score(X_test_lags, Y_test)
    return r2

