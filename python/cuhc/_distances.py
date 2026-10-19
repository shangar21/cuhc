"""Pairwise distances on the GPU."""

import numpy as np

from . import _lib

# Must match MetricId in src/capi.cu. Aliases follow scikit-learn's names.
METRIC_IDS = {
    "euclidean": 0,
    "l2": 0,
    "sqeuclidean": 1,
    "manhattan": 2,
    "cityblock": 2,
    "l1": 2,
    "chebyshev": 3,
    "minkowski": 4,
    "cosine": 5,
}


def validate_metric(metric, p):
    if metric not in METRIC_IDS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRIC_IDS)}")
    if metric == "minkowski" and not p > 0:
        raise ValueError(f"Minkowski p must be positive, got {p}")


def as_points(array, name):
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D (points x features), got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


FLOAT32_MAX = float(np.finfo(np.float32).max)


def check_distance_range(X, metric, p=2.0, Y=None, scale=1.0):
    """Raises ValueError if some distance (times `scale`) could exceed float32's range.

    Distances are computed in float32, and an overflow would silently give inf
    or NaN and with them a meaningless tree. The bound is exact per feature:
    |x_k - y_k| is at most the feature's range, so e.g. a squared Euclidean
    distance is at most sum_k range_k^2. Data far from the origin is fine as
    long as its spread isn't huge. `scale` covers factors applied after the
    metric (Ward's cluster sizes, HDBSCAN's 1 / alpha).
    """
    values = X if Y is None else np.vstack([X, Y])
    with np.errstate(over="ignore", invalid="ignore"):
        spread = values.max(0).astype(np.float64) - values.min(0)
        magnitude = np.abs(values).max(0).astype(np.float64)
        kind = METRIC_IDS[metric]
        if kind in (METRIC_IDS["euclidean"], METRIC_IDS["sqeuclidean"]):
            bound = np.sum(spread ** 2)          # the squared distance is accumulated
        elif kind == METRIC_IDS["manhattan"]:
            bound = np.sum(spread)
        elif kind == METRIC_IDS["chebyshev"]:
            bound = np.max(spread)
        elif kind == METRIC_IDS["minkowski"]:
            bound = np.sum(spread ** p)          # sum of |x_k - y_k|^p before the root
        else:                                    # cosine: dot products and squared norms
            bound = np.sum(magnitude ** 2)
        bound *= scale
    if not bound <= FLOAT32_MAX:
        raise ValueError(
            f"the data's range is too large for float32 {metric} distances (they could reach "
            f"{bound:.3g}, above float32's {FLOAT32_MAX:.3g}); rescale the data"
        )


def pairwise_distances(X, Y=None, metric="euclidean", p=2.0):
    """Calculate the distance between each row of X and each row of Y.

    This function is the same as `sklearn.metrics.pairwise_distances`. The GPU
    calculates the distances in float32.

    Parameters
    ----------
    X : array_like of shape (n_samples_X, n_features)
        The first set of points. All values must be finite.
    Y : array_like of shape (n_samples_Y, n_features), optional
        The second set of points. If Y is None, the function uses X for Y.
    metric : str, default="euclidean"
        One of the names in `cuhc.METRIC_IDS`.
    p : float, default=2.0
        The Minkowski order. The function uses it only if `metric="minkowski"`.

    Returns
    -------
    numpy.ndarray of shape (n_samples_X, n_samples_Y), float32
        `D[i, j]` is the distance between `X[i]` and `Y[j]`.

    Raises
    ------
    ValueError
        If the metric is not known, the shapes are not correct, the input contains
        NaN or infinite values, or the distances are too large for float32.

    Notes
    -----
    The full matrix must fit in the GPU memory and in the host memory. Its size is
    4 · len(X) · len(Y) bytes.

    Examples
    --------
    >>> import numpy as np, cuhc
    >>> cuhc.pairwise_distances(np.array([[0.0, 0.0], [3.0, 4.0]]))
    array([[0., 5.],
           [5., 0.]], dtype=float32)
    """
    validate_metric(metric, p)
    X = as_points(X, "X")
    if Y is not None:
        Y = as_points(Y, "Y")
        if Y.shape[1] != X.shape[1]:
            raise ValueError(
                f"X and Y must have the same number of features, got {X.shape[1]} and {Y.shape[1]}"
            )
    check_distance_range(X, metric, p, Y)
    numPointsB = X.shape[0] if Y is None else Y.shape[0]

    distances = np.empty((X.shape[0], numPointsB), dtype=np.float32)
    status = _lib.lib.cuhc_pairwise_distances(
        _lib.float_ptr(X), X.shape[0],
        None if Y is None else _lib.float_ptr(Y), numPointsB,
        X.shape[1], METRIC_IDS[metric], float(p),
        _lib.float_ptr(distances),
    )
    _lib.check(status)
    return distances
