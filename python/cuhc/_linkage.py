"""Merge trees (scipy-style linkage) and cutting them into flat clusters."""

import numpy as np

from . import _lib
from ._distances import METRIC_IDS, as_points, check_distance_range, validate_metric

# Must match hc::Linkage in src/linkage.cuh.
LINKAGE_IDS = {"single": 0, "complete": 1, "average": 2, "ward": 3}

# Linkages whose cluster distance only makes sense for Euclidean distance.
EUCLIDEAN_ONLY = {"ward"}

# Must match AlgorithmId in src/capi.cu.
ALGORITHM_IDS = {"dense": 0, "mst": 1, "nn_chain": 2, "rnn": 3}
# Linkages each algorithm can build.
ALGORITHM_LINKAGES = {
    "dense": set(LINKAGE_IDS), "mst": {"single"}, "nn_chain": {"ward"}, "rnn": {"ward"},
}


def validate_linkage(method, metric):
    if method not in LINKAGE_IDS:
        raise ValueError(f"unknown linkage method {method!r}; expected one of {list(LINKAGE_IDS)}")
    if method in EUCLIDEAN_ONLY and METRIC_IDS.get(metric) != METRIC_IDS["euclidean"]:
        raise ValueError(f"{method} linkage requires metric='euclidean', got {metric!r}")


def resolve_algorithm(algorithm, method, num_points=None):
    """The algorithm to run. "auto" picks the fastest exact one:
      single:              mst (O(n) memory)
      ward:                rnn (O(n * features) memory; faster than dense at every size)
      complete, average:   dense (no matrix-free exact algorithm)
    """
    if algorithm == "auto":
        return {"single": "mst", "ward": "rnn"}.get(method, "dense")
    if algorithm not in ALGORITHM_IDS:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; expected 'auto' or one of {list(ALGORITHM_IDS)}"
        )
    if method not in ALGORITHM_LINKAGES[algorithm]:
        raise ValueError(
            f"algorithm={algorithm!r} supports linkage {sorted(ALGORITHM_LINKAGES[algorithm])}, "
            f"not {method!r}"
        )
    return algorithm


def linkage(X, method="single", metric="euclidean", p=2.0, *, algorithm="auto"):
    """Build the full merge tree of the rows of X.

    This function is the same as `scipy.cluster.hierarchy.linkage`. The default
    method, `"single"`, is also the scipy default. The result is always exact.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        The points. cuhc changes all data types to float32. All values must be
        finite.
    method : {"single", "complete", "average", "ward"}, default="single"
        The linkage: the rule that gives the distance between two clusters.
        `"ward"` is possible only with `metric="euclidean"`.
    metric : str, default="euclidean"
        One of the names in `cuhc.METRIC_IDS`.
    p : float, default=2.0
        The Minkowski order. The function uses it only if `metric="minkowski"`.
    algorithm : {"auto", "dense", "mst", "rnn", "nn_chain"}, default="auto"
        The algorithm that builds the tree. All algorithms give the same exact
        tree. They have a different speed and use a different quantity of memory:

        - `"dense"`: the n × n float32 distance matrix on the GPU (4 · n² bytes).
          All linkages.
        - `"mst"`: a Borůvka minimum spanning tree, O(n) memory. Single linkage
          only.
        - `"rnn"`: rounds of reciprocal nearest neighbours, O(n · n_features)
          memory. Ward linkage only.
        - `"nn_chain"`: the nearest-neighbour chain, O(n · n_features) memory.
          Ward linkage only. It is slower than `"rnn"`.

        `"auto"` uses `"mst"` for single linkage, `"rnn"` for Ward linkage and
        `"dense"` for complete and average linkage.

    Returns
    -------
    Z : numpy.ndarray of shape (n_samples - 1, 4), float64
        The merge tree in the scipy format. Row i is one merge:
        `[cluster_a, cluster_b, height, size_of_new_cluster]`. The clusters 0 to
        n − 1 are the points. Row i makes the cluster n + i. The rows are in the
        sequence of their height.

    Raises
    ------
    ValueError
        If the method, the metric or the algorithm is not known, if the algorithm
        cannot build the linkage, if `method="ward"` with a metric that is not
        `"euclidean"`, if the input contains NaN or infinite values, or if the
        distances are too large for float32.
    MemoryError
        If the algorithm is `"dense"` and the n × n matrix is too large for the GPU
        memory.
    RuntimeError
        If CUDA has an error.

    Examples
    --------
    >>> import numpy as np, cuhc
    >>> X = np.array([[0.0], [1.0], [5.0], [7.0]])
    >>> cuhc.linkage(X, method="single")
    array([[0., 1., 1., 2.],
           [2., 3., 2., 2.],
           [4., 5., 4., 4.]])
    """
    validate_linkage(method, metric)
    resolve_algorithm(algorithm, method)  # validate before converting X
    validate_metric(metric, p)
    X = as_points(X, "X")
    n = X.shape[0]
    # Ward distances grow with cluster size: 2 * n_a * n_b / (n_a + n_b) <= n.
    check_distance_range(X, metric, p, scale=n if method == "ward" else 1.0)
    algorithm = resolve_algorithm(algorithm, method, n)

    tree = np.empty((max(n - 1, 0), 4), dtype=np.float64)
    if n == 1:
        return tree
    status = _lib.lib.cuhc_linkage(
        _lib.float_ptr(X), n, X.shape[1],
        METRIC_IDS[metric], float(p), LINKAGE_IDS[method], ALGORITHM_IDS[algorithm],
        _lib.double_ptr(tree),
    )
    if status == _lib.CUDA_ERROR_OUT_OF_MEMORY and algorithm == "dense":
        raise MemoryError(
            f"cuhc.linkage: the dense distance matrix for {n} points needs "
            f"{4 * n * n / 1e9:.1f} GB of GPU memory, which isn't available"
            + (" (ward linkage can use algorithm='rnn', which doesn't need it)"
               if method == "ward" else "")
        )
    _lib.check(status)
    return tree


def _ward_rnn_with_options(X, min_progress, dense_below=0, allow_dense=True):
    """Test hook: Ward via "rnn" with explicit hand-over settings (see hc::RnnOptions)."""
    X = as_points(X, "X")
    n = X.shape[0]
    tree = np.empty((max(n - 1, 0), 4), dtype=np.float64)
    _lib.check(_lib.lib.cuhc_ward_rnn_with_options(
        _lib.float_ptr(X), n, X.shape[1], float(min_progress), int(dense_below),
        int(bool(allow_dense)), _lib.double_ptr(tree)))
    return tree


def cut_tree(Z, num_merges):
    """Labels 0..k-1 for each leaf after applying the first num_merges merges of Z."""
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    # parent[c] = the cluster that c was merged into (or c itself if it's a root).
    parent = np.arange(2 * n - 1)
    merged = Z[:num_merges, :2].astype(np.intp)
    parent[merged[:, 0]] = n + np.arange(num_merges)
    parent[merged[:, 1]] = n + np.arange(num_merges)
    # Pointer jumping: after each pass parent[] skips twice as far up the tree.
    while True:
        grandparent = parent[parent]
        if np.array_equal(grandparent, parent):
            break
        parent = grandparent
    _, labels = np.unique(parent[:n], return_inverse=True)
    return labels


def fcluster(Z, t, criterion="maxclust"):
    """Cut a merge tree into flat clusters.

    This function is the same as `scipy.cluster.hierarchy.fcluster` for the two
    criteria below.

    Parameters
    ----------
    Z : array_like of shape (n_samples - 1, 4)
        A merge tree from `cuhc.linkage` or from scipy.
    t : int or float
        The number of clusters (`"maxclust"`) or the height (`"distance"`).
    criterion : {"maxclust", "distance"}, default="maxclust"
        - `"maxclust"`: apply the merges in sequence until t clusters are left.
          The result has t clusters or fewer.
        - `"distance"`: apply all the merges with a height of t or less.

    Returns
    -------
    numpy.ndarray of shape (n_samples,)
        One label for each point. The labels start at 1, as in scipy.

    Raises
    ------
    ValueError
        If the criterion is not known, or if `criterion="maxclust"` and t is less
        than 1.

    Examples
    --------
    >>> import numpy as np, cuhc
    >>> Z = cuhc.linkage(np.array([[0.0], [1.0], [5.0], [7.0]]))
    >>> cuhc.fcluster(Z, 2, criterion="maxclust")
    array([1, 1, 2, 2])
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    if criterion == "maxclust":
        if t < 1:
            raise ValueError(f"maxclust needs t >= 1, got {t}")
        num_merges = n - min(int(t), n)
    elif criterion == "distance":
        num_merges = int(np.count_nonzero(Z[:, 2] <= t))
    else:
        raise ValueError(f"unknown criterion {criterion!r}; expected 'maxclust' or 'distance'")
    return cut_tree(Z, num_merges) + 1
