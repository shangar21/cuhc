"""scikit-learn compatible estimators."""

import numbers

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from ._distances import as_points
from ._linkage import LINKAGE_IDS, cut_tree, linkage, resolve_algorithm, validate_linkage


class AgglomerativeClustering(ClusterMixin, BaseEstimator):
    """Agglomerative clustering on the GPU.

    This class is the same as `sklearn.cluster.AgglomerativeClustering`, with the
    same parameters, defaults and attributes. It is a scikit-learn estimator: it
    has `fit` and `fit_predict`, and it operates with `clone`, `Pipeline` and
    `GridSearchCV`. The result is exact. Single and Ward linkage do not keep the
    n × n distance matrix in memory. Complete and average linkage keep a float32
    matrix on the GPU.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters. If you set `distance_threshold`, set
        `n_clusters=None`.
    metric : str, default="euclidean"
        One of the names in `cuhc.METRIC_IDS`. Minkowski always uses p = 2 in
        this class. Ward linkage is possible only with `"euclidean"`.
    linkage : {"ward", "complete", "average", "single"}, default="ward"
        The rule that gives the distance between two clusters.
    distance_threshold : float or None, default=None
        Use this parameter instead of `n_clusters`. cuhc does not apply merges at
        this height or higher. Set `n_clusters=None` with it.
    compute_distances : bool, default=False
        If True, keep the merge heights in `distances_`. cuhc always keeps them if
        you set `distance_threshold`.
    algorithm : {"auto", "dense", "mst", "rnn", "nn_chain"}, default="auto"
        This parameter is not in scikit-learn. It is the same as in
        `cuhc.linkage`.

    Attributes
    ----------
    labels_ : numpy.ndarray of shape (n_samples,)
        The cluster of each point, from 0 to `n_clusters_` − 1.
    n_clusters_ : int
        The number of clusters.
    children_ : numpy.ndarray of shape (n_samples - 1, 2)
        The two clusters of each merge. The cluster numbers are the same as in the
        `cuhc.linkage` result.
    distances_ : numpy.ndarray of shape (n_samples - 1,)
        The height of each merge. Set only if `compute_distances` is True or
        `distance_threshold` is set.
    n_leaves_ : int
        The number of points.
    n_features_in_ : int
        The number of features in the data that `fit` received.
    n_connected_components_ : int
        Always 1.

    Notes
    -----
    These scikit-learn parameters are not available: `connectivity`, `memory` and
    `compute_full_tree`. cuhc always builds the full tree.

    Examples
    --------
    >>> import numpy as np, cuhc
    >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    >>> cuhc.AgglomerativeClustering(n_clusters=2).fit_predict(X)
    array([1, 1, 0, 0])
    """

    def __init__(self, n_clusters=2, *, metric="euclidean", linkage="ward",
                 distance_threshold=None, compute_distances=False, algorithm="auto"):
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances
        self.algorithm = algorithm

    def _validate_params(self):
        if (self.n_clusters is None) == (self.distance_threshold is None):
            raise ValueError(
                "Exactly one of n_clusters and distance_threshold has to be set, "
                "and the other needs to be None."
            )
        if self.n_clusters is not None and (
            not isinstance(self.n_clusters, numbers.Integral) or self.n_clusters < 1
        ):
            raise ValueError(f"n_clusters must be a positive integer, got {self.n_clusters!r}")
        if self.linkage not in LINKAGE_IDS:
            raise ValueError(
                f"linkage must be one of {list(LINKAGE_IDS)}, got {self.linkage!r}"
            )
        validate_linkage(self.linkage, self.metric)
        resolve_algorithm(self.algorithm, self.linkage)

    def fit(self, X, y=None):
        self._validate_params()
        X = as_points(X, "X")
        n = X.shape[0]
        if self.n_clusters is not None and self.n_clusters > n:
            raise ValueError(f"Cannot extract more clusters ({self.n_clusters}) than samples ({n})")

        Z = linkage(X, method=self.linkage, metric=self.metric, algorithm=self.algorithm)
        if self.distance_threshold is not None:
            # Same rule as sklearn: merges at or above the threshold split clusters.
            self.n_clusters_ = int(np.count_nonzero(Z[:, 2] >= self.distance_threshold)) + 1
        else:
            self.n_clusters_ = int(self.n_clusters)

        self.labels_ = cut_tree(Z, n - self.n_clusters_)
        self.children_ = Z[:, :2].astype(np.intp)
        if self.compute_distances or self.distance_threshold is not None:
            self.distances_ = Z[:, 2].copy()
        self.n_leaves_ = n
        self.n_features_in_ = X.shape[1]
        self.n_connected_components_ = 1
        return self
