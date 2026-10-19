"""HDBSCAN: GPU tree building plus a NumPy port of sklearn's cluster extraction.

The GPU builds the single-linkage tree of the mutual reachability distance
(src/hdbscan.cuh). Everything here works on that tree only. It is a port of
sklearn/cluster/_hdbscan/_tree.pyx (sklearn 1.9), kept close to the original,
including quirks that affect results, so that labels and probabilities match
sklearn's exactly. The one slow step, condensing, runs in C++
(hc::condenseTree); condense_tree_reference is the readable NumPy version it
was ported from, and the tests check that both agree. See docs/hdbscan.md.
"""

import numbers

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from . import _lib
from ._distances import METRIC_IDS, as_points, check_distance_range, validate_metric

NOISE = -1


# ---------------------------------------------------------------------------
# GPU part


# Must match hc::CoreMethod in src/hdbscan.cuh.
CORE_METHOD_IDS = {"auto": 0, "fused": 1, "radix": 2}


def hdbscan_tree(X, min_samples, metric="euclidean", alpha=1.0, p=2.0, core_method="auto"):
    """(single-linkage tree of mutual reachability, core distances), both from the GPU.

    core_method: "fused" keeps each row's k nearest in shared memory (k <= 64),
    "radix" computes distance blocks and radix-selects; "auto" picks fused when
    it can. Both are exact.
    """
    X = as_points(X, "X")
    check_distance_range(X, metric, p, scale=1.0 / alpha)
    n = X.shape[0]
    tree = np.empty((max(n - 1, 0), 4), dtype=np.float64)
    core = np.empty(n, dtype=np.float32)
    status = _lib.lib.cuhc_hdbscan_tree(
        _lib.float_ptr(X), n, X.shape[1], METRIC_IDS[metric], float(p),
        int(min_samples), float(alpha), CORE_METHOD_IDS[core_method],
        _lib.double_ptr(tree), _lib.float_ptr(core),
    )
    _lib.check(status)
    return tree, core


# ---------------------------------------------------------------------------
# Condensed tree
#
# The condensed tree is an edge list (parent, child, lambda, child_size):
#   - parent is always a cluster; clusters are numbered from n (the root) up,
#   - child is a point (< n) that "falls out" of parent at lambda, or a child
#     cluster that splits off at lambda,
#   - lambda = 1 / distance, so larger lambda = denser.


def _bfs_from_hierarchy(Z, root, n):
    """Nodes of the scipy-format tree Z below root, in breadth-first order."""
    result = []
    queue = np.array([root], dtype=np.intp)
    while len(queue):
        result.extend(queue.tolist())
        internal = queue[queue >= n] - n
        queue = Z[internal, :2].astype(np.intp).ravel()
    return result


def condense_tree(Z, min_cluster_size):
    """Same as condense_tree_reference, in C++ (hc::condenseTree)."""
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    n = len(Z) + 1
    capacity = n + 2 * (n // max(int(min_cluster_size), 1)) + 2
    parents, children = np.empty(capacity, np.int64), np.empty(capacity, np.int64)
    lambdas, sizes = np.empty(capacity, np.float64), np.empty(capacity, np.int64)
    rows = _lib.lib.cuhc_condense_tree(
        _lib.double_ptr(Z), n, int(min_cluster_size), capacity, _lib.int64_ptr(parents),
        _lib.int64_ptr(children), _lib.double_ptr(lambdas), _lib.int64_ptr(sizes),
    )
    if rows < 0:  # can't happen with the capacity above, but don't trust it silently
        raise RuntimeError(f"condensed tree needs {-rows} rows, only {capacity} allocated")
    return (parents[:rows].astype(np.intp), children[:rows].astype(np.intp),
            lambdas[:rows], sizes[:rows].astype(np.intp))


def condense_tree_reference(Z, min_cluster_size):
    """Walks the single-linkage tree from the root. At each split:
      both sides >= min_cluster_size: a real split, two new clusters;
      one side smaller:   the big side keeps the parent's label, and the points of
                          the small side fall out of the parent at this lambda;
      both sides smaller: all points fall out; the parent cluster ends here.
    """
    n = len(Z) + 1
    root = 2 * (n - 1)
    node_list = _bfs_from_hierarchy(Z, root, n)
    relabel = np.empty(root + 1, dtype=np.intp)
    relabel[root] = n
    next_label = n + 1
    ignore = np.zeros(root + 1, dtype=bool)
    parents, children, lambdas, sizes = [], [], [], []

    def emit(parent, child, lambda_value, size):
        parents.append(parent)
        children.append(child)
        lambdas.append(lambda_value)
        sizes.append(size)

    def drop_subtree(parent_label, subtree_root, lambda_value):
        for sub_node in _bfs_from_hierarchy(Z, subtree_root, n):
            if sub_node < n:
                emit(parent_label, sub_node, lambda_value, 1)
            ignore[sub_node] = True

    for node in node_list:
        if ignore[node] or node < n:
            continue
        left, right, distance, _ = Z[node - n]
        left, right = int(left), int(right)
        lambda_value = 1.0 / distance if distance > 0.0 else np.inf
        left_count = int(Z[left - n, 3]) if left >= n else 1
        right_count = int(Z[right - n, 3]) if right >= n else 1

        if left_count >= min_cluster_size and right_count >= min_cluster_size:
            relabel[left] = next_label
            next_label += 1
            emit(relabel[node], relabel[left], lambda_value, left_count)
            relabel[right] = next_label
            next_label += 1
            emit(relabel[node], relabel[right], lambda_value, right_count)
        elif left_count < min_cluster_size and right_count < min_cluster_size:
            drop_subtree(relabel[node], left, lambda_value)
            drop_subtree(relabel[node], right, lambda_value)
        elif left_count < min_cluster_size:
            relabel[right] = relabel[node]
            drop_subtree(relabel[node], left, lambda_value)
        else:
            relabel[left] = relabel[node]
            drop_subtree(relabel[node], right, lambda_value)

    return (np.array(parents, dtype=np.intp), np.array(children, dtype=np.intp),
            np.array(lambdas, dtype=np.float64), np.array(sizes, dtype=np.intp))


def compute_stability(condensed):
    """stability(C) = sum over C's rows of (lambda - lambda_birth(C)) * size."""
    parents, children, lambdas, sizes = condensed
    smallest_cluster = parents.min()
    largest = max(children.max(), smallest_cluster)
    births = np.full(largest + 1, np.nan)
    births[children] = lambdas
    births[smallest_cluster] = 0.0
    result = np.zeros(parents.max() - smallest_cluster + 1)
    np.add.at(result, parents - smallest_cluster, (lambdas - births[parents]) * sizes)
    return {int(smallest_cluster + i): float(s) for i, s in enumerate(result)}


def _max_lambdas(condensed):
    """deaths[C] = largest lambda among C's rows. Like sklearn, it looks at runs
    of consecutive rows with the same parent, and a later run overwrites an
    earlier one for the same parent."""
    parents, _, lambdas, _ = condensed
    deaths = np.zeros(parents.max() + 1)
    run_starts = np.concatenate([[0], np.flatnonzero(np.diff(parents)) + 1])
    for start, run_max in zip(run_starts, np.maximum.reduceat(lambdas, run_starts)):
        deaths[parents[start]] = run_max
    return deaths


def _bfs_from_cluster_tree(cluster_parents, cluster_children, root):
    result, queue = [], np.array([root], dtype=np.intp)
    while len(queue):
        result.extend(queue.tolist())
        queue = cluster_children[np.isin(cluster_parents, queue)]
    return result


def _traverse_upwards(cluster_parents, cluster_children, cluster_lambdas, epsilon, leaf,
                      allow_single_cluster):
    root = cluster_parents.min()
    parent = cluster_parents[cluster_children == leaf][0]
    if parent == root:
        return parent if allow_single_cluster else leaf
    parent_eps = 1.0 / cluster_lambdas[cluster_children == parent][0]
    if parent_eps > epsilon:
        return parent
    return _traverse_upwards(cluster_parents, cluster_children, cluster_lambdas, epsilon, parent,
                             allow_single_cluster)


def _epsilon_search(leaves, cluster_parents, cluster_children, cluster_lambdas, epsilon,
                    allow_single_cluster):
    """Replaces clusters that split at distances below epsilon by the ancestor
    that's at least epsilon apart from its sibling."""
    selected, processed = [], []
    for leaf in leaves:
        eps = 1.0 / cluster_lambdas[cluster_children == leaf][0]
        if eps < epsilon:
            if leaf not in processed:
                epsilon_child = _traverse_upwards(cluster_parents, cluster_children,
                                                  cluster_lambdas, epsilon, leaf,
                                                  allow_single_cluster)
                selected.append(epsilon_child)
                for sub_node in _bfs_from_cluster_tree(cluster_parents, cluster_children,
                                                       epsilon_child):
                    if sub_node != epsilon_child:
                        processed.append(sub_node)
        else:
            selected.append(leaf)
    return set(int(c) for c in selected)


def select_clusters(condensed, stability, cluster_selection_method="eom",
                    allow_single_cluster=False, cluster_selection_epsilon=0.0,
                    max_cluster_size=None):
    """The set of selected cluster ids (sklearn's _get_clusters, minus labelling)."""
    parents, children, lambdas, sizes = condensed
    stability = dict(stability)
    # Children have larger ids than their parents, so reverse id order is a
    # bottom-up order.
    node_list = sorted(stability, reverse=True)
    if not allow_single_cluster:
        node_list = node_list[:-1]  # exclude the root

    is_cluster_row = sizes > 1
    cluster_parents, cluster_children = parents[is_cluster_row], children[is_cluster_row]
    cluster_lambdas, cluster_sizes_arr = lambdas[is_cluster_row], sizes[is_cluster_row]
    is_cluster = {cluster: True for cluster in node_list}
    n_samples = children[sizes == 1].max() + 1
    if max_cluster_size is None:
        max_cluster_size = n_samples + 1  # never triggers
    cluster_sizes = dict(zip(cluster_children.tolist(), cluster_sizes_arr.tolist()))
    if allow_single_cluster:
        cluster_sizes[node_list[-1]] = int(cluster_sizes_arr[cluster_parents == node_list[-1]].sum())

    if cluster_selection_method == "eom":
        # Excess of mass: keep a cluster if it's more stable than all of its
        # descendants together; otherwise pass the descendants' stability up.
        for node in node_list:
            subtree_stability = float(np.sum(
                [stability[child] for child in cluster_children[cluster_parents == node]]))
            if subtree_stability > stability[node] or cluster_sizes[node] > max_cluster_size:
                is_cluster[node] = False
                stability[node] = subtree_stability
            else:
                for sub_node in _bfs_from_cluster_tree(cluster_parents, cluster_children, node):
                    if sub_node != node:
                        is_cluster[sub_node] = False

        if cluster_selection_epsilon != 0.0 and len(cluster_children) > 0:
            eom_clusters = [c for c in is_cluster if is_cluster[c]]
            selected = []
            if len(eom_clusters) == 1 and eom_clusters[0] == cluster_parents.min():
                if allow_single_cluster:
                    selected = eom_clusters
            else:
                selected = _epsilon_search(set(eom_clusters), cluster_parents, cluster_children,
                                           cluster_lambdas, cluster_selection_epsilon,
                                           allow_single_cluster)
            for c in is_cluster:
                is_cluster[c] = c in selected

    elif cluster_selection_method == "leaf":
        if len(cluster_children) == 0:
            leaves = set()
        else:
            leaves = set(cluster_children.tolist()) - set(cluster_parents.tolist())
        if len(leaves) == 0:
            for c in is_cluster:
                is_cluster[c] = False
            is_cluster[int(parents.min())] = True
        if cluster_selection_epsilon != 0.0:
            selected = _epsilon_search(leaves, cluster_parents, cluster_children,
                                       cluster_lambdas, cluster_selection_epsilon,
                                       allow_single_cluster)
        else:
            selected = leaves
        for c in is_cluster:
            is_cluster[c] = c in selected

    return {c for c in is_cluster if is_cluster[c]}


def label_points(condensed, clusters, allow_single_cluster, cluster_selection_epsilon):
    """Each point's label: its nearest selected ancestor cluster, else noise.

    sklearn does this with a union-find over the condensed rows; propagating
    "owner" top-down over the rows (parents always come before their children)
    gives the same result.
    """
    parents, children, lambdas, sizes = condensed
    root = int(parents.min())
    n = root
    cluster_label = {c: i for i, c in enumerate(sorted(clusters))}

    owner = {root: root}
    for parent, child in zip(parents[sizes > 1].tolist(), children[sizes > 1].tolist()):
        owner[child] = child if child in clusters else owner[parent]

    labels = np.full(n, NOISE, dtype=np.intp)
    point_rows = np.flatnonzero(children < n)
    point_owner = np.array([owner[p] for p in parents[point_rows].tolist()], dtype=np.intp)
    in_cluster = point_owner != root
    labels[children[point_rows[in_cluster]]] = [cluster_label[c] for c in point_owner[in_cluster]]

    if len(clusters) == 1 and allow_single_cluster and root in clusters:
        # The root itself is the only cluster: points join it if they stay in it
        # dense enough (lambda at which they fall out >= threshold).
        if cluster_selection_epsilon != 0.0:
            threshold = 1.0 / cluster_selection_epsilon
        else:
            threshold = lambdas[parents == root].max()
        at_root = point_rows[~in_cluster]
        keep = lambdas[at_root] >= threshold
        labels[children[at_root[keep]]] = cluster_label[root]
    return labels


def membership_probabilities(condensed, clusters, labels):
    """How strongly each point belongs to its cluster: the lambda at which it
    falls out, relative to the largest lambda in its cluster (capped at 1)."""
    parents, children, lambdas, _ = condensed
    deaths = _max_lambdas(condensed)
    cluster_of_label = np.array(sorted(clusters), dtype=np.intp)
    result = np.zeros(len(labels))

    point_rows = np.flatnonzero(children < parents.min())
    points = children[point_rows]
    labelled = labels[points] != NOISE
    point_rows, points = point_rows[labelled], points[labelled]
    max_lambda = deaths[cluster_of_label[labels[points]]]
    point_lambda = lambdas[point_rows]
    certain = (max_lambda == 0.0) | np.isinf(point_lambda)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.minimum(point_lambda, max_lambda) / max_lambda
    result[points] = np.where(certain, 1.0, ratio)
    return result


def tree_to_labels(Z, min_cluster_size, cluster_selection_method="eom",
                   allow_single_cluster=False, cluster_selection_epsilon=0.0,
                   max_cluster_size=None):
    condensed = condense_tree(Z, min_cluster_size)
    clusters = select_clusters(condensed, compute_stability(condensed), cluster_selection_method,
                               allow_single_cluster, cluster_selection_epsilon, max_cluster_size)
    labels = label_points(condensed, clusters, allow_single_cluster, cluster_selection_epsilon)
    return labels, membership_probabilities(condensed, clusters, labels)


# ---------------------------------------------------------------------------
# Estimator


class HDBSCAN(ClusterMixin, BaseEstimator):
    """Density-based clustering (HDBSCAN) on the GPU.

    This class is the same as `sklearn.cluster.HDBSCAN`, with the same
    parameters, defaults and results for the options below. It does not keep the
    n × n distance matrix in memory.

    HDBSCAN finds clusters as dense areas. It gives the label −1 (noise) to points
    in sparse areas. You do not set the number of clusters. You set the smallest
    group of points that is a cluster (`min_cluster_size`).

    Parameters
    ----------
    min_cluster_size : int, default=5
        The smallest group of points that is a cluster. The value must be 2 or
        more. This is the most important parameter.
    min_samples : int or None, default=None
        The number of points for the core distance. This number includes the point
        itself. A larger value gives more noise points. The default is
        `min_cluster_size`. With a value of 64 or less, cuhc uses its fastest
        kernel.
    cluster_selection_epsilon : float, default=0.0
        Merge clusters that are nearer than this distance. This prevents many small
        clusters in dense areas.
    max_cluster_size : int or None, default=None
        Do not select a cluster that has more points than this value (with
        `"eom"` only).
    metric : str, default="euclidean"
        One of the names in `cuhc.METRIC_IDS`. Minkowski always uses p = 2 in
        this class.
    alpha : float, default=1.0
        A distance scale. The mutual reachability distance is
        `max(core(a), core(b), d(a, b) / alpha)`.
    cluster_selection_method : {"eom", "leaf"}, default="eom"
        `"eom"` (excess of mass) selects the most stable clusters. `"leaf"` selects
        the smallest clusters at the bottom of the tree, which gives more clusters
        and smaller clusters.
    allow_single_cluster : bool, default=False
        If True, the result can be one cluster.

    Attributes
    ----------
    labels_ : numpy.ndarray of shape (n_samples,)
        The cluster of each point, from 0 to k − 1. The label −1 shows noise.
    probabilities_ : numpy.ndarray of shape (n_samples,)
        How strongly each point is in its cluster, from 0 to 1. The value is 0
        for noise.
    core_distances_ : numpy.ndarray of shape (n_samples,)
        The core distance of each point. This attribute is not in scikit-learn.
    n_features_in_ : int
        The number of features in the data that `fit` received.

    Notes
    -----
    These scikit-learn parameters are not available: `leaf_size`, `n_jobs`,
    `store_centers`, `copy`, `metric_params` and `metric="precomputed"`.

    Examples
    --------
    >>> import numpy as np, cuhc
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(0, 0.1, (50, 2)), rng.normal(5, 0.1, (50, 2))])
    >>> labels = cuhc.HDBSCAN(min_cluster_size=10).fit_predict(X)
    >>> sorted(set(labels.tolist()))
    [0, 1]
    """

    def __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0,
                 max_cluster_size=None, metric="euclidean", alpha=1.0,
                 cluster_selection_method="eom", allow_single_cluster=False):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.metric = metric
        self.alpha = alpha
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster

    def _validate_params(self):
        def is_int(value, minimum):
            return isinstance(value, numbers.Integral) and value >= minimum

        if not is_int(self.min_cluster_size, 2):
            raise ValueError(f"min_cluster_size must be an integer >= 2, got {self.min_cluster_size!r}")
        if self.min_samples is not None and not is_int(self.min_samples, 1):
            raise ValueError(f"min_samples must be None or an integer >= 1, got {self.min_samples!r}")
        if self.max_cluster_size is not None and not is_int(self.max_cluster_size, 1):
            raise ValueError(
                f"max_cluster_size must be None or an integer >= 1, got {self.max_cluster_size!r}")
        if not self.cluster_selection_epsilon >= 0:
            raise ValueError("cluster_selection_epsilon must be >= 0, got "
                             f"{self.cluster_selection_epsilon!r}")
        if not self.alpha > 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha!r}")
        if self.cluster_selection_method not in ("eom", "leaf"):
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf', got "
                             f"{self.cluster_selection_method!r}")
        validate_metric(self.metric, 2.0)

    def fit(self, X, y=None):
        self._validate_params()
        X = as_points(X, "X")
        n = X.shape[0]
        min_samples = self.min_cluster_size if self.min_samples is None else self.min_samples
        if min_samples > n:
            raise ValueError(f"min_samples ({min_samples}) must be at most the number of samples "
                             f"in X ({n})")

        Z, core = hdbscan_tree(X, min_samples, metric=self.metric, alpha=self.alpha)
        self.labels_, self.probabilities_ = tree_to_labels(
            Z, self.min_cluster_size, self.cluster_selection_method, self.allow_single_cluster,
            self.cluster_selection_epsilon, self.max_cluster_size,
        )
        self._single_linkage_tree_ = Z
        self.core_distances_ = core
        self.n_features_in_ = X.shape[1]
        return self
