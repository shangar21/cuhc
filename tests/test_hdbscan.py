"""cuhc.HDBSCAN against sklearn.cluster.HDBSCAN.

Contract: same constructor parameters as sklearn (min_cluster_size,
min_samples, cluster_selection_epsilon, max_cluster_size, metric, alpha,
cluster_selection_method, allow_single_cluster) with the same meaning:
  - min_samples counts the point itself (min_samples=1 gives core distance 0)
    and defaults to min_cluster_size,
  - mutual reachability is max(core(a), core(b), d(a, b) / alpha), as in
    sklearn's default path for feature input.

Why labels are not simply compared with sklearn's
-------------------------------------------------
Mutual reachability creates many exactly equal edge weights (every edge from a
point to a neighbour inside its core radius weighs core(point)). When several
groups join at the same height, the binary merge tree can pair them in any
order, and HDBSCAN's condensing step looks at that binary tree, so the final
labels can depend on the order. sklearn's own labels do: reshuffling only the
order of tied edges in sklearn's own MST changes its labels in 16 of 20 tries
(docs/hdbscan.md, "Ties"). sklearn's particular order comes from sequential
Prim's algorithm plus an unstable sort, which a parallel implementation can't
reproduce. So the tests check, for every case:

  1. the tree is sklearn's tree up to tie order: identical cophenetic distances
     (the height at which each pair of points first joins), which don't depend
     on tie order;
  2. labels and probabilities are exactly what sklearn's own extraction code
     produces from our tree;
  3. end to end, labels equal sklearn's when there are no ties
     (min_samples=1), and agree almost perfectly otherwise.
"""

import numpy as np
import pytest
from scipy.cluster.hierarchy import cophenet
from sklearn.base import ClusterMixin, clone
from sklearn.cluster import HDBSCAN as SkHDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances as sk_pairwise_distances

import cuhc
from _helpers import assert_same_partition, blobs, requires

pytestmark = requires("HDBSCAN")

# sklearn's extraction code is private; used to check our port of it exactly.
try:
    from sklearn.cluster._hdbscan._linkage import HIERARCHY_dtype
    from sklearn.cluster._hdbscan._tree import tree_to_labels as sklearn_tree_to_labels
except ImportError:  # pragma: no cover - depends on the sklearn version
    sklearn_tree_to_labels = None

# Probabilities are ratios of merge distances, so they inherit the relative
# error of the distances. float32 cosine distance between nearby points
# (1 - dot/norm, a difference of nearly equal numbers) is only good to ~1e-4
# relative, which shows up here as up to ~4e-4 absolute.
PROBABILITY_ATOL = 1e-3
# Cophenetic distances: float32 vs float64. The absolute part covers tiny
# cosine distances, which have large *relative* error for the same reason.
COPHENETIC_RTOL, COPHENETIC_ATOL = 1e-5, 1e-6
# With ties, labels may differ from sklearn's for a handful of points.
MIN_ARI_WITH_TIES = 0.98

EXTRACTION_PARAMS = ["min_cluster_size", "cluster_selection_method", "allow_single_cluster",
                     "cluster_selection_epsilon", "max_cluster_size"]


def call_sklearn(function, *args, **kwargs):
    """Runs sklearn code, skipping the test where sklearn itself crashes.

    sklearn 1.9's cluster_selection_epsilon search (traverse_upwards in
    _tree.pyx) assigns a 1-element array to a C integer, which NumPy >= 2.5
    rejects. It crashes whenever the search has to move up the tree, on any
    input, so there is no reference to compare with in those cases.
    """
    try:
        return function(*args, **kwargs)
    except TypeError as error:
        if "only 0-dimensional arrays can be converted to Python scalars" in str(error):
            pytest.skip("sklearn's epsilon search crashes with this NumPy version")
        raise


def fit_sklearn(X, metric="euclidean", **params):
    """sklearn on the same (float32) values, in float64, using its default algorithm."""
    model = SkHDBSCAN(metric=metric, copy=True, **params)
    return call_sklearn(model.fit, X.astype(np.float64))


def as_linkage(single_linkage_tree):
    """sklearn's structured single-linkage tree -> scipy (n-1) x 4 format."""
    return np.column_stack([single_linkage_tree["left_node"], single_linkage_tree["right_node"],
                            single_linkage_tree["value"],
                            single_linkage_tree["cluster_size"]]).astype(np.float64)


def check_against_sklearn(ours, ref, ties_possible=True):
    # 1. Same tree up to tie order.
    np.testing.assert_allclose(cophenet(ours._single_linkage_tree_),
                               cophenet(as_linkage(ref._single_linkage_tree_)),
                               rtol=COPHENETIC_RTOL, atol=COPHENETIC_ATOL)

    # 2. Exactly sklearn's extraction, applied to our tree.
    if sklearn_tree_to_labels is not None:
        Z = ours._single_linkage_tree_
        tree = np.empty(len(Z), dtype=HIERARCHY_dtype)
        tree["left_node"], tree["right_node"] = Z[:, 0], Z[:, 1]
        tree["value"], tree["cluster_size"] = Z[:, 2], Z[:, 3]
        params = {k: getattr(ours, k) for k in EXTRACTION_PARAMS}
        labels, probabilities = call_sklearn(sklearn_tree_to_labels, tree, **params)
        np.testing.assert_array_equal(ours.labels_, labels)
        np.testing.assert_allclose(ours.probabilities_, probabilities, rtol=0, atol=1e-12)

    # 3. End to end.
    if ties_possible:
        assert adjusted_rand_score(ref.labels_, ours.labels_) >= MIN_ARI_WITH_TIES
    else:
        assert_same_partition(ours.labels_, ref.labels_)
        np.testing.assert_allclose(ours.probabilities_, ref.probabilities_, atol=PROBABILITY_ATOL)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev", "cosine"])
@pytest.mark.parametrize("min_cluster_size, min_samples", [(5, None), (15, None), (10, 3), (20, 1)])
def test_matches_sklearn(metric, min_cluster_size, min_samples):
    X = blobs(n_samples=600, n_features=4, centers=6, noise_fraction=0.1, seed=min_cluster_size)
    params = dict(min_cluster_size=min_cluster_size, min_samples=min_samples)
    ours = cuhc.HDBSCAN(metric=metric, **params).fit(X)
    ref = fit_sklearn(X, metric=metric, **params)
    # min_samples=1 makes every core distance 0, so mutual reachability is the
    # plain distance and continuous data has no ties.
    check_against_sklearn(ours, ref, ties_possible=min_samples != 1)


@pytest.mark.parametrize("params", [
    {"cluster_selection_method": "leaf"},
    {"cluster_selection_epsilon": 0.5},
    {"cluster_selection_method": "leaf", "cluster_selection_epsilon": 0.5},
    {"alpha": 1.5},
    {"allow_single_cluster": True, "min_cluster_size": 100},
    {"max_cluster_size": 60},
], ids=lambda p: ",".join(f"{k}={v}" for k, v in p.items()))
@pytest.mark.parametrize("min_samples", [None, 1])
def test_options_match_sklearn(params, min_samples):
    X = blobs(n_samples=500, n_features=3, centers=4, noise_fraction=0.15, seed=21)
    params = {"min_cluster_size": 10, "min_samples": min_samples, **params}
    ours = cuhc.HDBSCAN(**params).fit(X)
    ref = fit_sklearn(X, **params)
    check_against_sklearn(ours, ref, ties_possible=min_samples != 1)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
@pytest.mark.parametrize("min_samples", [1, 2, 7, 40])
def test_core_distances(metric, min_samples):
    # Distance to the min_samples-th nearest point, the point itself being the 1st.
    X = blobs(n_samples=300, n_features=5, seed=min_samples)
    ours = cuhc.HDBSCAN(min_cluster_size=5, min_samples=min_samples, metric=metric).fit(X)
    D = sk_pairwise_distances(X.astype(np.float64), metric=metric)
    np.fill_diagonal(D, 0.0)
    expected = np.sort(D, axis=1)[:, min_samples - 1]
    np.testing.assert_allclose(ours.core_distances_, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("min_samples", [1, 5, 64, 65, 200])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_core_distance_methods_agree(min_samples, metric):
    # Fused (k <= 64) and blocked radix select (any k) must give identical
    # values; sizes not a multiple of the 128-row (64 for cosine) tile.
    from cuhc._hdbscan import hdbscan_tree

    X = blobs(n_samples=1000, n_features=6, seed=min_samples)
    _, radix = hdbscan_tree(X, min_samples, metric=metric, core_method="radix")
    D = sk_pairwise_distances(X.astype(np.float64), metric=metric)
    np.fill_diagonal(D, 0.0)
    np.testing.assert_allclose(radix, np.sort(D, axis=1)[:, min_samples - 1], rtol=1e-5, atol=1e-6)
    if min_samples <= 64:
        _, fused = hdbscan_tree(X, min_samples, metric=metric, core_method="fused")
        np.testing.assert_array_equal(fused, radix)
    else:
        with pytest.raises(RuntimeError, match="algorithm"):
            hdbscan_tree(X, min_samples, metric=metric, core_method="fused")


@pytest.mark.parametrize("min_cluster_size", [2, 5, 25, 100])
def test_condense_tree_matches_reference(min_cluster_size):
    # The C++ condensing must reproduce the NumPy reference exactly, row order
    # included (later steps depend on it).
    from cuhc._hdbscan import condense_tree, condense_tree_reference

    model = cuhc.HDBSCAN(min_cluster_size=5).fit(blobs(n_samples=3000, noise_fraction=0.1, seed=4))
    Z = model._single_linkage_tree_
    for ours, ref in zip(condense_tree(Z, min_cluster_size),
                         condense_tree_reference(Z, min_cluster_size)):
        np.testing.assert_array_equal(ours, ref)


def test_core_distances_with_duplicate_points():
    # Duplicates are at distance 0; the radix select must count them correctly.
    X = np.repeat(np.random.default_rng(3).normal(size=(50, 3)), 4, axis=0).astype(np.float32)
    ours = cuhc.HDBSCAN(min_cluster_size=5, min_samples=4).fit(X)
    np.testing.assert_array_equal(ours.core_distances_, 0.0)
    ours = cuhc.HDBSCAN(min_cluster_size=5, min_samples=5).fit(X)
    assert np.all(ours.core_distances_ > 0)


def test_pure_noise_is_all_noise_like_sklearn():
    X = np.random.default_rng(5).uniform(size=(200, 2)).astype(np.float32)
    ours = cuhc.HDBSCAN(min_cluster_size=50).fit(X)
    ref = fit_sklearn(X, min_cluster_size=50)
    assert_same_partition(ours.labels_, ref.labels_)


def test_defaults_match_sklearn():
    ours, ref = cuhc.HDBSCAN(), SkHDBSCAN()
    for name in ["min_cluster_size", "min_samples", "cluster_selection_epsilon",
                 "max_cluster_size", "metric", "alpha", "cluster_selection_method",
                 "allow_single_cluster"]:
        assert getattr(ours, name) == getattr(ref, name), name


def test_fit_predict_returns_labels():
    X = blobs(n_samples=200, noise_fraction=0.1)
    model = cuhc.HDBSCAN(min_cluster_size=10)
    np.testing.assert_array_equal(model.fit_predict(X), model.labels_)


def test_is_a_sklearn_estimator():
    model = cuhc.HDBSCAN(min_cluster_size=7, metric="manhattan")
    assert isinstance(model, ClusterMixin)
    assert clone(model).get_params() == model.get_params()


@pytest.mark.parametrize("params, message", [
    ({"min_cluster_size": 1}, "min_cluster_size"),
    ({"min_samples": 0}, "min_samples"),
    ({"min_samples": 1000}, "at most the number of samples"),
    ({"alpha": 0.0}, "alpha"),
    ({"cluster_selection_epsilon": -1.0}, "epsilon"),
    ({"cluster_selection_method": "biggest"}, "cluster_selection_method"),
    ({"metric": "hamming"}, "unknown metric"),
])
def test_rejects_invalid_parameters(params, message):
    with pytest.raises(ValueError, match=message):
        cuhc.HDBSCAN(**params).fit(blobs(n_samples=100))


def small_condensed_tree():
    """Hand-built condensed tree over 10 points (root cluster 10):

        10 ──(split at distance 3.0)──┬── 11 ──(split at distance 0.4)──┬── 13: points 0-2
                                      │                                  └── 14: points 3-5
                                      └── 12: points 6-9
    Points fall out of their leaf at distance 0.2.
    """
    rows = [(10, 11, 1 / 3.0, 6), (10, 12, 1 / 3.0, 4), (11, 13, 1 / 0.4, 3), (11, 14, 1 / 0.4, 3)]
    rows += [(13, p, 5.0, 1) for p in range(0, 3)]
    rows += [(14, p, 5.0, 1) for p in range(3, 6)]
    rows += [(12, p, 5.0, 1) for p in range(6, 10)]
    parents, children, lambdas, sizes = map(np.array, zip(*rows))
    return parents, children, lambdas.astype(np.float64), sizes


# sklearn crashes in exactly this code path with NumPy >= 2.5 (see call_sklearn),
# so the expected answers are worked out by hand from sklearn's algorithm:
# a selected cluster that split off at a distance below epsilon is replaced by
# its nearest ancestor that split off at a distance above epsilon (or, when the
# search reaches the root, by the root only if allow_single_cluster).
@pytest.mark.parametrize("epsilon, allow_single_cluster, expected", [
    (0.0, False, {12, 13, 14}),  # leaves
    (0.3, False, {12, 13, 14}),  # 13 and 14 split at 0.4 > 0.3: kept
    (0.5, False, {11, 12}),      # 13, 14 split at 0.4 < 0.5: replaced by 11 (3.0 > 0.5)
    (5.0, False, {11, 12}),      # nothing above 5.0: stop just below the root
    (5.0, True, {10}),           # ... or take the root when allowed
])
def test_leaf_selection_with_epsilon(epsilon, allow_single_cluster, expected):
    from cuhc._hdbscan import compute_stability, label_points, select_clusters

    condensed = small_condensed_tree()
    clusters = select_clusters(condensed, compute_stability(condensed), "leaf",
                               allow_single_cluster, epsilon)
    assert clusters == expected
    labels = label_points(condensed, clusters, allow_single_cluster, epsilon)
    assert (labels >= 0).all()
    groups = {frozenset(np.flatnonzero(labels == label).tolist()) for label in set(labels)}
    leaf_points = {13: {0, 1, 2}, 14: {3, 4, 5}, 12: {6, 7, 8, 9},
                   11: {0, 1, 2, 3, 4, 5}, 10: set(range(10))}
    assert groups == {frozenset(leaf_points[c]) for c in expected}
