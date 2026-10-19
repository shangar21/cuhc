"""cuhc.linkage (the merge tree) against scikit-learn's full tree.

Contract: cuhc.linkage(X, method, metric) returns the merge tree in scipy's
format, an (n-1, 4) array whose row i is
    [cluster_a, cluster_b, merge_distance, size_of_new_cluster]
where leaves are 0..n-1 and the cluster created by row i is n + i.

Reference: sklearn.cluster.AgglomerativeClustering with the full tree and
compute_distances, which exposes children_ (columns 0-1) and distances_
(column 2). For non-Ward linkages it's given a float64 precomputed distance
matrix, so every cuhc metric can be checked, not only those sklearn supports
natively.
"""

import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering as SkAgglomerativeClustering
from sklearn.metrics import pairwise_distances as sk_pairwise_distances

import cuhc
from _helpers import blobs, requires

pytestmark = requires("linkage")

LINKAGES = ["single", "complete", "average", "ward"]
METRICS = ["euclidean", "manhattan", "chebyshev", "cosine"]
RTOL, ATOL = 1e-4, 1e-5

# (method, metric) pairs that are defined. Ward only makes sense for Euclidean.
CASES = [(m, metric) for m in LINKAGES for metric in METRICS
         if m != "ward" or metric == "euclidean"]


def sklearn_full_tree(X, method, metric):
    X64 = X.astype(np.float64)
    if method == "ward":
        model = SkAgglomerativeClustering(n_clusters=None, distance_threshold=0.0,
                                          linkage="ward", metric="euclidean")
        model.fit(X64)
    else:
        model = SkAgglomerativeClustering(n_clusters=None, distance_threshold=0.0,
                                          linkage=method, metric="precomputed")
        model.fit(sk_pairwise_distances(X64, metric=metric))
    return model.children_, model.distances_


def subtree_sizes(children, n):
    sizes = np.ones(2 * n - 1, dtype=np.int64)
    for i, (a, b) in enumerate(children):
        sizes[n + i] = sizes[a] + sizes[b]
    return sizes[n:]


@pytest.mark.parametrize("method, metric", CASES, ids=[f"{m}-{d}" for m, d in CASES])
@pytest.mark.parametrize("n_samples", [2, 3, 130, 400])
def test_tree_matches_sklearn(method, metric, n_samples):
    X = blobs(n_samples=n_samples, seed=n_samples)
    Z = np.asarray(cuhc.linkage(X, method=method, metric=metric))
    ref_children, ref_distances = sklearn_full_tree(X, method, metric)

    assert Z.shape == (n_samples - 1, 4)
    # Same merges in the same order (the pair order within a row doesn't matter).
    np.testing.assert_array_equal(np.sort(Z[:, :2].astype(np.int64), axis=1),
                                  np.sort(ref_children, axis=1))
    np.testing.assert_allclose(Z[:, 2], ref_distances, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(Z[:, 3].astype(np.int64),
                                  subtree_sizes(ref_children, n_samples))


@pytest.mark.parametrize("method", LINKAGES)
def test_merge_distances_never_decrease(method):
    # All four linkages are "reducible": merge heights are monotone.
    Z = np.asarray(cuhc.linkage(blobs(seed=7), method=method))
    assert np.all(np.diff(Z[:, 2]) >= -ATOL)


def test_defaults_match_scipy_linkage():
    # linkage mirrors scipy.cluster.hierarchy.linkage, whose default is single/euclidean.
    X = blobs(n_samples=100, seed=8)
    np.testing.assert_array_equal(np.asarray(cuhc.linkage(X)),
                                  np.asarray(cuhc.linkage(X, method="single", metric="euclidean")))


@pytest.mark.parametrize("metric", ["manhattan", "cosine"])
def test_ward_rejects_non_euclidean(metric):
    with pytest.raises(ValueError, match="euclidean"):
        cuhc.linkage(blobs(n_samples=20), method="ward", metric=metric)


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        cuhc.linkage(blobs(n_samples=20), method="median-of-means")


def test_single_point_has_empty_tree():
    Z = np.asarray(cuhc.linkage(np.zeros((1, 3), dtype=np.float32), method="single"))
    assert Z.shape == (0, 4)


@pytest.mark.parametrize("method", LINKAGES)
def test_larger_tree_matches_sklearn(method):
    # Thousands of merges: exercises the nearest-neighbour rescans and lets
    # float32 rounding accumulate.
    X = blobs(n_samples=2500, n_features=16, centers=12, seed=31)
    Z = np.asarray(cuhc.linkage(X, method=method))
    ref_children, ref_distances = sklearn_full_tree(X, method, "euclidean")
    np.testing.assert_array_equal(np.sort(Z[:, :2].astype(np.int64), axis=1),
                                  np.sort(ref_children, axis=1))
    np.testing.assert_allclose(Z[:, 2], ref_distances, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("method", LINKAGES)
def test_duplicate_points_give_a_valid_tree(method):
    # Ties make the merge order ambiguous, so there's nothing exact to compare
    # with; the tree must still be well-formed.
    rng = np.random.default_rng(9)
    X = np.repeat(rng.normal(size=(40, 3)), 5, axis=0).astype(np.float32)
    n = len(X)
    Z = np.asarray(cuhc.linkage(X, method=method))

    assert Z.shape == (n - 1, 4)
    ids = Z[:, :2].astype(np.int64)
    assert np.all(ids[:, 0] < ids[:, 1])
    # Every cluster except the root is merged exactly once, and only after it exists.
    np.testing.assert_array_equal(np.sort(ids.ravel()), np.arange(2 * n - 2))
    assert np.all(ids < n + np.arange(n - 1)[:, None])
    assert np.all(np.diff(Z[:, 2]) >= -ATOL)
    np.testing.assert_array_equal(Z[:, 3], subtree_sizes(ids, n))
    # 40 distinct points, each repeated 5 times: the first 160 merges are at distance 0.
    np.testing.assert_allclose(Z[:160, 2], 0.0, atol=ATOL)
    assert Z[160, 2] > ATOL


# Distances are float32: input whose distances would overflow is rejected
# instead of producing a meaningless tree.

def huge_spread(n=200):
    return (1.01 ** np.arange(n, dtype=np.float64) * 1e20)[:, None].astype(np.float32)


@pytest.mark.parametrize("method", ["single", "ward"])
def test_rejects_distances_that_overflow_float32(method):
    with pytest.raises(ValueError, match="too large for float32"):
        cuhc.linkage(huge_spread(), method=method)
    with pytest.raises(ValueError, match="too large for float32"):
        cuhc.pairwise_distances(huge_spread())
    with pytest.raises(ValueError, match="too large for float32"):
        cuhc.HDBSCAN().fit(huge_spread())


def test_accepts_data_far_from_the_origin():
    # The bound uses each feature's spread, not its magnitude: 1e20 squared
    # overflows float32, but differences of ~1e16 don't. (The spread must
    # still exceed float32's spacing at 1e20, ~1e13, to keep points apart.)
    X = 1e20 + 1e16 * blobs(n_samples=300, n_features=3, seed=61).astype(np.float64)
    X = X.astype(np.float32)
    Z = cuhc.linkage(X, method="single")
    assert np.isfinite(Z[:, 2]).all() and Z[-1, 2] > 1e15


def test_ward_bound_includes_cluster_sizes():
    # Squared distances fit, but Ward's 2 n_a n_b / (n_a + n_b) factor
    # (up to n) would overflow.
    X = np.array([[0.0], [1e18]] * 500, dtype=np.float32)
    cuhc.linkage(X, method="single")
    with pytest.raises(ValueError, match="too large for float32"):
        cuhc.linkage(X, method="ward")
