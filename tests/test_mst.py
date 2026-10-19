"""Single linkage through the Borůvka minimum spanning tree (algorithm="mst").

The MST path never stores the distance matrix, so it's checked against both
scikit-learn and cuhc's own dense path.
"""

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster as scipy_fcluster

import cuhc
from _helpers import assert_same_dendrogram, assert_same_partition, blobs
from test_linkage import ATOL, RTOL, sklearn_full_tree, subtree_sizes

METRICS = ["euclidean", "sqeuclidean", "manhattan", "chebyshev", "cosine"]


def assert_same_tree(Z, ref_children, ref_distances):
    np.testing.assert_array_equal(np.sort(Z[:, :2].astype(np.int64), axis=1),
                                  np.sort(ref_children, axis=1))
    np.testing.assert_allclose(Z[:, 2], ref_distances, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev", "cosine"])
@pytest.mark.parametrize("n_samples", [2, 3, 127, 128, 129, 300])
def test_matches_sklearn(metric, n_samples):
    # Sizes around the 128-point tile edge (64 for cosine).
    X = blobs(n_samples=n_samples, seed=n_samples)
    Z = cuhc.linkage(X, method="single", metric=metric, algorithm="mst")
    assert_same_tree(Z, *sklearn_full_tree(X, "single", metric))


def test_matches_sklearn_on_many_points():
    # Many Borůvka rounds, components of very different sizes.
    #
    # With thousands of merges, some heights that differ in float64 (sklearn)
    # round to the same float32 value (cuhc), e.g. 1.782150126 and 1.782150202
    # both become 1.782150149. Tied merges may come out in either order, which
    # also renumbers the clusters they create, so the trees are compared as
    # sets of clusters rather than row by row.
    X = blobs(n_samples=8000, n_features=10, centers=20, seed=12)
    Z = cuhc.linkage(X, method="single", algorithm="mst")
    ref_children, ref_distances = sklearn_full_tree(X, "single", "euclidean")
    ref_Z = np.column_stack([ref_children, ref_distances, subtree_sizes(ref_children, len(X))])
    assert_same_dendrogram(Z, ref_Z, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("metric", METRICS)
def test_matches_dense_path(metric):
    X = blobs(n_samples=1500, n_features=12, seed=13)
    mst = cuhc.linkage(X, method="single", metric=metric, algorithm="mst")
    dense = cuhc.linkage(X, method="single", metric=metric, algorithm="dense")
    np.testing.assert_allclose(mst[:, 2], dense[:, 2], rtol=RTOL, atol=ATOL)
    for k in [2, 10, 100]:
        assert_same_partition(scipy_fcluster(mst, k, "maxclust"),
                              scipy_fcluster(dense, k, "maxclust"))


def test_auto_uses_mst_for_single_linkage():
    X = blobs(n_samples=500, seed=14)
    np.testing.assert_array_equal(cuhc.linkage(X, method="single"),
                                  cuhc.linkage(X, method="single", algorithm="mst"))


def test_many_ties_give_the_same_heights_as_dense():
    # Points on an integer grid: huge numbers of equal distances, so Borůvka's
    # tie-breaking must still produce a tree (no cycles, no missing edges).
    grid = np.stack(np.meshgrid(np.arange(20), np.arange(20)), axis=-1).reshape(-1, 2)
    X = grid.astype(np.float32)
    mst = cuhc.linkage(X, method="single", algorithm="mst")
    dense = cuhc.linkage(X, method="single", algorithm="dense")
    assert mst.shape == (len(X) - 1, 4)
    np.testing.assert_array_equal(mst[:, 2], dense[:, 2])  # all exactly 1.0
    # With all heights equal, any merge order is a valid tree, and the order
    # depends on how ties are broken (by point index, and spatial pruning
    # renumbers the points). Check the structure instead of the order.
    ids = mst[:, :2].astype(np.int64)
    n = len(X)
    np.testing.assert_array_equal(np.sort(ids.ravel()), np.arange(2 * n - 2))
    np.testing.assert_array_equal(mst[:, 3], subtree_sizes(ids, n))


def test_agglomerative_clustering_with_mst():
    X = blobs(n_samples=600, seed=15)
    ours = cuhc.AgglomerativeClustering(n_clusters=6, linkage="single", algorithm="mst").fit(X)
    dense = cuhc.AgglomerativeClustering(n_clusters=6, linkage="single", algorithm="dense").fit(X)
    assert_same_partition(ours.labels_, dense.labels_)


@pytest.mark.parametrize("call, message", [
    (lambda X: cuhc.linkage(X, method="complete", algorithm="mst"), "supports linkage"),
    (lambda X: cuhc.linkage(X, method="single", algorithm="fastest"), "unknown algorithm"),
    (lambda X: cuhc.AgglomerativeClustering(linkage="ward", algorithm="mst").fit(X),
     "supports linkage"),
    (lambda X: cuhc.linkage(np.where(X > 0, X, np.nan), method="single"), "NaN"),
])
def test_rejects_invalid_requests(call, message):
    with pytest.raises(ValueError, match=message):
        call(blobs(n_samples=50))
