"""Ward linkage through the nearest-neighbour chain (algorithm="nn_chain").

The chain never stores the distance matrix, so it's checked against both
scikit-learn and cuhc's own dense path.
"""

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster as scipy_fcluster

import cuhc
from _helpers import assert_same_dendrogram, assert_same_partition, blobs
from test_linkage import ATOL, RTOL, sklearn_full_tree, subtree_sizes


def sklearn_ward_tree(X):
    children, distances = sklearn_full_tree(X, "ward", "euclidean")
    return np.column_stack([children, distances, subtree_sizes(children, len(X))])


@pytest.mark.parametrize("n_samples", [2, 3, 255, 256, 257, 600])
@pytest.mark.parametrize("n_features", [1, 7, 64])
def test_matches_sklearn(n_samples, n_features):
    # Sizes around the 256-slot block range; 1 feature has the most ties.
    X = blobs(n_samples=n_samples, n_features=n_features, seed=n_samples + n_features)
    Z = cuhc.linkage(X, method="ward", algorithm="nn_chain")
    assert_same_dendrogram(Z, sklearn_ward_tree(X), rtol=RTOL, atol=ATOL)


def test_matches_sklearn_on_many_points():
    # Thousands of chain steps and merges; many blocks.
    X = blobs(n_samples=6000, n_features=10, centers=25, seed=41)
    Z = cuhc.linkage(X, method="ward", algorithm="nn_chain")
    assert_same_dendrogram(Z, sklearn_ward_tree(X), rtol=RTOL, atol=ATOL)


def test_matches_dense_path():
    X = blobs(n_samples=3000, n_features=16, seed=42)
    chain = cuhc.linkage(X, method="ward", algorithm="nn_chain")
    dense = cuhc.linkage(X, method="ward", algorithm="dense")
    assert_same_dendrogram(chain, dense, rtol=RTOL, atol=ATOL)
    for k in [2, 10, 100]:
        assert_same_partition(scipy_fcluster(chain, k, "maxclust"),
                              scipy_fcluster(dense, k, "maxclust"))


def test_tree_is_well_formed():
    X = blobs(n_samples=1000, seed=43)
    Z = cuhc.linkage(X, method="ward", algorithm="nn_chain")
    n = len(X)
    ids = Z[:, :2].astype(np.int64)
    np.testing.assert_array_equal(np.sort(ids.ravel()), np.arange(2 * n - 2))
    assert np.all(ids < n + np.arange(n - 1)[:, None])  # clusters exist before they merge
    assert np.all(np.diff(Z[:, 2]) >= 0)                 # heights in order
    np.testing.assert_array_equal(Z[:, 3], subtree_sizes(ids, n))


def test_duplicate_points():
    # Exact ties everywhere: zero-distance merges first, and the chain must
    # still terminate and produce a valid tree.
    rng = np.random.default_rng(44)
    X = np.repeat(rng.normal(size=(60, 3)), 5, axis=0).astype(np.float32)
    Z = cuhc.linkage(X, method="ward", algorithm="nn_chain")
    dense = cuhc.linkage(X, method="ward", algorithm="dense")
    np.testing.assert_allclose(Z[:, 2], dense[:, 2], rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(Z[:240, 2], 0.0)
    assert_same_partition(scipy_fcluster(Z, 60, "maxclust"), np.repeat(np.arange(60), 5))


def test_agglomerative_clustering_with_nn_chain():
    X = blobs(n_samples=800, seed=45)
    ours = cuhc.AgglomerativeClustering(n_clusters=7, algorithm="nn_chain").fit(X)
    dense = cuhc.AgglomerativeClustering(n_clusters=7, algorithm="dense").fit(X)
    assert_same_partition(ours.labels_, dense.labels_)


@pytest.mark.parametrize("method", ["single", "complete", "average"])
def test_rejects_other_linkages(method):
    with pytest.raises(ValueError, match="supports linkage"):
        cuhc.linkage(blobs(n_samples=50), method=method, algorithm="nn_chain")
