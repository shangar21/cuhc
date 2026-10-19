"""Spatial pruning (src/spatial.cuh) must not change any result.

Each test runs the same input with pruning forced on and forced off. Pruning
renumbers the points in spatial order, and that changes two things only:
  - which of two equal-weight edges wins a tie, so the order of equal-height
    merges (HDBSCAN's mutual reachability has many exact ties);
  - for Ward, which cluster sits in the lower slot, so the last bit of a
    merged centroid (heights agree to ~3e-7 relative).
Single linkage on continuous data has no ties, so it must be identical.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

import cuhc
from cuhc import _lib
from cuhc._hdbscan import hdbscan_tree
from scipy.cluster.hierarchy import cophenet
from _helpers import assert_same_dendrogram


@pytest.fixture
def both_modes():
    """Calls fn() with pruning on and off; returns both results."""
    def run(fn):
        results = []
        for mode in ["always", "never"]:
            _lib.set_pruning(mode)
            try:
                results.append(fn())
            finally:
                _lib.set_pruning("auto")
        return results
    return run


def clustered(n, n_features, seed):
    X, _ = make_blobs(n_samples=n, n_features=n_features, centers=30,
                      cluster_std=np.random.default_rng(seed).uniform(0.3, 2.0, 30),
                      random_state=seed)
    return X.astype(np.float32)


@pytest.mark.parametrize("n_features", [1, 2, 3, 8, 20])
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "manhattan", "chebyshev",
                                    "minkowski"])
def test_single_linkage_is_unchanged(both_modes, n_features, metric):
    # 3000 points = 24 tiles: the walk visits, skips and wraps at both ends.
    X = clustered(3000, n_features, seed=n_features)
    pruned, full = both_modes(lambda: cuhc.linkage(X, method="single", metric=metric, p=3.0))
    np.testing.assert_array_equal(pruned, full)


@pytest.mark.parametrize("n_features", [2, 8, 20])
def test_ward_is_unchanged(both_modes, n_features):
    X = clustered(3000, n_features, seed=10 + n_features)
    pruned, full = both_modes(lambda: cuhc.linkage(X, method="ward", algorithm="rnn"))
    assert_same_dendrogram(pruned, full, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_features", [2, 8])
@pytest.mark.parametrize("min_samples", [1, 5, 40])
@pytest.mark.parametrize("alpha", [1.0, 0.7])
def test_hdbscan_tree_is_unchanged(both_modes, n_features, min_samples, alpha):
    X = clustered(3000, n_features, seed=20 + n_features)
    (pruned_tree, pruned_core), (full_tree, full_core) = both_modes(
        lambda: hdbscan_tree(X, min_samples, alpha=alpha))
    np.testing.assert_array_equal(pruned_core, full_core)
    # The same tree up to the order of tied merges: equal heights, and equal
    # cophenetic distances (the height at which each pair of points joins).
    np.testing.assert_array_equal(np.sort(pruned_tree[:, 2]), np.sort(full_tree[:, 2]))
    np.testing.assert_array_equal(cophenet(pruned_tree), cophenet(full_tree))


def test_sizes_that_are_not_whole_tiles(both_modes):
    for n in [129, 255, 257, 1000]:
        X = clustered(n, 3, seed=n)
        pruned, full = both_modes(lambda: cuhc.linkage(X, method="single"))
        np.testing.assert_array_equal(pruned, full)


def test_pruning_skips_tiles_on_clustered_data():
    X = clustered(20000, 2, seed=30)
    _lib.prune_stats()  # reset
    _lib.set_pruning("always")
    try:
        cuhc.linkage(X, method="single")
    finally:
        _lib.set_pruning("auto")
    computed, total = _lib.prune_stats()
    assert total > 0 and computed < 0.2 * total


def test_cosine_is_never_pruned(both_modes):
    # Cosine has no box bound; forcing pruning on must leave it alone.
    X = clustered(2000, 4, seed=31)
    _lib.prune_stats()
    pruned, full = both_modes(lambda: cuhc.linkage(X, method="single", metric="cosine"))
    np.testing.assert_array_equal(pruned, full)
    assert _lib.prune_stats() == (0, 0)


def test_rejects_unknown_mode():
    with pytest.raises(KeyError):
        _lib.set_pruning("sometimes")
