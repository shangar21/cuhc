"""Ward linkage through rounds of reciprocal nearest neighbours (algorithm="rnn").

Checked against scikit-learn, cuhc's dense path and the chain, including the
hand-over to the dense and chain tails that rounds fall back to when they stop
making progress.
"""

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster as scipy_fcluster

import cuhc
from cuhc._linkage import _ward_rnn_with_options
from _helpers import assert_same_dendrogram, assert_same_partition, blobs
from test_linkage import ATOL, RTOL, subtree_sizes
from test_nn_chain import sklearn_ward_tree


def geometric_line(n):
    """Points on a line with gaps growing by 1%: every point's nearest is its
    left neighbour, so a round merges about one pair -- the rounds' worst case."""
    x = 1.01 ** np.arange(n, dtype=np.float64)
    return ((x - x[0]) / (x[-1] - x[0]))[:, None].astype(np.float32)


@pytest.mark.parametrize("n_samples", [2, 3, 127, 128, 129, 600])
@pytest.mark.parametrize("n_features", [1, 7, 64])
def test_matches_sklearn(n_samples, n_features):
    # Sizes around the 128-point tile; 1 feature has the most ties.
    X = blobs(n_samples=n_samples, n_features=n_features, seed=n_samples + n_features)
    Z = cuhc.linkage(X, method="ward", algorithm="rnn")
    assert_same_dendrogram(Z, sklearn_ward_tree(X), rtol=RTOL, atol=ATOL)


def test_matches_sklearn_on_many_points():
    # Dozens of rounds, many tiles.
    X = blobs(n_samples=6000, n_features=10, centers=25, seed=51)
    Z = cuhc.linkage(X, method="ward", algorithm="rnn")
    assert_same_dendrogram(Z, sklearn_ward_tree(X), rtol=RTOL, atol=ATOL)


def test_matches_chain_and_dense():
    X = blobs(n_samples=3000, n_features=16, seed=52)
    rnn = cuhc.linkage(X, method="ward", algorithm="rnn")
    for other in ["nn_chain", "dense"]:
        Z = cuhc.linkage(X, method="ward", algorithm=other)
        assert_same_dendrogram(rnn, Z, rtol=RTOL, atol=ATOL)
        for k in [2, 10, 100]:
            assert_same_partition(scipy_fcluster(rnn, k, "maxclust"),
                                  scipy_fcluster(Z, k, "maxclust"))


@pytest.mark.parametrize("tail", ["dense", "chain"])
def test_tails_match_sklearn(tail):
    # min_progress=1 leaves the rounds after the first one, so most merges
    # come from the tail; dense_below forces the dense tail from the start.
    X = blobs(n_samples=2500, n_features=8, seed=53)
    reference = sklearn_ward_tree(X)
    Z = _ward_rnn_with_options(X, min_progress=1.0, allow_dense=(tail == "dense"))
    assert_same_dendrogram(Z, reference, rtol=RTOL, atol=ATOL)
    if tail == "dense":
        Z = _ward_rnn_with_options(X, min_progress=0.0, dense_below=len(X))
        assert_same_dendrogram(Z, reference, rtol=RTOL, atol=ATOL)
        # Rounds first, dense for the last 500 clusters.
        Z = _ward_rnn_with_options(X, min_progress=0.0, dense_below=500)
        assert_same_dendrogram(Z, reference, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("allow_dense", [True, False])
def test_worst_case_hands_over(allow_dense):
    X = geometric_line(1500)
    Z = _ward_rnn_with_options(X, min_progress=0.01, allow_dense=allow_dense)
    assert_same_dendrogram(Z, sklearn_ward_tree(X), rtol=RTOL, atol=ATOL)
    # Without the hand-over: ~750 rounds, slow but still correct.
    Z = _ward_rnn_with_options(X, min_progress=0.0)
    assert_same_dendrogram(Z, sklearn_ward_tree(X), rtol=RTOL, atol=ATOL)


def test_tree_is_well_formed():
    X = blobs(n_samples=1000, seed=54)
    Z = cuhc.linkage(X, method="ward", algorithm="rnn")
    n = len(X)
    ids = Z[:, :2].astype(np.int64)
    np.testing.assert_array_equal(np.sort(ids.ravel()), np.arange(2 * n - 2))
    assert np.all(ids < n + np.arange(n - 1)[:, None])  # clusters exist before they merge
    assert np.all(np.diff(Z[:, 2]) >= 0)                 # heights in order
    np.testing.assert_array_equal(Z[:, 3], subtree_sizes(ids, n))


def test_deterministic():
    # Merges within a round are recorded with atomics; the result must not
    # depend on their order.
    X = blobs(n_samples=4000, n_features=4, seed=55)
    first = cuhc.linkage(X, method="ward", algorithm="rnn")
    for _ in range(3):
        np.testing.assert_array_equal(cuhc.linkage(X, method="ward", algorithm="rnn"), first)


def test_duplicate_points():
    rng = np.random.default_rng(56)
    X = np.repeat(rng.normal(size=(60, 3)), 5, axis=0).astype(np.float32)
    Z = cuhc.linkage(X, method="ward", algorithm="rnn")
    dense = cuhc.linkage(X, method="ward", algorithm="dense")
    np.testing.assert_allclose(Z[:, 2], dense[:, 2], rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(Z[:240, 2], 0.0)
    assert_same_partition(scipy_fcluster(Z, 60, "maxclust"), np.repeat(np.arange(60), 5))


def test_auto_uses_rnn_for_ward():
    X = blobs(n_samples=500, seed=57)
    np.testing.assert_array_equal(cuhc.linkage(X, method="ward"),
                                  cuhc.linkage(X, method="ward", algorithm="rnn"))
    ours = cuhc.AgglomerativeClustering(n_clusters=7).fit(X)
    dense = cuhc.AgglomerativeClustering(n_clusters=7, algorithm="dense").fit(X)
    assert_same_partition(ours.labels_, dense.labels_)


@pytest.mark.parametrize("method", ["single", "complete", "average"])
def test_rejects_other_linkages(method):
    with pytest.raises(ValueError, match="supports linkage"):
        cuhc.linkage(blobs(n_samples=50), method=method, algorithm="rnn")
