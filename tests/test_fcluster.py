"""cuhc.fcluster against scipy.cluster.hierarchy.fcluster, on the same tree."""

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster as scipy_fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage

import cuhc
from _helpers import assert_same_partition, blobs


@pytest.fixture(scope="module")
def tree():
    return scipy_linkage(blobs(n_samples=300, seed=4).astype(np.float64), method="average")


@pytest.mark.parametrize("t", [1, 2, 7, 50, 300])
def test_maxclust_matches_scipy(tree, t):
    ours = cuhc.fcluster(tree, t, criterion="maxclust")
    ref = scipy_fcluster(tree, t, criterion="maxclust")
    assert ours.min() == 1  # 1-based, like scipy
    assert_same_partition(ours, ref)


@pytest.mark.parametrize("quantile", [0.0, 0.5, 0.9, 0.99, 1.0])
def test_distance_matches_scipy(tree, quantile):
    t = float(np.quantile(tree[:, 2], quantile))
    assert_same_partition(cuhc.fcluster(tree, t, criterion="distance"),
                          scipy_fcluster(tree, t, criterion="distance"))


def test_works_on_cuhc_tree():
    X = blobs(n_samples=200, seed=5)
    labels = cuhc.fcluster(cuhc.linkage(X, method="ward"), 5)
    assert len(np.unique(labels)) == 5


@pytest.mark.parametrize("kwargs, message", [
    ({"t": 0}, "t >= 1"),
    ({"t": 3, "criterion": "inconsistent"}, "criterion"),
])
def test_rejects_bad_arguments(tree, kwargs, message):
    with pytest.raises(ValueError, match=message):
        cuhc.fcluster(tree, **kwargs)
