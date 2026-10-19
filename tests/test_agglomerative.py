"""cuhc.AgglomerativeClustering against sklearn.cluster.AgglomerativeClustering.

Contract: same constructor parameters (n_clusters, metric, linkage,
distance_threshold, compute_distances) and same fitted attributes (labels_,
n_clusters_, n_leaves_, n_features_in_, children_, distances_) as sklearn, and
it is a real sklearn estimator (works with clone, get_params, pipelines).
Label numbers may differ; the clusters themselves must be identical.
"""

import numpy as np
import pytest
from sklearn.base import ClusterMixin, clone
from sklearn.cluster import AgglomerativeClustering as SkAgglomerativeClustering
from sklearn.metrics import pairwise_distances as sk_pairwise_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cuhc
from _helpers import assert_same_partition, blobs, requires

pytestmark = requires("AgglomerativeClustering")

LINKAGES = ["single", "complete", "average", "ward"]
METRICS = ["euclidean", "manhattan", "chebyshev", "cosine"]
CASES = [(m, metric) for m in LINKAGES for metric in METRICS
         if m != "ward" or metric == "euclidean"]
CASE_IDS = [f"{m}-{d}" for m, d in CASES]


def fit_sklearn(X, linkage, metric, **params):
    """sklearn reference, via a float64 precomputed matrix for non-Ward linkages."""
    X64 = X.astype(np.float64)
    if linkage == "ward":
        return SkAgglomerativeClustering(linkage="ward", metric="euclidean", **params).fit(X64)
    model = SkAgglomerativeClustering(linkage=linkage, metric="precomputed", **params)
    return model.fit(sk_pairwise_distances(X64, metric=metric))


@pytest.mark.parametrize("linkage, metric", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("n_clusters", [1, 2, 5, 12])
def test_labels_match_sklearn(linkage, metric, n_clusters):
    X = blobs(n_samples=300, seed=n_clusters)
    ours = cuhc.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage,
                                        metric=metric).fit(X)
    ref = fit_sklearn(X, linkage, metric, n_clusters=n_clusters)
    assert ours.n_clusters_ == ref.n_clusters_ == n_clusters
    assert_same_partition(ours.labels_, ref.labels_)


@pytest.mark.parametrize("linkage, metric", CASES, ids=CASE_IDS)
def test_distance_threshold_matches_sklearn(linkage, metric):
    X = blobs(n_samples=300, seed=11)
    ref_tree = fit_sklearn(X, linkage, metric, n_clusters=None, distance_threshold=0.0)
    # Cut halfway between two merges near the top of the tree, so float32 vs
    # float64 rounding can't move a merge across the threshold.
    heights = np.sort(ref_tree.distances_)
    threshold = float((heights[-6] + heights[-5]) / 2)

    ours = cuhc.AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                        linkage=linkage, metric=metric).fit(X)
    ref = fit_sklearn(X, linkage, metric, n_clusters=None, distance_threshold=threshold)
    assert ours.n_clusters_ == ref.n_clusters_
    assert_same_partition(ours.labels_, ref.labels_)


@pytest.mark.parametrize("linkage", LINKAGES)
def test_fitted_attributes_match_sklearn(linkage):
    X = blobs(n_samples=120, n_features=5, seed=3)
    ours = cuhc.AgglomerativeClustering(n_clusters=4, linkage=linkage,
                                        compute_distances=True).fit(X)
    ref = fit_sklearn(X, linkage, "euclidean", n_clusters=4, compute_distances=True)

    assert ours.n_leaves_ == ref.n_leaves_ == 120
    assert ours.n_features_in_ == 5
    assert ours.children_.shape == ref.children_.shape == (119, 2)
    np.testing.assert_array_equal(np.sort(ours.children_, axis=1), np.sort(ref.children_, axis=1))
    np.testing.assert_allclose(ours.distances_, ref.distances_, rtol=1e-4, atol=1e-5)


def test_defaults_match_sklearn():
    ours, ref = cuhc.AgglomerativeClustering(), SkAgglomerativeClustering()
    for name in ["n_clusters", "metric", "linkage", "distance_threshold", "compute_distances"]:
        assert getattr(ours, name) == getattr(ref, name), name


def test_fit_predict_returns_labels():
    X = blobs(n_samples=100)
    model = cuhc.AgglomerativeClustering(n_clusters=3)
    np.testing.assert_array_equal(model.fit_predict(X), model.labels_)


def test_is_a_sklearn_estimator():
    model = cuhc.AgglomerativeClustering(n_clusters=3, linkage="average", metric="manhattan")
    assert isinstance(model, ClusterMixin)
    assert clone(model).get_params() == model.get_params()

    X = blobs(n_samples=150)
    labels = make_pipeline(StandardScaler(), clone(model)).fit_predict(X)
    ref = fit_sklearn(StandardScaler().fit_transform(X).astype(np.float32), "average",
                      "manhattan", n_clusters=3)
    assert_same_partition(labels, ref.labels_)


@pytest.mark.parametrize("params, message", [
    ({"linkage": "ward", "metric": "cosine"}, "euclidean"),
    ({"n_clusters": 3, "distance_threshold": 1.0}, "n_clusters"),
    ({"n_clusters": None, "distance_threshold": None}, "n_clusters"),
    ({"linkage": "median-of-means"}, "linkage"),
])
def test_rejects_invalid_parameters_like_sklearn(params, message):
    with pytest.raises(ValueError, match=message):
        cuhc.AgglomerativeClustering(**params).fit(blobs(n_samples=20))
