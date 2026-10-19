"""cuhc.pairwise_distances against sklearn.metrics.pairwise_distances."""

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances as sk_pairwise_distances

import cuhc

# (cuhc metric, sklearn metric, kwargs for both)
METRICS = [
    ("euclidean", "euclidean", {}),
    ("sqeuclidean", "sqeuclidean", {}),
    ("manhattan", "manhattan", {}),
    ("chebyshev", "chebyshev", {}),
    ("minkowski", "minkowski", {"p": 3.0}),
    ("minkowski", "minkowski", {"p": 1.5}),
    ("cosine", "cosine", {}),
]
METRIC_IDS = [f"{m}-{kw['p']}" if kw else m for m, _, kw in METRICS]

# Sizes chosen to hit tile edges: the kernel uses 128x128 (64x64 for cosine)
# output tiles and 8-feature slices.
SHAPES = [(1, 1), (7, 3), (64, 8), (129, 9), (300, 64), (257, 100)]

# cuhc works in float32; the reference is computed in float64.
RTOL, ATOL = 1e-4, 1e-4


def reference(X, Y, sk_metric, kwargs):
    Y64 = None if Y is None else Y.astype(np.float64)
    return sk_pairwise_distances(X.astype(np.float64), Y64, metric=sk_metric, **kwargs)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("metric, sk_metric, kwargs", METRICS, ids=METRIC_IDS)
def test_matches_sklearn_single_input(metric, sk_metric, kwargs, shape):
    X = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    ours = cuhc.pairwise_distances(X, metric=metric, **kwargs)
    assert ours.shape == (shape[0], shape[0])
    assert ours.dtype == np.float32
    np.testing.assert_allclose(ours, reference(X, None, sk_metric, kwargs), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("metric, sk_metric, kwargs", METRICS, ids=METRIC_IDS)
def test_matches_sklearn_two_inputs(metric, sk_metric, kwargs):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 20)).astype(np.float32)
    Y = rng.normal(loc=0.5, size=(333, 20)).astype(np.float32)
    ours = cuhc.pairwise_distances(X, Y, metric=metric, **kwargs)
    assert ours.shape == (150, 333)
    np.testing.assert_allclose(ours, reference(X, Y, sk_metric, kwargs), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("metric, sk_metric, kwargs", METRICS, ids=METRIC_IDS)
def test_symmetric_with_zero_diagonal(metric, sk_metric, kwargs):
    X = np.random.default_rng(2).normal(size=(200, 17)).astype(np.float32)
    D = cuhc.pairwise_distances(X, metric=metric, **kwargs)
    np.testing.assert_allclose(D, D.T, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=ATOL)


def test_cosine_zero_vector_matches_sklearn():
    # sklearn treats a zero vector as having cosine similarity 0 with everything,
    # i.e. distance 1.
    X = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    Y = np.array([[0.0, 0.0, 0.0], [-1.0, 0.5, 2.0]], dtype=np.float32)
    ours = cuhc.pairwise_distances(X, Y, metric="cosine")
    np.testing.assert_allclose(ours, reference(X, Y, "cosine", {}), atol=ATOL)


def test_large_values_keep_relative_accuracy():
    X = np.random.default_rng(3).normal(scale=1e4, size=(100, 32)).astype(np.float32)
    ours = cuhc.pairwise_distances(X, metric="euclidean")
    np.testing.assert_allclose(ours, reference(X, None, "euclidean", {}), rtol=RTOL)


@pytest.mark.parametrize("make_input", [
    pytest.param(lambda X: X.astype(np.float64), id="float64"),
    pytest.param(lambda X: np.asfortranarray(X), id="fortran-order"),
    pytest.param(lambda X: np.repeat(X, 2, axis=1)[:, ::2], id="strided"),
    pytest.param(lambda X: X.tolist(), id="list"),
])
def test_accepts_any_array_layout(make_input):
    X = np.random.default_rng(4).normal(size=(50, 6)).astype(np.float32)
    ours = cuhc.pairwise_distances(make_input(X))
    np.testing.assert_allclose(ours, reference(X, None, "euclidean", {}), rtol=RTOL, atol=ATOL)


def test_metric_aliases_match():
    X = np.random.default_rng(5).normal(size=(40, 5)).astype(np.float32)
    np.testing.assert_array_equal(cuhc.pairwise_distances(X, metric="l2"),
                                  cuhc.pairwise_distances(X, metric="euclidean"))
    np.testing.assert_array_equal(cuhc.pairwise_distances(X, metric="l1"),
                                  cuhc.pairwise_distances(X, metric="manhattan"))
    np.testing.assert_array_equal(cuhc.pairwise_distances(X, metric="cityblock"),
                                  cuhc.pairwise_distances(X, metric="manhattan"))


@pytest.mark.parametrize("call, message", [
    (lambda: cuhc.pairwise_distances(np.zeros((3, 2)), metric="nope"), "unknown metric"),
    (lambda: cuhc.pairwise_distances(np.zeros((3, 2)), np.zeros((3, 4))), "same number of features"),
    (lambda: cuhc.pairwise_distances(np.zeros(3)), "must be 2D"),
    (lambda: cuhc.pairwise_distances(np.zeros((0, 3))), "non-empty"),
    (lambda: cuhc.pairwise_distances(np.zeros((3, 2)), metric="minkowski", p=0), "positive"),
    (lambda: cuhc.pairwise_distances(np.array([[0.0, np.nan]])), "NaN or infinity"),
    (lambda: cuhc.pairwise_distances(np.zeros((2, 2)), np.array([[np.inf, 0.0]])), "NaN or infinity"),
])
def test_rejects_bad_input(call, message):
    with pytest.raises(ValueError, match=message):
        call()
