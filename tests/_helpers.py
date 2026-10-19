"""Shared test data and assertions."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

import cuhc


def requires(name):
    """Skip a test (or module, via pytestmark) until cuhc.<name> exists."""
    return pytest.mark.skipif(not hasattr(cuhc, name), reason=f"cuhc.{name} not implemented yet")


def blobs(n_samples=400, n_features=8, centers=5, noise_fraction=0.0, seed=0):
    """Clustered float32 data, optionally with uniform background noise.

    Values come from continuous distributions, so tied distances (which would
    let two correct implementations merge in different orders) are vanishingly
    unlikely. Returned as float32 because that's what cuhc computes in; pass
    X.astype(np.float64) to scikit-learn so both see identical input.
    """
    rng = np.random.default_rng(seed)
    n_noise = int(n_samples * noise_fraction)
    X, _ = make_blobs(
        n_samples=n_samples - n_noise,
        n_features=n_features,
        centers=centers,
        cluster_std=rng.uniform(0.5, 2.0, size=centers),
        random_state=seed,
    )
    if n_noise:
        low, high = X.min(axis=0), X.max(axis=0)
        X = np.vstack([X, rng.uniform(low, high, size=(n_noise, n_features))])
    return X.astype(np.float32)


def assert_same_partition(ours, ref):
    """Labels must describe the same clusters; the label numbers may differ.

    Noise (label -1, used by HDBSCAN) must match exactly.
    """
    ours, ref = np.asarray(ours), np.asarray(ref)
    assert ours.shape == ref.shape
    np.testing.assert_array_equal(ours == -1, ref == -1, err_msg="noise points differ")
    clustered = ref != -1
    pairs = set(zip(ours[clustered].tolist(), ref[clustered].tolist()))
    # A bijection between our labels and theirs: each label pairs with exactly one other.
    assert len(pairs) == len({o for o, _ in pairs}) == len({r for _, r in pairs}), (
        f"partitions differ: {len({o for o, _ in pairs})} of our clusters vs "
        f"{len({r for _, r in pairs})} reference clusters map onto {len(pairs)} pairs"
    )


def tree_clusters(Z):
    """{(smallest point, size): merge height} for every cluster a merge tree creates.

    In a hierarchy any two clusters are nested or disjoint, so (smallest point,
    size) identifies a cluster uniquely. Two trees with the same clusters at the
    same heights are the same dendrogram, even if tied merges happened in a
    different order (which renumbers the cluster ids in Z).
    """
    Z = np.asarray(Z)
    n = len(Z) + 1
    smallest = np.empty(2 * n - 1, dtype=np.int64)
    smallest[:n] = np.arange(n)
    clusters = {}
    for i, (a, b, height, size) in enumerate(Z):
        smallest[n + i] = min(smallest[int(a)], smallest[int(b)])
        clusters[(int(smallest[n + i]), int(size))] = float(height)
    return clusters


def assert_same_dendrogram(Z, ref_Z, rtol=1e-4, atol=1e-5):
    """Same clusters at the same heights; the order of tied merges may differ."""
    ours, ref = tree_clusters(Z), tree_clusters(ref_Z)
    assert ours.keys() == ref.keys(), (
        f"{len(ours.keys() - ref.keys())} clusters differ between the trees"
    )
    keys = sorted(ref)
    np.testing.assert_allclose([ours[k] for k in keys], [ref[k] for k in keys],
                               rtol=rtol, atol=atol)
