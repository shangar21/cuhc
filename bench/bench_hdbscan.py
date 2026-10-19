"""Times cuhc.HDBSCAN against sklearn.cluster.HDBSCAN.

Usage: python3 bench/bench_hdbscan.py [--sizes 5000 20000 ...] [--sklearn-max 50000]
"""

import argparse
import pathlib
import sys
import time

import numpy as np
from sklearn.cluster import HDBSCAN as SkHDBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import cuhc  # noqa: E402
from cuhc._hdbscan import hdbscan_tree, tree_to_labels  # noqa: E402


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 20000, 50000])
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=25)
    parser.add_argument("--sklearn-max", type=int, default=50000)
    args = parser.parse_args()
    mcs = args.min_cluster_size

    cuhc.HDBSCAN().fit(np.random.rand(20, 2))  # CUDA context, kernel loading

    print(f"{args.features} features, min_cluster_size={mcs}, 20 blobs + 10% noise")
    print(f"{'n':>8} {'cuhc s':>8} {'(GPU tree':>10} {'extract)':>9} {'sklearn s':>10} "
          f"{'speedup':>8} {'ARI':>7}")
    for n in args.sizes:
        rng = np.random.default_rng(n)
        X, _ = make_blobs(n_samples=int(n * 0.9), n_features=args.features, centers=20,
                          cluster_std=rng.uniform(0.5, 2.0, 20), random_state=n)
        X = np.vstack([X, rng.uniform(X.min(0), X.max(0), (n - len(X), args.features))])
        X = X.astype(np.float32)

        (Z, _), tree_s = timed(lambda: hdbscan_tree(X, mcs))
        _, extract_s = timed(lambda: tree_to_labels(Z, mcs))
        model, total_s = timed(lambda: cuhc.HDBSCAN(min_cluster_size=mcs).fit(X))
        line = f"{n:>8} {total_s:>8.2f} {tree_s:>10.2f} {extract_s:>9.2f}"
        if n <= args.sklearn_max:
            ref, sk_s = timed(lambda: SkHDBSCAN(min_cluster_size=mcs, copy=True).fit(
                X.astype(np.float64)))
            ari = adjusted_rand_score(ref.labels_, model.labels_)
            line += f" {sk_s:>10.2f} {sk_s / total_s:>7.1f}x {ari:>7.4f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
