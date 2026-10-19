"""Times cuhc.linkage against scikit-learn's AgglomerativeClustering (scipy underneath).

Usage: python3 bench/bench_linkage.py [--sizes 2000 5000 ...] [--methods single ...]
                                       [--algorithm auto|dense|mst|nn_chain|rnn] [--sklearn-max 20000]
"""

import argparse
import pathlib
import sys
import time

import numpy as np
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import cuhc  # noqa: E402
from cuhc._linkage import cut_tree  # noqa: E402


def best_time(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2000, 5000, 10000, 20000, 40000])
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--methods", nargs="+", default=["single", "complete", "average", "ward"])
    parser.add_argument("--algorithm", default="auto", choices=["auto", "dense", "mst", "nn_chain", "rnn"])
    parser.add_argument("--sklearn-max", type=int, default=20000,
                        help="skip sklearn above this size (it needs 4*n^2 bytes of RAM and is slow)")
    args = parser.parse_args()

    cuhc.linkage(np.random.rand(10, 2).astype(np.float32))  # CUDA context, kernel loading

    print(f"algorithm={args.algorithm}, {args.features} features")
    print(f"{'n':>7} {'method':>9} {'cuhc s':>9} {'sklearn s':>10} {'speedup':>8}  same labels (k=10)")
    for n in args.sizes:
        X = np.random.default_rng(n).normal(size=(n, args.features)).astype(np.float32)
        for method in args.methods:
            repeats = 3 if n <= 10000 else 1
            run = lambda: cuhc.linkage(X, method=method, algorithm=args.algorithm)  # noqa: E731
            ours_s = best_time(run, repeats)
            line = f"{n:>7} {method:>9} {ours_s:>9.3f}"
            if n <= args.sklearn_max:
                sk = AgglomerativeClustering(n_clusters=10, linkage=method)
                sk_s = best_time(lambda: sk.fit(X.astype(np.float64)), 1)
                ours_labels = cut_tree(run(), n - 10)
                same = len(set(zip(ours_labels, sk.labels_))) == 10
                line += f" {sk_s:>10.3f} {sk_s / ours_s:>7.1f}x  {same}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
