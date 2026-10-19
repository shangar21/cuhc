"""Times the matrix-free paths with spatial pruning in auto mode, forced on and forced off.

Usage: python3 bench/bench_pruning.py [--sizes 100000 ...] [--features 2 8 32]
                                      [--tasks single ward hdbscan] [--data blobs normal]
"""

import argparse
import pathlib
import sys
import time

import numpy as np
from sklearn.datasets import make_blobs

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import cuhc  # noqa: E402
from cuhc import _lib  # noqa: E402

TASKS = {
    "single": lambda X: cuhc.linkage(X, method="single"),
    "ward": lambda X: cuhc.linkage(X, method="ward"),
    "hdbscan": lambda X: cuhc.HDBSCAN(min_cluster_size=25).fit(X),
}


def make_data(kind, n, n_features, seed):
    if kind == "normal":
        return np.random.default_rng(seed).normal(size=(n, n_features)).astype(np.float32)
    X, _ = make_blobs(n_samples=n, n_features=n_features, centers=20,
                      cluster_std=np.random.default_rng(seed).uniform(0.5, 2.0, 20),
                      random_state=seed)
    return X.astype(np.float32)


def timed(mode, task, X, repeats):
    """Best time of `repeats` runs, and the share of tiles the pruned passes computed."""
    _lib.set_pruning(mode)
    best = float("inf")
    for _ in range(repeats):
        _lib.prune_stats()
        start = time.perf_counter()
        TASKS[task](X)
        best = min(best, time.perf_counter() - start)
        computed, total = _lib.prune_stats()
    _lib.set_pruning("auto")
    return best, (computed / total if total else float("nan"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100000])
    parser.add_argument("--features", type=int, nargs="+", default=[2, 8, 16, 32])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--data", nargs="+", default=["blobs", "normal"], choices=["blobs", "normal"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["auto", "always", "never"],
                        choices=["auto", "always", "never"])
    args = parser.parse_args()

    cuhc.linkage(np.random.rand(10, 2).astype(np.float32))  # CUDA context, kernel loading
    print(f"{'data':>7} {'n':>8} {'d':>3} {'task':>8} {'auto s':>8} {'always s':>9} {'tiles':>7} "
          f"{'never s':>8} {'auto speedup':>13}")
    for kind in args.data:
        for n in args.sizes:
            for d in args.features:
                X = make_data(kind, n, d, seed=n + d)
                for task in args.tasks:
                    nan = (float("nan"), float("nan"))
                    run = lambda mode: timed(mode, task, X, args.repeats) if mode in args.modes else nan  # noqa: E731
                    auto_s, _ = run("auto")
                    pruned_s, share = run("always")
                    full_s, _ = run("never")
                    print(f"{kind:>7} {n:>8} {d:>3} {task:>8} {auto_s:>8.3f} {pruned_s:>9.3f} "
                          f"{share:>7.1%} {full_s:>8.3f} {full_s / auto_s:>12.2f}x", flush=True)


if __name__ == "__main__":
    main()
