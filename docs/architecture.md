# Architecture

cuhc does hierarchical clustering on the GPU. It covers these methods:

- Agglomerative clustering with single, complete, average and Ward linkage.
- Density-based clustering (HDBSCAN).
- Built-in distance metrics, and metrics that a user writes as a C++ struct.
- Data sets from a small number of points to millions of points.

The Python API is the same as the scikit-learn API and the scipy API.

This document describes the full design. These documents give the details:

- `docs/distance_matrix.md`: the distance kernel.
- `docs/dense_linkage.md`: exact clustering with the dense matrix.
- `docs/boruvka_mst.md`: single linkage with a minimum spanning tree.
- `docs/hdbscan.md`: HDBSCAN.
- `docs/rnn_ward.md`: Ward linkage in rounds of reciprocal nearest neighbours.
- `docs/nn_chain.md`: Ward linkage with the nearest-neighbour chain.
- `docs/all_pairs_passes.md`: symmetry and spatial pruning for the all-pairs passes.

Contents:

1. [The main idea: all methods make a merge tree](#1-the-main-idea-all-methods-make-a-merge-tree)
2. [The stages](#2-the-stages)
3. [How cuhc selects a tree builder](#3-how-cuhc-selects-a-tree-builder)
4. [HDBSCAN with the single-linkage code](#4-hdbscan-with-the-single-linkage-code)
5. [Python API](#5-python-api)
6. [Code layout](#6-code-layout)
7. [Design rules](#7-design-rules)
8. [Tests](#8-tests)
9. [Planned work](#9-planned-work)

---

## 1. The main idea: all methods make a merge tree

Agglomerative clustering (single, complete, average and Ward) and HDBSCAN all make the
same object: a binary **merge tree** (dendrogram) of the n points. cuhc keeps it in the
scipy format. This is an array `Z` with n − 1 rows and 4 columns. Row i records one
merge:

```
Z[i] = [cluster_a, cluster_b, merge_distance, size_of_new_cluster]
```

The leaves (single points) are the clusters 0 to n − 1. Row i makes the cluster n + i.
The rows are in merge sequence.

Example: single linkage on 4 points at 0, 1, 5 and 7 on a line:

```
row  cluster_a  cluster_b  distance  size        makes
 0       0          1         1        2          cluster 4 = {0, 1}
 1       2          3         2        2          cluster 5 = {2, 3}
 2       4          5         4        4          cluster 6 = all points
```

The methods are different in only two parts:

- **How cuhc builds the tree:** the distance between clusters, and the algorithm.
- **How cuhc finds the clusters in the tree:** a cut at a height or at a number of
  clusters, or the HDBSCAN stability selection.

Thus, the library is a sequence of independent stages, with `Z` between them. A new
linkage needs only a tree builder. A new cluster selection needs only an extraction step.
Neither changes the other.

---

## 2. The stages

```
 points X ──► (1) metric ──► (2) transform ──► (3) tree builder ──► Z ──► (4) extraction ──► labels
              built-in or     optional, e.g.     algorithm from            cut at k clusters
              user-defined    HDBSCAN mutual     linkage and size          or a threshold, or
                              reachability                                  HDBSCAN stability
              ─────────────── GPU ──────────────────────────────────       ──── CPU ────
```

### 2.1 Metric

A metric is a struct with `init`, `accumulate(acc, a, b)` and `finalize(acc)` (refer to
`src/distances.cuh` and `docs/distance_matrix.md` section 2). All kernels that use
distances are templates of the metric. Thus, you write a metric one time, and it operates
in all algorithms.

The **built-in metrics** are compiled before use: euclidean, sqeuclidean, manhattan,
chebyshev, minkowski(p) and cosine.

Two traits tell the kernels what a metric permits (`metrics::IsSymmetric` and
`metrics::HasBoxBound`, refer to `docs/all_pairs_passes.md`).

### 2.2 Transform (optional)

A transform changes each distance into a different edge weight. The kernels apply it
before they find the minimum of each row. cuhc uses two transforms: the HDBSCAN mutual
reachability distance (section 4) and the Ward distance between clusters
(`docs/rnn_ward.md`). The tree builders do not change for a transform.

### 2.3 Tree builder

The tree builder makes `Z` on the GPU. There are several algorithms, with very different
memory needs. The correct algorithm depends on the linkage and on the data size
(section 3).

### 2.4 Extraction

The extraction changes `Z` into labels. It uses only the tree (O(n) memory, and
O(n log n) time or less). Thus, it operates on the CPU, where it is easy to read and to
make the same as scikit-learn:

- **Cut at a number of clusters or a height:** `n_clusters` or `distance_threshold` for
  `AgglomerativeClustering`, and `fcluster` for the scipy API.
- **HDBSCAN:** condense the tree with `min_cluster_size`, calculate the stability of each
  cluster, then select clusters by excess of mass (`eom`) or `leaf`, optionally with
  `cluster_selection_epsilon`. The membership `probabilities_` come from the same
  condensed tree. The condensing step is in C++. The other steps are in NumPy.

---

## 3. How cuhc selects a tree builder

The important question is: when two clusters merge, how does the algorithm find the
distance from the new cluster to all other clusters? The answer decides the memory that a
linkage needs.

| Linkage | The cluster distance comes from | Builder | Memory |
|---|---|---|---|
| single | nothing more: single linkage is the minimum spanning tree (MST) of the points | Borůvka MST (`"mst"`) | O(n) |
| HDBSCAN | the same, on the mutual reachability distance | Borůvka MST | O(n) |
| ward (Euclidean only) | the cluster sizes and centroids | reciprocal-nearest-neighbour rounds (`"rnn"`) | O(n·d) |
| average, complete | all pairs of points between the clusters | dense matrix with Lance-Williams (`"dense"`) | O(n²) |

All builders are exact.

The builders:

- **Dense matrix.** Calculate the n × n matrix with the tiled kernel. Then find the
  nearest active pair (a parallel argmin) and change one row and one column with the
  Lance-Williams formula, again and again. This algorithm is simple, and it is the
  reference for the other algorithms. Approximately 55,000 points fit in 12 GB in
  float32.
- **Borůvka MST.** In each round, each component finds its nearest point in a *different*
  component. Each component keeps the shortest of those edges. The algorithm adds the
  edges, and the components merge. The number of components decreases to half or less in
  each round. Thus, there are approximately log₂ n rounds.
  - Each round is one pass of the tiled distance kernel. The kernel finds the minimum and
    the argmin of each row and does not write the distances. Threads that share an output
    row combine their minimums with warp shuffles.
  - A sort of the MST edges and a union-find give `Z`.
- **Reciprocal-nearest-neighbour rounds.** In each round, all clusters find their nearest
  cluster together (one all-pairs pass with the tiled kernel), and all pairs that are the
  nearest cluster of each other merge. This is correct for "reducible" linkages, which
  includes Ward. The limit of the pass is the arithmetic, not the memory bandwidth.
- **Nearest-neighbour chain.** Follow the nearest neighbours from cluster to cluster until
  two clusters are the nearest cluster of each other, then merge them. This is also
  correct for reducible linkages. Each query compares one cluster with all others. Thus,
  the memory bandwidth is the limit, and the chain is slower than the rounds. cuhc uses it
  as `algorithm="nn_chain"`, and as a fallback for the rounds.

Centroid and median linkage are *not* reducible. Thus, the rounds and the chain cannot
build them, and cuhc does not supply them.

`algorithm="auto"` selects `"mst"` for single linkage, `"rnn"` for Ward and `"dense"` for
complete and average linkage.

**Scale** (RTX 3080, 32 features, random normal data, which does not prune): at 1,000,000
points, single linkage takes approximately 33 s, and Ward approximately 52 s. Both use
almost no memory. With clustered data, spatial pruning makes them much faster
(`docs/all_pairs_passes.md`). With billions of points, one all-pairs pass is not
possible. That requires an approximate nearest-neighbour search (section 9).

---

## 4. HDBSCAN with the single-linkage code

HDBSCAN is single linkage on a changed distance, with a different extraction step:

```
core(x)         = distance from x to its min_samples-th nearest point (x itself is the first)
mreach(a, b)    = max( core(a), core(b), d(a, b) / alpha )
HDBSCAN tree    = single-linkage tree of mreach
HDBSCAN labels  = condense the tree with min_cluster_size, select by stability
```

Thus:

1. **The core distances** come from one k-nearest-neighbour pass with the metric of the
   user. A fused kernel keeps the k nearest points of each row in shared memory (k ≤ 64).
   For larger k, distance blocks and an exact radix select give the result.
2. **Mutual reachability** is a transform. It is an edge weight that the Borůvka kernel
   applies, with the core distances of the two points.
3. **The tree** comes from the same Borůvka MST as single linkage.
4. **The labels** come from the HDBSCAN extraction.

Only step 4 is specific to HDBSCAN, and it is cheap CPU code on the tree. HDBSCAN also
receives all metrics and the scale of single linkage.

**Ties.** Mutual reachability makes many edge weights that are exactly equal. The HDBSCAN
labels can depend on the sequence of the tied merges. This is also true for the
scikit-learn labels. Thus, cuhc gives exactly the scikit-learn labels when there are no
ties, and exactly the scikit-learn tree except for the tie sequence. Refer to
`docs/hdbscan.md` section 6.

---

## 5. Python API

The API has three levels, from the scipy functions to the scikit-learn estimators:

```python
import cuhc

# distances
D = cuhc.pairwise_distances(X, Y=None, metric="euclidean")

# scipy API: build the tree, then cut it (defaults: single, euclidean)
Z = cuhc.linkage(X, method="average", metric="manhattan")
labels = cuhc.fcluster(Z, t=10, criterion="maxclust")

# scikit-learn estimators
model = cuhc.AgglomerativeClustering(n_clusters=10, linkage="ward").fit(X)
model.labels_, model.children_, model.distances_

hdb = cuhc.HDBSCAN(min_cluster_size=50, min_samples=10, metric="cosine").fit(X)
hdb.labels_, hdb.probabilities_
```

Compatibility rules:

- **The same names, meanings and defaults** as `sklearn.cluster.AgglomerativeClustering`,
  `sklearn.cluster.HDBSCAN`, `sklearn.metrics.pairwise_distances` and
  `scipy.cluster.hierarchy.linkage`. The tests examine the defaults.
- **The estimators are subclasses of `BaseEstimator` and `ClusterMixin`.** Thus, `clone`,
  `get_params`, `Pipeline`, `GridSearchCV` and `fit_predict` operate.
- **Additional options are keyword-only.** `algorithm=` ("auto", "dense", "mst", "rnn",
  "nn_chain") is on `linkage` and `AgglomerativeClustering`.
- **Some scikit-learn parameters are not available,** for example `connectivity` and
  `memory`. The README lists them.
- **Input:** any data that NumPy can convert, in any memory layout and data type. cuhc
  calculates in float32.

---

## 6. Code layout

```
src/                         C++ and CUDA core, header-only templates
  distances.cuh              metric structs and metric traits
  distance_matrix.cuh        tiled distance loop and the kernels that write or reduce its results
  reductions.cuh             (distance, index) argmin with warp shuffles and atomic operations
  spatial.cuh                k-d order, tile boxes, pruned walk kernel for the all-pairs passes
  mst.cuh                    Borůvka MST, and MST -> single-linkage tree
  hdbscan.cuh                core distances (fused top-k, radix select), mutual reachability,
                             tree condensing
  rnn_ward.cuh               Ward in rounds of reciprocal nearest neighbours
  nn_chain.cuh               nearest-neighbour chain for Ward (one cooperative kernel)
  linkage.cuh                dense matrix with Lance-Williams and a nearest-neighbour cache
  capi.cu                    extern "C" functions for Python
python/cuhc/                 Python package
  __init__.py                public API
  _lib.py                    loads libcuhc.so and declares the C functions
  _distances.py              pairwise_distances, input validation
  _linkage.py                linkage, fcluster, tree cut (NumPy)
  _estimators.py             AgglomerativeClustering
  _hdbscan.py                HDBSCAN: condensing, stability, selection (NumPy), estimator
bench/                       benchmarks (C++ kernels, Python compared with scikit-learn)
tests/                       pytest, compared with scikit-learn and scipy
docs/                        design documents
```

**The binding is `ctypes`.** CMake builds `libcuhc.so` directly into `python/cuhc/`, and
`_lib.py` declares the `extern "C"` functions of `src/capi.cu`. This needs only NumPy, and
the package operates from the source tree without installation.

The limit of this binding: each call copies the input to the GPU and the results back. A
nanobind binding (section 9) would let arrays stay on the GPU between calls. Only `_lib.py`
and `capi.cu` would change. The other Python code calls the same functions.

**Errors:** the C functions return 0, a `cudaError_t`, or a negative library error code.
Python changes codes that are not zero into a `RuntimeError` with `cuhc_error_string`.
Python examines the arguments first (shapes, metric names, parameter values, the float32
range of the distances), and gives a `ValueError`, the same as scikit-learn.

---

## 7. Design rules

1. **No silent approximation.** All algorithms are exact. If the dense matrix does not fit,
   cuhc gives a `MemoryError`. For Ward linkage, the message tells you to use
   `algorithm="rnn"`.
2. **Examine combinations before the calculation.** Ward linkage requires the Euclidean
   metric (the same as scikit-learn). All other linkages operate with all metrics.
3. **One metric definition for all algorithms.** The algorithms are templates of the
   metric. Nothing is specific to the Euclidean metric, except optional fast paths that
   must give the same results.
4. **The same results as scikit-learn where scikit-learn defines the behaviour,** also for
   special cases. For example, the cosine distance to a zero vector is 1, and the HDBSCAN
   `min_samples` includes the point itself.
5. **Keep manual options** (`algorithm=`) for users who know their data better than
   "auto".

---

## 8. Tests

To run the tests, build the library (`cmake -S . -B build && cmake --build build -j`).
Then use `python3 -m pytest tests`.

### 8.1 scikit-learn is the reference

The tests compare the cuhc output with scikit-learn or scipy on the same input:

| Test file | Reference | What the test compares |
|---|---|---|
| `test_pairwise_distances.py` | `sklearn.metrics.pairwise_distances` | values, for all metrics, at sizes on and near the tile edges |
| `test_linkage.py` | full `AgglomerativeClustering` tree (`children_`, `distances_`) | merge sequence, merge heights, cluster sizes, the float32 range check |
| `test_agglomerative.py` | `sklearn.cluster.AgglomerativeClustering` | labels, `n_clusters_`, `children_`, `distances_`, defaults, errors, estimator behaviour |
| `test_fcluster.py` | `scipy.cluster.hierarchy.fcluster` | flat clusters from the same tree |
| `test_mst.py` | scikit-learn and the dense algorithm of cuhc | single linkage with the MST, also with many ties |
| `test_rnn_ward.py` | scikit-learn, the dense algorithm and the chain | Ward with the rounds, each fallback, the worst case, the same result on each run |
| `test_nn_chain.py` | scikit-learn and the dense algorithm | Ward with the chain, also with duplicate points |
| `test_spatial_pruning.py` | cuhc without pruning | single linkage, Ward and the HDBSCAN tree with pruning: identical, or the same tree except for the tie sequence |
| `test_hdbscan.py` | `sklearn.cluster.HDBSCAN` and the scikit-learn extraction code | the tree except for the tie sequence (cophenetic distances), the extraction exactly, labels exactly without ties; all options |

For linkages other than Ward, and for HDBSCAN, the reference receives a float64
*precomputed* distance matrix. Thus, the tests can examine all cuhc metrics, also metrics
that the scikit-learn clustering classes do not accept by name (for example chebyshev).

To run all tests with spatial pruning always on (or always off), set
`CUHC_TEST_PRUNING=always` (or `never`).

### 8.2 How the tests compare results

- **Labels are compared as partitions** (`_helpers.assert_same_partition`). The clusters
  must be identical, but the label numbers can be different. Noise (−1) must be identical.
- **Floating-point values have tolerances,** because cuhc calculates in float32 and the
  reference in float64: `rtol = atol = 1e-4` for distances and merge heights. HDBSCAN
  probabilities use `atol = 1e-3` (section 8.3).
- **The test data does not have ties.** With equal distances, two correct algorithms can
  merge in different sequences. Thus, the test data comes from continuous distributions
  (`_helpers.blobs`). Distance thresholds are halfway between two merge heights. Thus,
  rounding cannot move a merge across the cut.
- **Large trees are compared as sets of clusters.** With thousands of merges, float64
  heights that are almost equal can become the same float32 value. These ties change the
  merge sequence and the cluster numbers. `_helpers.assert_same_dendrogram` compares trees
  by their clusters, each identified by (smallest point, size), and by their heights.
  Thus, it accepts any sequence of tied merges, and nothing else (refer to
  `docs/boruvka_mst.md` section 8).
- **One test is skipped.** In that case, scikit-learn itself stops with an error (an error
  of scikit-learn 1.9 with NumPy 2.5 in its `cluster_selection_epsilon` search, refer to
  `docs/hdbscan.md` section 6).

### 8.3 Known precision limit: cosine

The cosine formula `1 − dot / sqrt(|a|²·|b|²)` subtracts two almost equal numbers when
points are near each other. Thus, small cosine distances have only approximately 1e-4
relative accuracy in float32. This causes differences of up to approximately 4e-4 in the
HDBSCAN probabilities. The labels do not change.

A more accurate method is to normalize the vectors first. Then `1 − cos = |â − b̂|² / 2`,
which has no cancellation.

---

## 9. Planned work

- **Metrics from Python.** The user gives the source of `accumulate` (and optionally
  `init`, `finalize` and `Acc`) as strings. cuhc puts them into the same kernel templates,
  compiles them at run time with NVRTC, and keeps them for subsequent calls. After the
  compilation, they have the same speed as the built-in metrics:

  ```python
  canberra = cuhc.Metric(
      "canberra",
      accumulate="float s = fabsf(a) + fabsf(b); return s > 0.f ? acc + fabsf(a - b) / s : acc;",
  )
  ```

- **A nanobind binding, and GPU input without a copy.** CuPy and PyTorch arrays that are
  already on the GPU, through `__cuda_array_interface__` or DLPack.
- **An optional approximate mode** (`approximate=True`) for data that is too large for one
  all-pairs pass. It would build a sparse k-nearest-neighbour graph, and run the builders
  on its edges. The result is exact only if the graph contains all edges that the exact
  algorithm uses. Thus, cuhc would use it only on request.
- **Faster distance kernels:** a tensor-core path for the Euclidean and cosine metrics
  (`|a|² + |b|² − 2a·b`), and `cp.async` double buffering (refer to
  `docs/distance_matrix.md` section 8).
- **Cosine precision:** normalized vectors (section 8.3).
- **The dense merge loop in a CUDA graph,** to decrease the cost of the kernel starts
  (`docs/dense_linkage.md` section 9).
