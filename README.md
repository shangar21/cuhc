# cuhc: hierarchical clustering on the GPU

`cuhc` is a CUDA library for hierarchical clustering and density-based clustering. Its
Python API is the same as the scikit-learn API and the scipy API.

| cuhc | Replaces |
|---|---|
| `cuhc.AgglomerativeClustering` | `sklearn.cluster.AgglomerativeClustering` |
| `cuhc.HDBSCAN` | `sklearn.cluster.HDBSCAN` |
| `cuhc.linkage` | `scipy.cluster.hierarchy.linkage` |
| `cuhc.fcluster` | `scipy.cluster.hierarchy.fcluster` |
| `cuhc.pairwise_distances` | `sklearn.metrics.pairwise_distances` |

All results are exact. cuhc builds the same tree as scikit-learn and scipy. It does not
use approximations.

Most methods do not keep the n × n distance matrix in memory. Thus, one consumer GPU can
cluster millions of points.

```python
import numpy as np
import cuhc

X = np.random.default_rng(0).normal(size=(100_000, 32)).astype(np.float32)

labels = cuhc.AgglomerativeClustering(n_clusters=10).fit_predict(X)  # Ward linkage
labels = cuhc.HDBSCAN(min_cluster_size=25).fit_predict(X)            # density-based
Z = cuhc.linkage(X, method="ward")                                   # scipy tree
```

Documentation: https://cuhc.readthedocs.io

## Contents

- [Performance](#performance)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Example: customer segments](#example-customer-segments)
- [Concepts](#concepts)
- [API reference](#api-reference)
  - [`linkage`](#cuhclinkage)
  - [`fcluster`](#cuhcfcluster)
  - [`AgglomerativeClustering`](#cuhcagglomerativeclustering)
  - [`HDBSCAN`](#cuhchdbscan)
  - [`pairwise_distances`](#cuhcpairwise_distances)
  - [Metrics](#metrics)
  - [Algorithms](#algorithms)
- [How cuhc works](#how-cuhc-works)
- [Precision, limits and errors](#precision-limits-and-errors)
- [Differences from scikit-learn and scipy](#differences-from-scikit-learn-and-scipy)
- [Development](#development)
- [License](#license)

---

## Performance

The table shows times on an RTX 3080 GPU. Each run builds the full tree. The linkage
data is random normal data with 32 features. The HDBSCAN data has 20 clusters, 10% noise
and 8 features. The scikit-learn column shows scikit-learn 1.9 on the CPU.

| Task | Points | cuhc | scikit-learn | Speed increase |
|---|---|---|---|---|
| Ward linkage | 20,000 | 0.038 s | 24.0 s | approximately 630 times |
| Ward linkage | 100,000 | 0.53 s | — | |
| Ward linkage | 1,000,000 | 51 s | — | |
| Single linkage | 20,000 | 0.027 s | 6.7 s | approximately 250 times |
| Single linkage | 1,000,000 | 34 s | — | |
| HDBSCAN | 100,000 | 0.26 s | 105 s | approximately 400 times |
| HDBSCAN | 1,000,000 | 8.9 s | — | |
| Complete or average linkage | 20,000 | 0.41 s | 22 s to 28 s | 55 to 68 times |

Clustered data with few features is faster again, because of spatial pruning. For
example, single linkage on 1,000,000 points in 20 clusters with 2 features takes 2.4 s.
The scikit-learn times change from run to run by up to approximately 40%.

A dash (—) shows that scikit-learn cannot do the task in practice. It must keep the
n × n matrix in RAM (4 TB for 1,000,000 points), or it uses many hours.

---

## Installation

You must have:

- An NVIDIA GPU and its driver.
- The CUDA toolkit (`nvcc`, version 13.2 was tested) and CMake 3.24 or higher.
- Python 3.9 or higher.

There are no prebuilt wheels at this time. pip builds the CUDA library from the source.
This takes approximately two minutes.

1. Clone the repository:

   ```bash
   git clone https://github.com/shangar21/cuhc.git
   cd cuhc
   ```

2. Install the package. pip also installs NumPy and scikit-learn:

   ```bash
   pip install .
   ```

3. Do a test of the installation:

   ```bash
   python -c "import cuhc; print(cuhc.linkage([[0.0], [1.0]]))"
   ```

By default, the build makes code for the GPUs in the computer that builds it. To build
for other GPUs, set their compute capabilities. For example, use `80` for an A100, `86`
for the RTX 30 series and `89` for the RTX 40 series:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80;86;89" pip install .
```

To install the test dependencies (pytest and scipy) also, use `pip install ".[test]"`.

---

## Quick start

### Agglomerative clustering with the scikit-learn API

```python
import cuhc
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=50_000, centers=8, n_features=16, random_state=0)

model = cuhc.AgglomerativeClustering(n_clusters=8, linkage="ward").fit(X)
model.labels_      # The cluster of each point, 0 to 7
model.children_    # The merge tree, the same as in scikit-learn
```

To cut the tree at a distance instead of at a number of clusters:

```python
model = cuhc.AgglomerativeClustering(n_clusters=None, distance_threshold=50.0).fit(X)
model.n_clusters_  # The number of clusters at this distance
model.distances_   # The height of each merge
```

### The merge tree with the scipy API

Build the tree one time. Then cut it as many times as necessary. A cut is fast and
runs on the CPU.

Average linkage keeps the n × n matrix in GPU memory. Thus, this example uses 20,000 of
the points. Refer to [Algorithms](#algorithms).

```python
Z = cuhc.linkage(X[:20_000], method="average", metric="cosine")
labels_5 = cuhc.fcluster(Z, 5, criterion="maxclust")      # 5 clusters or fewer, labels 1 to 5
labels_t = cuhc.fcluster(Z, 0.3, criterion="distance")    # merges at height 0.3 or less

# Z is a standard scipy linkage matrix. The scipy tools also use it:
from scipy.cluster.hierarchy import dendrogram
dendrogram(Z, truncate_mode="lastp", p=30, no_plot=True)
```

### Density-based clustering

```python
model = cuhc.HDBSCAN(min_cluster_size=25).fit(X)
model.labels_         # -1 shows noise
model.probabilities_  # How strongly each point is in its cluster (0 for noise)
```

---

## Example: customer segments

This example uses the Kaggle "Mall Customers" data set
(`vjchoudhary7/customer-segmentation-tutorial-in-python`). It has 200 customers, each with
an annual income and a spending score. The example finds groups of customers with Ward
linkage and with HDBSCAN, and compares the results with scikit-learn.

Download `Mall_Customers.csv` from Kaggle into `~/Downloads`. Then:

```python
import csv
import pathlib

import numpy as np
import cuhc

path = pathlib.Path.home() / "Downloads" / "Mall_Customers.csv"
with open(path, newline="") as f:
    X = np.array([[float(row["Annual Income (k$)"]), float(row["Spending Score (1-100)"])]
                  for row in csv.DictReader(f)])

Z = cuhc.linkage(X, method="ward")                                # the tree for a dendrogram
segments = cuhc.AgglomerativeClustering(n_clusters=5).fit_predict(X)
dense = cuhc.HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(X)

print(np.bincount(segments))            # customers in each Ward segment
print(np.sum(dense == -1), "noise points")
```

```
[21 23 85 39 32]
33 noise points
```

The full script, `examples/customer_segments/cluster_customers.py`, draws this figure. The
top row shows cuhc and the bottom row shows scikit-learn. The Ward segments are the same
in both rows. The HDBSCAN result is different for one customer. The data has only
integer values, so many distances are equal, and equal distances can merge in a
different sequence.

![Ward dendrogram, Ward segments and HDBSCAN clusters of the mall customers, for cuhc (top) and scikit-learn (bottom)](https://raw.githubusercontent.com/shangar21/cuhc/main/docs/assets/customer_segments.png)

The [tutorial](https://cuhc.readthedocs.io/en/latest/tutorial_customer_segments/) on the
documentation site explains each step.

---

## Concepts

### Hierarchical clustering

At the start, each point is a cluster. The algorithm finds the two nearest clusters and
merges them. It continues until there is only one cluster.

The result is a merge tree (a dendrogram). To get flat clusters, you cut the tree. You
can cut it at a height or at a number of clusters.

### Linkage

The linkage is the rule that gives the distance between two clusters:

| Linkage | Distance between clusters A and B | Result |
|---|---|---|
| `single` | The distance of the nearest pair of points, one point from A and one point from B | Long chains. Finds clusters of any shape. |
| `complete` | The distance of the farthest pair of points | Compact clusters with a similar diameter. |
| `average` | The mean distance of all the pairs of points | A result between `single` and `complete`. |
| `ward` | The increase in the total variance in the clusters if A and B merge | Compact clusters of similar size. This is the usual default. |

### HDBSCAN

HDBSCAN is hierarchical clustering on a distance that includes the density of the data.

1. For each point, HDBSCAN calculates the core distance. The core distance is the
   distance to the `min_samples`-th nearest point. In dense areas the core distance is
   small. In sparse areas it is large.
2. The distance between two points a and b changes to
   `max(core(a), core(b), d(a, b))`. This is the mutual reachability distance.
3. HDBSCAN builds the single-linkage tree of the mutual reachability distance.
4. HDBSCAN cuts the tree at lower and lower heights. It keeps the clusters that stay for
   the longest range of heights (the most stable clusters).
5. Points that are not in a kept cluster are noise. Their label is −1.

You do not set the number of clusters. You set `min_cluster_size`, which is the smallest
group of points that is a cluster.

### The linkage matrix `Z`

`Z` is the scipy format of a merge tree. It is an array of float64 values with n − 1 rows
and 4 columns. Each row is one merge. The rows are in merge order:

```
Z[i] = [cluster_a, cluster_b, height, number_of_points_in_the_new_cluster]
```

Clusters 0 to n − 1 are the points. Row i makes the cluster with the number `n + i`.

---

## API reference

### `cuhc.linkage`

```python
cuhc.linkage(X, method="single", metric="euclidean", p=2.0, *, algorithm="auto") -> Z
```

This function builds the full merge tree of the rows of `X`. It is the same as
`scipy.cluster.hierarchy.linkage`. The default `method="single"` is also the scipy
default.

| Parameter | Type | Description |
|---|---|---|
| `X` | Array, shape (n_samples, n_features) | The points. cuhc changes all data types to float32. All values must be finite. |
| `method` | `"single"`, `"complete"`, `"average"`, `"ward"` | The linkage. Refer to [Linkage](#linkage). `"ward"` is possible only with `metric="euclidean"`. |
| `metric` | String | One of the [metrics](#metrics). |
| `p` | Float, more than 0 | The Minkowski order. cuhc uses it only with `metric="minkowski"`. |
| `algorithm` | `"auto"`, `"dense"`, `"mst"`, `"rnn"`, `"nn_chain"` | The algorithm that builds the tree. Refer to [Algorithms](#algorithms). All algorithms give the same exact tree. They have a different speed and use a different quantity of memory. `"auto"` selects the best algorithm. |

**Return value:** `Z`, the merge tree in the scipy format. Refer to
[The linkage matrix `Z`](#the-linkage-matrix-z). The rows are in the sequence of their
height.

**Errors:**

- `ValueError` occurs if:
  - The method, the metric or the algorithm is not known.
  - The algorithm cannot build the linkage.
  - The method is `ward` and the metric is not `euclidean`.
  - The input contains NaN or infinite values.
  - The distances in the data are too large for float32.
- `MemoryError` occurs if the algorithm is `"dense"` and the n × n matrix is too large for
  the GPU memory. Complete linkage and average linkage always use `"dense"`.
- `RuntimeError` occurs if CUDA has an error.

**Examples:**

```python
Z = cuhc.linkage(X, method="ward")                            # rnn rounds, no matrix
Z = cuhc.linkage(X, method="single", metric="manhattan")      # minimum spanning tree
Z = cuhc.linkage(X, method="average", metric="minkowski", p=3)
```

### `cuhc.fcluster`

```python
cuhc.fcluster(Z, t, criterion="maxclust") -> labels
```

This function cuts a merge tree into flat clusters. It is the same as
`scipy.cluster.hierarchy.fcluster` for the two criteria below.

| Parameter | Description |
|---|---|
| `Z` | A merge tree from `cuhc.linkage` or from scipy. |
| `t` | The number of clusters (`"maxclust"`) or the height (`"distance"`). |
| `criterion="maxclust"` | Applies the merges in sequence until `t` clusters are left. The result has `t` clusters or fewer. |
| `criterion="distance"` | Applies all the merges with a height of `t` or less. |

**Return value:** an array of integers with one label for each point. The labels start
at 1, as in scipy.

### `cuhc.AgglomerativeClustering`

```python
cuhc.AgglomerativeClustering(n_clusters=2, *, metric="euclidean", linkage="ward",
                             distance_threshold=None, compute_distances=False,
                             algorithm="auto")
```

This class is a scikit-learn estimator. It has `fit`, `fit_predict`, `get_params` and
`set_params`. It operates with `clone`, `Pipeline` and `GridSearchCV`. Its parameters,
defaults and attributes are the same as in `sklearn.cluster.AgglomerativeClustering`.

| Parameter | Description |
|---|---|
| `n_clusters` | The number of clusters. If you set `distance_threshold`, set `n_clusters=None`. |
| `metric` | One of the [metrics](#metrics). Minkowski always uses p = 2 in this class. `ward` linkage is possible only with `"euclidean"`. |
| `linkage` | `"ward"` (default), `"complete"`, `"average"` or `"single"`. |
| `distance_threshold` | Use this parameter instead of `n_clusters`. cuhc does not apply merges at this height or higher. Set `n_clusters=None` with it. |
| `compute_distances` | If `True`, cuhc keeps the merge heights in `distances_`. cuhc always keeps them if you set `distance_threshold`. |
| `algorithm` | This parameter is not in scikit-learn. It is the same as in [`linkage`](#cuhclinkage). |

| Attribute | Description |
|---|---|
| `labels_` | The cluster of each point, from 0 to `n_clusters_` − 1. |
| `n_clusters_` | The number of clusters. |
| `children_` | An array with n − 1 rows and 2 columns. Each row shows the two clusters of one merge. The cluster numbers are the same as in `Z`. |
| `distances_` | The height of each merge. cuhc sets this attribute only if `compute_distances` is `True` or `distance_threshold` is set. |
| `n_leaves_`, `n_features_in_`, `n_connected_components_` | The same as in scikit-learn. |

cuhc does not support these scikit-learn parameters: `connectivity`, `memory` and
`compute_full_tree`. cuhc always builds the full tree, because this is fast.

### `cuhc.HDBSCAN`

```python
cuhc.HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0,
             max_cluster_size=None, metric="euclidean", alpha=1.0,
             cluster_selection_method="eom", allow_single_cluster=False)
```

This class is a scikit-learn estimator. It is the same as `sklearn.cluster.HDBSCAN`. It
does not keep the n × n distance matrix in memory.

| Parameter | Description |
|---|---|
| `min_cluster_size` | The smallest group of points that is a cluster. The value must be 2 or more. This is the most important parameter. |
| `min_samples` | The number of points for the core distance. This number includes the point itself. A larger value gives more noise points. The default is `min_cluster_size`. With a value of 64 or less, cuhc uses its fastest kernel. |
| `cluster_selection_epsilon` | cuhc merges clusters that are nearer than this distance. This prevents many small clusters in dense areas. |
| `max_cluster_size` | cuhc does not select a cluster that has more points than this value (with `"eom"` only). |
| `metric` | One of the [metrics](#metrics). Minkowski always uses p = 2 in this class. |
| `alpha` | A distance scale. The mutual reachability distance is `max(core(a), core(b), d(a, b) / alpha)`. Usually you do not change it. |
| `cluster_selection_method` | `"eom"` (excess of mass, the default) selects the most stable clusters. `"leaf"` selects the smallest clusters at the bottom of the tree. `"leaf"` gives more clusters and smaller clusters. |
| `allow_single_cluster` | If `True`, the result can be one cluster. If `False`, cuhc does not select a single cluster. |

| Attribute | Description |
|---|---|
| `labels_` | The cluster of each point, from 0 to k − 1. The label −1 shows noise. |
| `probabilities_` | How strongly each point is in its cluster, from 0 to 1. The value is 0 for noise. |
| `core_distances_` | The core distance of each point. This attribute is not in scikit-learn. |
| `n_features_in_` | The number of features in the data that `fit` received. |

cuhc does not support these scikit-learn parameters: `leaf_size`, `n_jobs`,
`store_centers`, `copy`, `metric_params` and `metric="precomputed"`.

### `cuhc.pairwise_distances`

```python
cuhc.pairwise_distances(X, Y=None, metric="euclidean", p=2.0) -> D
```

This function calculates the distance between each row of `X` and each row of `Y`. If
`Y` is `None`, cuhc uses `X` for `Y`. It is the same as
`sklearn.metrics.pairwise_distances`.

**Return value:** an array of float32 values, with `len(X)` rows and `len(Y)` columns.

The full matrix must fit in the GPU memory and in the host memory. Its size is
4 · len(X) · len(Y) bytes.

### Metrics

| Name (other names) | Distance |
|---|---|
| `"euclidean"` (`"l2"`) | √Σ(xᵢ − yᵢ)² |
| `"sqeuclidean"` | Σ(xᵢ − yᵢ)² |
| `"manhattan"` (`"cityblock"`, `"l1"`) | Σ\|xᵢ − yᵢ\| |
| `"chebyshev"` | max \|xᵢ − yᵢ\| |
| `"minkowski"` | (Σ\|xᵢ − yᵢ\|ᵖ)^(1/p), with the parameter `p` |
| `"cosine"` | 1 − x·y / (‖x‖‖y‖). The value is 1 for a zero vector, as in scikit-learn. |

`cuhc.METRIC_IDS` and `cuhc.LINKAGE_IDS` show all the names that cuhc accepts.

### Algorithms

The `algorithm` parameter of `linkage` and `AgglomerativeClustering` selects how cuhc
builds the tree. All algorithms build the same exact tree. Only the sequence of merges
with equal heights can be different.

| Algorithm | Linkages | GPU memory | Use |
|---|---|---|---|
| `"mst"` | single | O(n) | `"auto"` uses it for single linkage. |
| `"rnn"` | ward | O(n · features) | `"auto"` uses it for Ward linkage. |
| `"dense"` | all | 4 · n² bytes (approximately 50,000 points on a 12 GB GPU) | `"auto"` uses it for complete and average linkage. |
| `"nn_chain"` | ward | O(n · features) | This algorithm is slower than `"rnn"`. It is available for comparison. |

Complete linkage and average linkage must have the dense matrix. There is no known exact
algorithm for them without the matrix. Thus, the maximum number of points is
approximately √(0.9 · free GPU bytes / 4).

---

## How cuhc works

The files in `docs/` give a full explanation of each method. This section gives a short
explanation.

### The distance kernel

Refer to `docs/distance_matrix.md`.

The distance kernel has the same structure as a matrix multiplication. Each block of GPU
threads loads 128 points from each side into fast shared memory. Then it calculates a
tile of 128 × 128 distances. Thus, the kernel uses each loaded value 128 times.

The metric is a small C++ struct with three functions: `init`, `accumulate` and
`finalize`. The compiler puts the metric into the kernel. Thus, a new metric has the same
speed as a built-in metric. At high dimensions, the kernel has approximately 95% of the
instruction rate of cuBLAS.

Most algorithms do not keep the tiles. The kernel reduces each tile in registers to the
nearest point for each row. Then it discards the tile. Thus, these algorithms do not
use n² memory.

### Single linkage: minimum spanning tree

Refer to `docs/boruvka_mst.md`.

The single-linkage tree is the minimum spanning tree of the points. cuhc merges the edges
of the tree in the sequence of their length, shortest first.

The Borůvka algorithm builds the minimum spanning tree in approximately log₂ n rounds. In
each round:

1. Each point finds its nearest point in a different component. One GPU pass does this
   for all the points.
2. Each component selects its shortest edge.
3. The components that these edges connect merge.

### Ward linkage: reciprocal-nearest-neighbour rounds

Refer to `docs/rnn_ward.md`.

The Ward distance between two clusters uses only their sizes and centroids:

```
ward(A, B) = sqrt(2·|A|·|B| / (|A| + |B|)) · ‖centroid(A) − centroid(B)‖
```

Thus, cuhc does not use a matrix of point distances. In each round:

1. Each cluster finds its nearest cluster. One all-pairs pass over the centroids does
   this.
2. Each pair of clusters that are the nearest cluster of each other merges. These pairs
   are "reciprocal nearest neighbours". All the pairs merge at the same time.
3. cuhc moves the remaining clusters together in memory. Thus, the subsequent pass is
   smaller.

This gives the same tree as the usual method, which merges the nearest pair one pair at a
time. The reason is that Ward linkage is "reducible". When two clusters merge, the new
cluster is not nearer to a third cluster than its two parts were. Thus, a merge cannot
come between a reciprocal pair.

With usual data, 5% to 60% of the clusters merge in each round. Thus, the work decreases
quickly. With some rare data, the rounds merge very few clusters. For this data, cuhc
gives the remaining clusters to the dense algorithm or to the nearest-neighbour chain.

### Faster all-pairs passes

Refer to `docs/all_pairs_passes.md`.

Single linkage, Ward linkage and HDBSCAN all use passes over all pairs of points. Two
methods make these passes faster. Both keep the result exact.

- **Symmetry.** For all built-in metrics, d(x, y) is equal to d(y, x). Thus, the pass
  calculates each pair of points one time only. This makes each pass up to 2 times
  faster.
- **Spatial pruning.** cuhc sorts the points so that each tile of 128 points is a
  compact area, and calculates a bounding box for each tile. If no point in a tile can be
  the nearest point of a row, the pass does not calculate the tile. This is very
  effective for clustered data and for data with few features. For example, single
  linkage on 1,000,000 points in 20 clusters with 2 features is approximately 12 times
  faster. Random data with many features does not prune. A small test before the passes
  finds this, and cuhc then does not prune.

### HDBSCAN

Refer to `docs/hdbscan.md`.

HDBSCAN has three stages:

1. Core distances. Each GPU block keeps a sorted list of the k nearest points for its
   rows in shared memory. The block reads all the points one time.
2. Tree. cuhc uses the same Borůvka algorithm as for single linkage. The edge weight is
   the mutual reachability distance, `max(core(a), core(b), d(a, b) / alpha)`.
3. Cluster selection. This is a copy of the scikit-learn code. It condenses the tree,
   calculates the stability of each cluster, selects the most stable clusters and gives
   a label to each point.

### Complete and average linkage

Refer to `docs/dense_linkage.md`.

These linkages use the n × n matrix. cuhc keeps the nearest neighbour of each row. After
each merge, the Lance–Williams formula gives the distances of the new cluster. cuhc scans
again only the rows whose nearest neighbour was in the merge. Each merge uses three
kernels. The CPU does not stop the GPU between the merges.

---

## Precision, limits and errors

- **float32 on the GPU.** The GPU calculates distances in float32. The CPU builds the
  tree in float64. With usual data, the merge heights are the same as in scipy to
  approximately 1e-7 (relative). The labels are the same. Merges with equal heights can
  occur in a different sequence. The scikit-learn results also change with this
  sequence.
- **Unstable data.** Ward linkage is very sensitive on some data, for example thousands
  of points on a line with almost equal gaps. On this data, a change of 1e-7 in the input
  changes the scipy result. No float32 code can give the same tree as scipy on this data.
- **Range.** If the distances are too large for float32, cuhc gives a `ValueError`. For
  the Euclidean metric, the limit is a squared spread of approximately 3.4e38. If this
  occurs, scale the data to a smaller range. Data far from the origin is satisfactory.
  Only the spread of the data is important.
- **Input.** If the input contains NaN or infinite values, cuhc gives a `ValueError`.
  scikit-learn does the same.
- **GPU memory.** Only `"dense"` and `pairwise_distances` use memory that increases with
  n². Complete linkage and average linkage use `"dense"`. The other methods use
  approximately n · features · 4 bytes, plus small arrays for each point.
- **Repeatability.** On the same GPU, each run gives the same result.

---

## Differences from scikit-learn and scipy

- cuhc calculates in float32. The output arrays (`Z` and `distances_`) are float64, as in
  scipy.
- `AgglomerativeClustering` does not have `connectivity`, `memory` or
  `compute_full_tree`.
- `HDBSCAN` does not have `leaf_size`, `n_jobs`, `store_centers`, `metric_params` or
  precomputed distances.
- `fcluster` has only the criteria `"maxclust"` and `"distance"`.
- You can set the Minkowski `p` in `linkage` and `pairwise_distances`. You cannot set it
  in the estimators at this time.
- cuhc adds `algorithm` to `linkage` and `AgglomerativeClustering`. It adds
  `core_distances_` to `HDBSCAN`.

---

## Development

For development, build the library in the source tree. The build writes
`python/cuhc/libcuhc.so`, and the tests use the package in `python/` directly:

```bash
cmake -S . -B build && cmake --build build -j
python -m pytest tests -q          # compares the results with scikit-learn and scipy
python bench/bench_linkage.py --methods ward single --sizes 20000 100000
python bench/bench_hdbscan.py --sizes 100000 --sklearn-max 100000
./build/bench_distance 20000 64 euclidean    # compares the distance kernel with cuBLAS
```

To build the documentation site and look at it in a browser (at
http://127.0.0.1:8000):

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

The site uses MkDocs. The API reference comes from the docstrings in the source files.
The build does not import cuhc, so it does not need a GPU. `.readthedocs.yaml` lets
Read the Docs build and host the site from the repository.

The repository has these directories:

```
src/            The CUDA and C++ core (header-only templates) and the C API (capi.cu)
python/cuhc/    The Python package. It loads libcuhc.so with ctypes.
tests/          The pytest tests. Each test compares cuhc with scikit-learn or scipy.
bench/          The benchmarks.
docs/           The design notes. They explain how each algorithm operates and why.
```

The planned work is in `docs/architecture.md` section 9. It includes these items:

- User-defined metrics from Python.
- Input from CuPy and PyTorch without a copy.
- An optional approximate mode for data that is too large for the GPU memory.

---

## License

cuhc has the MIT license. Refer to the `LICENSE` file.
