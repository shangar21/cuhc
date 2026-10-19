# HDBSCAN

This document explains how cuhc calculates HDBSCAN. The code is in these files:

- `src/hdbscan.cuh`: the core distances, the mutual reachability distance and the tree
  condensing step.
- `src/mst.cuh` and `src/distance_matrix.cuh`: the minimum spanning tree with an edge
  weight.
- `python/cuhc/_hdbscan.py`: the cluster extraction and the `HDBSCAN` estimator.

Contents:

1. [What HDBSCAN calculates](#1-what-hdbscan-calculates)
2. [The stages and where they operate](#2-the-stages-and-where-they-operate)
3. [Core distances](#3-core-distances)
4. [Mutual reachability as an edge weight](#4-mutual-reachability-as-an-edge-weight)
5. [Extraction: condense, stability, selection](#5-extraction-condense-stability-selection)
6. [Ties: why the labels are not always equal to the scikit-learn labels](#6-ties-why-the-labels-are-not-always-equal-to-the-scikit-learn-labels)
7. [What the tests examine](#7-what-the-tests-examine)
8. [Results](#8-results)
9. [Limits](#9-limits)

---

## 1. What HDBSCAN calculates

HDBSCAN finds clusters as dense areas. It gives the label "noise" to points in sparse
areas.

1. **Core distance:** `core(x)` is the distance from x to its `min_samples`-th nearest
   point. The point x itself is the first point. The core distance is small in dense
   areas and large in sparse areas.
2. **Mutual reachability:** `mreach(a, b) = max(core(a), core(b), d(a, b) / alpha)`. This
   moves sparse points away from all other points. Thus, sparse points cannot connect two
   clusters.
3. **Tree:** the single-linkage tree of mreach. This is the minimum spanning tree (MST) of
   mreach (refer to `docs/boruvka_mst.md`).
4. **Condensed tree:** go through the tree from the top. If both sides of a split have
   `min_cluster_size` points or more, the split is real. Smaller pieces are points that
   "fall out" of a cluster.
5. **Selection:** select the clusters that stay for the longest time (the most "stable"
   clusters). All other points are noise (−1).

Steps 1 to 3 cost O(n²·d), and they operate on the GPU. Steps 4 and 5 use only the tree
(size O(n)), and they operate on the CPU. The condensing step is in C++. The other steps
are in NumPy.

**Two conventions are the same as in scikit-learn:**

- `min_samples` includes the point itself (`min_samples=1` gives a core distance of 0).
  Its default is `min_cluster_size`.
- For `alpha`, scikit-learn has two code paths with different results:
  - Its default path for feature input (KD-tree and Prim) and the original `hdbscan`
    library use `max(core, core, d / alpha)`.
  - Its path for precomputed distances divides the full matrix first. This gives
    `max(core, core, d) / alpha`.

  cuhc uses the first formula, because `HDBSCAN(alpha=...).fit(X)` uses it for feature
  input. The two formulas give the same result if `alpha = 1`.

---

## 2. The stages and where they operate

```
X ──► core distances ──► Borůvka MST with mreach weights ──► sort edges ──► Z ──► condense ──► select ──► labels,
      (GPU, fused top-k)   (GPU pass + CPU union-find)         (C++)              (C++)        (NumPy)    probabilities
      hdbscan.cuh           mst.cuh + distance_matrix.cuh       mst.cuh            hdbscan.cuh  _hdbscan.py
```

The C function `cuhc_hdbscan_tree` returns `Z` and the core distances. `HDBSCAN.fit` does
the extraction on `Z`. The estimator keeps the tree as `_single_linkage_tree_` (scipy
format) and the core distances as `core_distances_`.

For data with a low intrinsic dimension, the core-distance kernel and the MST passes skip
tiles that are too far away. Refer to `docs/all_pairs_passes.md`.

---

## 3. Core distances

`coreDistances` in `src/hdbscan.cuh` has two exact methods. With `core_method="auto"` (the
default), cuhc uses the fused method if k = `min_samples` is 64 or less. This is almost
all real use.

### 3.1 Fused top-k (k ≤ 64)

The kernel is `kthNearestFusedKernel`. It writes only the n results to global memory.

- **Grid shape.** There is one block for each tile of 128 rows. The block **goes through
  all the column tiles**, and calls `computeTileDistances` for each of them. All the
  columns of a row go through one block. Thus, the running data of the row can stay in
  the shared memory of that block, without atomic operations between blocks.
- **Sequence of the column tiles.** The block starts with its own tile. Then it goes out
  by index: I, I+1, I−1, I+2, and so on. The near tiles come first, so the lists change
  rarely after the first tiles. For points in spatial order (`docs/all_pairs_passes.md`),
  the sequence 0, 1, 2, … is the worst case: each tile is nearer than the tile before it,
  and most distances change the lists.
- **Data for each row.** An ascending list of the k smallest distances, in shared memory:
  `TileRows × k` floats. The k-th entry is the current threshold of the row.
- **Which threads use a list.** Only the 16 threads that calculate that row. In the tile
  layout, they are 16 consecutive lanes of one warp (half of a warp). No other warp uses
  the list. Thus, `__syncwarp()` is sufficient to put their changes in sequence.
- **Insertion.** After each tile, each thread makes a bitmask. The bitmask shows which of
  its 8 distances for the row are smaller than the threshold. Then, until no lane of the
  warp has work:
  1. `__ballot_sync` finds the lanes that have candidates.
  2. The lowest of those lanes *in each half-warp* puts one value into the list of its
     row, and moves the larger entries down. This is O(k), done by one thread.

  The loop continues until the *full warp* has no work. Thus, all 32 lanes come to each
  `__ballot_sync` together, also when the two halves operate on different rows.
- **Why one insertion at a time is satisfactory.** A value goes into the list only if it
  is smaller than the current k-th value of the row. In the first tile, almost all values
  go in, but the threshold decreases quickly. For a row of n values in an effectively
  random sequence, the expected number of insertions is approximately k·(1 + ln(n/k)).
  For k = 25 and n = 1,000,000, this is approximately 290 insertions. The row also has
  1,000,000 distance calculations of `numFeatures` steps each.
- **Exact results.** The list contains exact float distances. Thus, the result is the
  same, bit for bit, as the radix select. The tests examine this.

The shared memory has 128 × k floats, plus the 8.4 KB of the tile loop. This must be less
than 48 KB. Thus, k is 64 or less.

### 3.2 Blocked rows and radix select (any k)

**Blocked rows.** The kernel calculates the distances for a block of rows at a time (up
to 512 MB, and one tile of 128 rows or more) into a buffer that it uses again. It uses
the same tiled kernel, with `pointsA` = the rows of the block and `pointsB` = all the
points. The full n² matrix is never in memory.

**Radix select** (`kthSmallestPerRowKernel`, one block for each row). Only the k-th
smallest value of each row is necessary, not a sorted row:

- Floats that are not negative have the same sequence as their 32-bit patterns, read as
  unsigned integers. Thus, the kernel can find the k-th smallest value **one byte at a
  time**, from the most significant byte.
- Pass 1: make a histogram with 256 bins of the top byte of each value. Go through the
  bins until the running count is k or more. That bin is the top byte of the answer. k
  becomes the rank *in* that bin.
- Passes 2 to 4: the same, but count only the values whose selected bytes agree.
- After 4 passes, all 32 bits are known. Thus, the answer is exact, with no tolerance and
  no sort.

The distance of the row to its own point is set to 0. This is the meaning of "the point
itself is the first point". It also removes float errors, for example 1e−7 for a cosine
distance d(x, x).

**Histogram increments for a full warp.** Near distances have the same high bytes. Thus,
in pass 1, most lanes of a warp use the same bin. Shared-memory atomic operations on one
address occur one after the other. `warpAggregatedIncrement` prevents this:

- `__match_any_sync(mask, bucket)` tells each lane which other lanes of the warp have the
  same bucket.
- The lowest of them does one `atomicAdd(bucket, popcount)`.

Thus, there is one atomic operation for each different bucket in a warp, not one for
each lane.

**Cost.** One all-pairs distance pass, plus a write of the blocks and 4 reads of them. At
500,000 points, this memory traffic makes it approximately 2.5 times slower than the
fused kernel. cuhc uses it for k > 64, or if `core_method="radix"`.

---

## 4. Mutual reachability as an edge weight

The Borůvka pass (`docs/boruvka_mst.md`) has an **edge-weight** parameter. The kernel
applies it to each distance before the minimum of each row. The edge weight has two
parts: data for each point, which the kernel loads one time for each row and column of a
thread, and the combination for each pair:

```cpp
struct DistanceWeight {                       // plain single linkage
  using Value = int;
  Value value(int) const { return 0; }
  float operator()(float d, Value, Value) const { return d; }
};
struct MutualReachabilityWeight {             // HDBSCAN
  const float *core; float alpha;
  using Value = float;
  Value value(int point) const { return core[point]; }
  float operator()(float d, Value coreI, Value coreJ) const {
    return fmaxf(fmaxf(coreI, coreJ), alpha == 1.0f ? d : d / alpha);
  }
};
```

The two parts are important for speed. If the kernel loads `core[i]` and `core[j]` for
each pair, the loads are a large part of the time. A division for each pair is also
expensive. Thus, the weight divides only if `alpha` is not 1.

The MST code, its rule for ties and its conversion to `Z` are the same for single linkage
and for HDBSCAN. Thus, HDBSCAN operates with all metrics.

---

## 5. Extraction: condense, stability, selection

`_hdbscan.py` is a port of `_tree.pyx` from scikit-learn 1.9. It stays close to the
original, also where small details change the results. Thus, for the same tree, it gives
**exactly** the scikit-learn labels and probabilities. The tests compare it with the
scikit-learn code.

The condensing step is in C++ (`hc::condenseTree` in `hdbscan.cuh`, called through
`cuhc_condense_tree`). At 300,000 points, a NumPy version used 4.0 s of the 4.2 s of the
extraction. The other steps are NumPy vector operations, and they are fast. The C++ code
is a line-for-line port of the NumPy version, which stays as `condense_tree_reference`. A
test makes sure that the two give the same rows *in the same sequence*. The subsequent
steps (the cluster numbers, and the run logic of `max_lambdas`) depend on the sequence.

Notation: `lambda = 1 / distance` (a larger lambda is denser).

**Condense** (`condense_tree`). Go through the tree from the root, breadth first. At each
merge node (from the top, this is a split):

- Both sides have `min_cluster_size` points or more: this is a real split, and both sides
  become new clusters.
- One side is smaller: the large side *continues* the parent cluster. The points of the
  small side fall out of the parent at this lambda.
- Both sides are smaller: all points fall out, and the parent cluster ends.

The result is a list of edges `(parent, child, lambda, child_size)`. A child is a point
or a cluster. The cluster numbers start at n (the root).

**Stability** (`compute_stability`). For a cluster C that starts at `lambda_birth`:
`stability(C) = Σ over the rows of C of (lambda − lambda_birth) × size`. This is
approximately how many points the cluster has, for how long.

**Selection** (`select_clusters`):

- **eom** (excess of mass): from the bottom up, keep a cluster if it is more stable than
  all its descendants together. If not, give the total stability of the descendants to
  the parent. `max_cluster_size` also applies here.
- **leaf:** take the leaves of the cluster tree.
- **cluster_selection_epsilon:** if a selected cluster split off at a distance below
  epsilon, replace it with its nearest ancestor that split off above epsilon. If the
  search comes to the root, the root is used only if `allow_single_cluster=True`.

**Labels** (`label_points`): each point receives its nearest selected ancestor cluster,
or noise. scikit-learn does this with a union-find over the condensed rows. The rows of a
parent always come before the rows of its children. Thus, cuhc moves an "owner" down from
the top. This gives the same result, and it is easier to read.

**Probabilities** (`membership_probabilities`): `min(lambda_point, lambda_max) /
lambda_max`. `lambda_max` is the largest lambda in the rows of the cluster. The same as
`max_lambdas` in scikit-learn, it uses consecutive runs of rows for each parent.

---

## 6. Ties: why the labels are not always equal to the scikit-learn labels

Mutual reachability makes many **edge weights that are exactly equal**. Each edge from a
point to a neighbour inside its core radius has the weight `core(point)`. In a test set
of 600 points, 113 of the 599 MST edges have the same weight as another edge.

If three or more groups join at the same height, the binary merge tree can pair them in
any sequence: `((A, B), C)` or `(A, (B, C))`. Both are correct single-linkage trees,
because each pair of points joins at the same height. **But the condensing step uses the
binary tree.** If A and B are small and C is large, `((A, B), C)` can make a cluster A∪B
that `(A, (B, C))` does not make. Thus, the HDBSCAN labels can depend on the sequence of
the ties.

This is also true for scikit-learn. Take the scikit-learn MST, keep the weight sequence,
and change only the sequence of the tied edges. Then the scikit-learn code gives
different labels in **16 of 20** tests. The scikit-learn sequence comes from the Prim
algorithm (one point at a time) and from the unstable `argsort` of NumPy. A parallel
algorithm cannot make the same sequence without a loss of the parallelism.

Thus, labels that are equal to the scikit-learn labels are possible only **without
ties**, and cuhc gives them in that case. With ties, the cuhc labels are the labels of
the scikit-learn algorithm for one correct tie sequence. That sequence is
`(weight, smaller point, larger point)`, and it is the same on each run. In the tests,
0 to 3 points of 600 are different (ARI 0.99 or more). On the larger benchmarks, the ARI
is 1.0.

An HDBSCAN without this dependence is possible: a group of simultaneous merges becomes
one split into several parts. The original paper uses this definition. Its results are
intentionally different from scikit-learn, so cuhc does not use it.

**A scikit-learn error.** With NumPy 2.5 or later, the `cluster_selection_epsilon` search
of scikit-learn 1.9 stops with an error
(`TypeError: only 0-dimensional arrays can be converted to Python scalars`) when it must
move up the tree. `traverse_upwards` puts an array of one element into a C integer. Plain
`sklearn.cluster.HDBSCAN` has this error, for example with
`cluster_selection_method="leaf", cluster_selection_epsilon=0.5, min_samples=1` on the
test data. The cuhc port takes the single element, which is the intended operation. The
tests skip the comparison in exactly those cases. A condensed tree made by hand examines
that path instead.

---

## 7. What the tests examine

`tests/test_hdbscan.py` examines all metrics, combinations of `min_cluster_size` and
`min_samples`, and options (`leaf`, epsilon, `alpha`, `allow_single_cluster`,
`max_cluster_size`):

1. **The tree is the scikit-learn tree, except for the tie sequence.** The test compares
   the cophenetic distances (the height at which each pair of points first joins). These
   do not depend on the tie sequence. The tolerance is 1e-5 relative, plus 1e-6 absolute
   for very small cosine distances.
2. **The extraction is exactly the scikit-learn code.** The private scikit-learn function
   `tree_to_labels`, on the cuhc tree, must give the same labels and probabilities (to
   1e-12).
3. **The full result:** the same partition as scikit-learn without ties
   (`min_samples=1`), and ARI 0.98 or more with ties.

The tests also examine:

- The core distances, compared with a NumPy reference, also with duplicate points
  (distance 0).
- The fused and the radix-select core distances, which must be identical bit for bit
  (k = 1, 5 and 64; and 65 and 200 with radix select only), on a size that is not a
  multiple of the tile.
- The C++ condensing step, which must be identical to the NumPy reference, also in the
  row sequence.
- The epsilon search on a tree made by hand, with expected answers from the scikit-learn
  algorithm.
- Data that is only noise.
- The defaults, the estimator behaviour and incorrect parameters.

`tests/test_spatial_pruning.py` makes sure that spatial pruning does not change the tree.

---

## 8. Results

RTX 3080, 8 features, 20 blobs plus 10% uniform noise, `min_cluster_size=25` (thus
`min_samples=25` and the fused kernel):

| n | cuhc total | GPU tree | extraction | scikit-learn | Speed increase | ARI compared with scikit-learn |
|---|---|---|---|---|---|---|
| 5,000 | 0.01 s | 0.00 s | 0.01 s | 0.29 s | approximately 30 times | 0.998 |
| 20,000 | 0.03 s | 0.02 s | 0.02 s | 3.9 s | approximately 115 times | 1.000 |
| 50,000 | 0.11 s | 0.08 s | 0.04 s | 20.9 s | approximately 185 times | 1.000 |
| 100,000 | 0.26 s | 0.25 s | 0.06 s | 104.8 s | approximately 400 times | 1.000 |
| 200,000 | 0.83 s | 0.72 s | 0.17 s | | | |
| 500,000 | 4.2 s | 3.8 s | 0.44 s | | | |
| 1,000,000 | 8.9 s | 8.1 s | 0.91 s | | | |

The ARI values are rounded. The 20,000-point value is 0.9997. This data has 8 features
and 20 clusters, so spatial pruning operates (`docs/all_pairs_passes.md`). For random
normal data with 32 features, which does not prune, HDBSCAN on 1,000,000 points takes
approximately 30 s.

The columns are different runs. Thus, the parts do not add up exactly.

The time is mostly in the all-pairs passes: one for the core distances, and one for each
Borůvka round. The extraction is a small part.

To measure again, use `python3 bench/bench_hdbscan.py --sizes ...`.

---

## 9. Limits

- **Small n.** The fused kernel has one block for each 128 rows. Below approximately
  20,000 points, the GPU has too few blocks (40 blocks at 5,000 points). Two to four
  blocks for each row tile, with a merge of their lists, would correct this. The time is
  already only some milliseconds.
- **Ties** (section 6). The labels depend on the tie sequence, the same as in
  scikit-learn.
- **k > 64** uses the radix select, which is approximately 2.5 times slower.
