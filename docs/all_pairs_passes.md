# All-pairs passes: symmetry and spatial pruning

This document explains two methods that make the all-pairs passes faster. The passes
are in `src/distance_matrix.cuh` and `src/spatial.cuh`. Three algorithms use them:

- Single linkage (Borůvka minimum spanning tree, `docs/boruvka_mst.md`).
- Ward linkage (reciprocal-nearest-neighbour rounds, `docs/rnn_ward.md`).
- HDBSCAN (core distances and the minimum spanning tree, `docs/hdbscan.md`).

Contents:

1. [The all-pairs pass](#1-the-all-pairs-pass)
2. [Symmetry](#2-symmetry)
3. [Spatial pruning](#3-spatial-pruning)
4. [When cuhc uses pruning](#4-when-cuhc-uses-pruning)
5. [Results](#5-results)
6. [Effects on the output](#6-effects-on-the-output)
7. [Limits](#7-limits)

---

## 1. The all-pairs pass

In one pass, each point finds its nearest point in a different component. "Component"
has a different meaning in each algorithm:

| Algorithm | Component | Edge weight |
|---|---|---|
| Single linkage | A tree of the minimum spanning forest | The distance |
| HDBSCAN | A tree of the minimum spanning forest | `max(core(a), core(b), d(a, b) / alpha)` |
| Ward linkage | The cluster itself | `2·nₐ·n_b / (nₐ + n_b) · d²` |

The kernel divides the n × n pairs into tiles of 128 × 128 pairs. Each block of threads
calculates one tile in registers. Then it keeps only the minimum of each row. The kernel
does not write the distances to memory.

Before these changes, the pass calculated all n² pairs. At 1,000,000 points this is
10¹² pairs for each pass. Borůvka does approximately 20 passes. Ward does approximately
60 passes, on fewer clusters each time.

---

## 2. Symmetry

### 2.1 The method

For all built-in metrics, d(x, y) is equal to d(y, x), bit for bit:

- Euclidean, Manhattan, Chebyshev and Minkowski use only |xₖ − yₖ|. In IEEE arithmetic,
  x − y is exactly −(y − x).
- Cosine uses x·y, ‖x‖² and ‖y‖². Multiplication and addition are commutative.

The edge weights are also symmetric. For example, the Ward weight multiplies
`nₐ · n_b` before it doubles the result, because `(2 · nₐ) · n_b` can round differently
from `(2 · n_b) · nₐ`.

Thus, the pass must calculate each unordered pair one time only. The symmetric kernel
(`nearestOtherComponentSymmetricKernel`) operates as follows:

1. The grid has one block for each tile (I, J) with I ≤ J. This is the upper triangle
   of tiles, approximately half of all the tiles.
2. Each block calculates the weights of its tile one time.
3. The block finds the minimum of each row, the same as before.
4. If I < J, the block also finds the minimum of each column. Pair (row, column) is
   also pair (column, row), so the column point uses the same weight.
5. A diagonal tile (I = J) contains both directions. It does step 3 only.

### 2.2 The column minimum

The threads that own one column are in different rows of the thread grid. The kernel
combines them in three steps:

1. Each thread finds the minimum of its 8 rows.
2. Two threads in each warp own the same column. A warp shuffle combines them.
3. Each warp does an `atomicMin` on a shared-memory array with one entry for each
   column. Then the block does one global `atomicMin` for each column.

Each candidate is a packed pair (weight, index). The packed value has the weight in the
high 32 bits and the index in the low 32 bits. Thus, `atomicMin` selects the lower index
if two weights are equal. This is the same tie rule as the full kernel.

### 2.3 Results

The results are identical, bit for bit, to the full kernel. The speed increase depends
on the dimension:

| Features | Speed increase (200,000 points) |
|---|---|
| 2 | 1.27 to 1.29 times |
| 8 | 1.28 to 1.36 times |
| 32 | 1.63 to 1.67 times |
| 128 | 1.83 to 1.94 times |

At high dimensions, the tile calculation is most of the time. Thus, the speed increase
is almost 2 times. At low dimensions, the work after the tile calculation (the minimum
of rows and columns) is a large part of the time. That work does not decrease.

The kernel is used for all metrics that have `metrics::IsSymmetric`. A user-defined
metric does not have it, unless the user adds it.

---

## 3. Spatial pruning

### 3.1 The idea

Most tiles contain no nearest neighbour. If two groups of points are far apart, no
point in one group is the nearest point of a point in the other group. Spatial pruning
finds these tiles and does not calculate them. The result stays exact.

The method has four parts:

1. Sort the points so that each tile contains points that are near each other.
2. Calculate a bounding box for each tile.
3. Calculate a lower bound for the weight of all pairs between two boxes.
4. Do not calculate a tile if its lower bound is larger than the current best weight of
   all rows.

### 3.2 The sort

`spatial::spatialOrder` puts the points in k-d tree order. The function operates on the
CPU:

1. Start with all the points as one range.
2. Find the feature with the largest spread in the range. The function uses a sample of
   64 points for this.
3. Split the range at a tile boundary near its middle, along that feature
   (`std::nth_element`).
4. Do steps 2 and 3 again for each half, until each range is one tile (128 points).

At 1,000,000 points, the sort takes approximately 0.3 seconds. A GPU kernel then copies
the points into the new order.

The algorithms operate on the sorted points. Before cuhc returns the result, it changes
each index back to the original index:

- Single linkage and HDBSCAN: the edges of the minimum spanning tree.
- HDBSCAN: the core distances.
- Ward linkage: the representative point of each cluster.

### 3.3 The bounding boxes

`spatial::buildTileBoxes` calculates these values for each tile:

- The minimum and the maximum of each feature (the box).
- The smallest per-point value of the edge weight: the core distance for HDBSCAN, or the
  cluster size for Ward.
- The smallest and the largest component number in the tile.

### 3.4 The lower bound

The gap between two boxes in feature k is:

```
gapₖ = max(0, lowB[k] − highA[k], lowA[k] − highB[k])
```

For each pair of points (x in box A, y in box B), |xₖ − yₖ| ≥ gapₖ.

Some metrics never decrease when one |xₖ − yₖ| increases. These metrics have
`metrics::HasBoxBound`: Euclidean, squared Euclidean, Manhattan, Chebyshev and
Minkowski. For these metrics, the metric of the gaps is a lower bound for all the pairs:

```
bound = finalize(accumulate(... accumulate(init(), gap₀, 0) ..., gap_d, 0))
```

Cosine does not have this property, because it uses the direction of the vectors. Thus,
cuhc does not prune for cosine.

All three edge weights never decrease when the distance or the per-point values
increase. Thus, the weight of the bound with the smallest values of the two tiles is a
lower bound for all the edges between the two tiles.

The calculated distance and the calculated bound can have rounding errors. To prevent
an incorrect skip, cuhc multiplies the bound by `1 − slack`. The slack is
`1e-5 + (features + 8) · 5e-7` (`spatial::boxSlack`). This is much larger than the
rounding error, and too small to decrease the pruning.

### 3.5 The walk kernel

`nearestOtherComponentWalkKernel` has one block for each row tile. Each block operates
as follows:

1. Keep the best edge of each of its 128 rows in shared memory.
2. Visit the column tiles in this sequence: the row tile itself, then the tiles at
   index distance 1, 2, 3 and more (I, I+1, I−1, I+2, …). After the sort, tiles with
   near indices are near in space. Thus, good candidates come first.
3. Before each tile, thread 0 does the test:
   - If all the points of the two tiles are in one component, skip the tile. No pair in
     it counts.
   - Calculate the lower bound of the tile. If the bound is larger than the largest
     current best of the rows, skip the tile.
4. If the block does not skip the tile, calculate it and update the best edges.
5. After each calculated tile, find the new largest best of the rows.
6. At the end, write the best edge of each row.

A skip needs a bound that is strictly larger than the current best. Thus, a skipped
pair cannot have the same weight as a best edge, and the tie rule does not change.

This kernel does not use symmetry. Each block owns its rows, so it can use its current
best edges for the test. Thus, the kernel calculates each tile pair two times, one time
from each side. For this reason, pruning is better than symmetry only if it skips more
than half of the tiles.

### 3.6 Core distances

The core-distance kernel (`kthNearestFusedKernel` in `src/hdbscan.cuh`) already has one
block for each row tile. It keeps the k nearest distances of each row in shared memory.
Pruning adds the same test before each tile. The bound is the largest k-th distance of
the rows.

The kernel now always visits the tiles in the walk sequence, also without pruning. The
old sequence was 0, 1, 2, …. For sorted points, each tile in that sequence is nearer
than the tile before it. Thus, most distances changed the lists, and each change is
slow. In the walk sequence, the near tiles come first, and the lists change rarely after
the first tiles.

---

## 4. When cuhc uses pruning

Pruning is useful only if the data has a low intrinsic dimension. Clustered data and
data with few features usually prune well. Random data with many features does not
prune. In 8 or more dimensions, the boxes of random normal data overlap almost all the
other boxes.

cuhc makes the decision in these steps (`src/capi.cu`):

1. Do not prune if one of these conditions is true:
   - The metric does not have `metrics::HasBoxBound` (for example cosine).
   - There are fewer than 16,384 points. The passes are fast without pruning.
   - There are more than 64 features.
2. Sort the points.
3. Run the walk kernel on a sample of approximately 32 row tiles
   (`spatial::estimatePrunedShare`). This costs a small part of one pass.
4. If the sample calculates more than half of its tiles, do not prune. The points stay
   in the sorted order, which has no cost for the other kernels.
5. During the passes, if one pass calculates more than half of the tiles, do not prune
   in the subsequent passes (`spatial::kPruneGiveUpShare`).

For tests and benchmarks, `cuhc._lib.set_pruning` sets the mode: `"auto"` (the
default), `"always"` or `"never"`. `cuhc._lib.prune_stats()` gives the number of tiles
that the pruned passes calculated, and the total number of tiles.

---

## 5. Results

RTX 3080, 100,000 points, `bench/bench_pruning.py`. "Tiles" is the share of tiles that
the pruned passes calculated. The speed increase compares `"auto"` with `"never"` (best
of 3 runs).

| Data | Features | Task | Tiles | Auto | Never | Speed increase |
|---|---|---|---|---|---|---|
| Blobs | 2 | single | 1.5% | 0.067 s | 0.276 s | 4.1 times |
| Blobs | 2 | ward | 3.6% | 0.061 s | 0.093 s | 1.5 times |
| Blobs | 2 | hdbscan | 1.7% | 0.165 s | 0.358 s | 2.2 times |
| Blobs | 8 | single | 17% | 0.123 s | 0.228 s | 1.9 times |
| Blobs | 8 | ward | 17% | 0.112 s | 0.152 s | 1.4 times |
| Blobs | 8 | hdbscan | 27% | 0.227 s | 0.296 s | 1.3 times |
| Blobs | 32 | single | 30% | 0.341 s | 0.426 s | 1.25 times |
| Blobs | 32 | hdbscan | 49% | 0.412 s | 0.468 s | 1.14 times |
| Normal | 8 | hdbscan | 100% | 0.192 s | 0.181 s | 0.94 times |

For data that does not prune, the sort and the sample cost approximately 6% at 100,000
points.

At 1,000,000 points, the passes are most of the time, and pruning has a larger effect
(one run of each, `--repeats 1`):

| Data | Features | Task | Auto | Never | Speed increase |
|---|---|---|---|---|---|
| Blobs | 2 | single | 2.36 s | 28.0 s | 11.9 times |
| Blobs | 2 | ward | 1.10 s | 6.74 s | 6.1 times |
| Blobs | 2 | hdbscan | 3.58 s | 25.4 s | 7.1 times |
| Blobs | 8 | single | 5.71 s | 25.9 s | 4.5 times |
| Blobs | 8 | ward | 3.41 s | 13.5 s | 4.0 times |
| Blobs | 8 | hdbscan | 5.94 s | 23.0 s | 3.9 times |
| Blobs | 32 | single | 27.6 s | 46.4 s | 1.7 times |
| Blobs | 32 | ward | 13.4 s | 44.4 s | 3.3 times |
| Blobs | 32 | hdbscan | 22.3 s | 36.1 s | 1.6 times |
| Normal | 32 | single | 32.7 s | 32.3 s | 0.99 times |
| Normal | 32 | ward | 52.5 s | 52.0 s | 0.99 times |
| Normal | 32 | hdbscan | 30.6 s | 30.2 s | 0.99 times |

For random normal data, the sample shows that pruning does not help. Then cuhc does not
prune, and the sort and the sample cost approximately 1%.

---

## 6. Effects on the output

Pruning changes the index of each point during the calculation. This has two effects:

- **Ties.** If two edges have exactly the same weight, the edge to the lower index wins.
  The index is now the index in the sorted order. Thus, merges with equal heights can
  occur in a different sequence. HDBSCAN has many such ties, because many mutual
  reachability distances are equal to a core distance. The tree is the same, but the
  extracted labels can change a little (adjusted Rand index approximately 0.998 in the
  tests). The scikit-learn labels also change with the tie sequence.
- **Ward rounding.** The merged centroid uses the cluster in the lower slot as its
  start value. The slots are in a different sequence, so the last bit of a centroid can
  change. The merge heights stay the same to approximately 3e-7 (relative).

Without ties, single linkage gives a result that is identical, bit for bit. The core
distances are always identical.

`tests/test_spatial_pruning.py` runs each algorithm with pruning on and off, and
compares the results. To run all the tests with pruning always on, set
`CUHC_TEST_PRUNING=always`.

---

## 7. Limits

- Pruning does not help for data with a high intrinsic dimension. cuhc then does not
  use it (section 4).
- The walk kernel does not use symmetry. If pruning skips fewer than half of the tiles,
  the symmetric kernel is faster.
- The k-d sort operates on the CPU. At 10,000,000 points it takes several seconds.
- Each block keeps the box of its row tile in shared memory (2 · features floats).
  Thus, cuhc does not prune above 256 features, also in the `"always"` mode.
