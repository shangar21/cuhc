# Borůvka MST: single linkage without the distance matrix

This document explains how cuhc calculates exact single linkage in O(n) memory
(`algorithm="mst"`). With this method, cuhc can cluster millions of points. The code is
in these files:

- `src/mst.cuh`: the Borůvka algorithm and the conversion to a merge tree.
- `src/distance_matrix.cuh`: the GPU pass (`nearestOtherComponentKernel` and its
  symmetric version).
- `src/reductions.cuh`: the argmin functions.

The faster forms of the GPU pass (symmetry and spatial pruning) are in
`docs/all_pairs_passes.md`.

Contents:

1. [Why single linkage is a minimum spanning tree](#1-why-single-linkage-is-a-minimum-spanning-tree)
2. [Which other methods use the MST](#2-which-other-methods-use-the-mst)
3. [The Borůvka algorithm](#3-the-borůvka-algorithm)
4. [The GPU pass](#4-the-gpu-pass)
5. [Ties](#5-ties)
6. [From the MST to the merge tree](#6-from-the-mst-to-the-merge-tree)
7. [Results](#7-results)
8. [How the tests compare trees](#8-how-the-tests-compare-trees)

---

## 1. Why single linkage is a minimum spanning tree

Think of the points as a complete graph. Each pair of points is an edge, and the weight
of the edge is the distance. A **minimum spanning tree (MST)** is the set of n − 1 edges
that connects all the points with the smallest total weight.

Single linkage merges the two clusters with the **shortest edge between them**, again and
again. The Kruskal MST algorithm takes the **shortest edge that connects two different
components**, again and again. This is the same rule. The components of Kruskal are the
clusters of single linkage, and they merge in the same sequence at the same heights.

Thus, calculate the MST with any method, sort its edges by weight, and merge along them
in that sequence. The result is exactly the single-linkage tree. The MST has only n − 1
edges, so the algorithm does not keep n² values.

---

## 2. Which other methods use the MST

Of the agglomerative methods, **only single linkage** uses the MST. Its cluster distance
is "the shortest edge between the clusters", which is what the MST keeps. The other
linkages use information that the MST does not keep:

| Linkage | The cluster distance depends on | MST? | Algorithm without the matrix |
|---|---|---|---|
| single | the nearest pair | **yes** | this document |
| complete | the farthest pair | no | none (dense matrix, `docs/dense_linkage.md`) |
| average | all pairs | no | none (dense matrix, `docs/dense_linkage.md`) |
| ward | centroids and sizes | no | reciprocal-nearest-neighbour rounds (`docs/rnn_ward.md`) |

**HDBSCAN** also uses this code. It is single linkage on the "mutual reachability"
distance (`docs/hdbscan.md`).

---

## 3. The Borůvka algorithm

The Kruskal algorithm sorts all the edges, which are n² edges. The Borůvka algorithm uses
only "nearest neighbour" queries, and a GPU does these queries well:

```
each point is a component
while there is more than one component:
    for each point p:      the nearest point in a different component    (GPU, all pairs)
    for each component C:  the shortest edge of its points                 (CPU, O(n))
    add those edges; merge the components that they connect               (CPU, union-find)
```

**Why it is correct (the cut property).** Divide the points into two groups in any way.
Then the shortest edge between the groups is in the MST. The shortest edge out of a
component is the shortest edge between "this component" and "all the other points".
Thus, it is in the MST, and the algorithm can add all the edges of a round together.

**Why it is fast.** In each round, each component merges with one or more other
components. Thus, the number of components decreases to half or less in each round, and
there are log₂ n rounds or fewer. Usually, the components merge in larger groups.

**Cost.** Each round is one pass over all n² pairs. This is the same arithmetic as the
distance matrix, but the algorithm keeps no distances. The memory is the points plus some
arrays of length n.

The host loop is `boruvkaMst` in `src/mst.cuh`. In each round, the CPU work is O(n):

- Copy `component[]` to the GPU (n integers).
- Copy `nearest[]` back (n × 8 bytes).
- Find the shortest edge of each component.
- Do the union-find.
- Give the components the numbers 0 to k − 1.

At 1,000,000 points, the copies are approximately 12 MB for each round. This is very
small, compared with the GPU pass.

---

## 4. The GPU pass

`nearestOtherComponentKernel` uses the tiled distance loop of `docs/distance_matrix.md`
(`computeTileDistances`). Only the work after the distance calculation is different:

```
pairwiseDistanceKernel:        write tileDistances[r][c] to the matrix
nearestOtherComponentKernel:   for each row, keep the nearest column in another component
```

These are the steps for one block (a tile of 128 × 128, 16 × 16 threads, 8 × 8 outputs
for each thread):

1. **The distances are in registers.** After `computeTileDistances`, each thread has its
   64 distances in `tileDistances[8][8]`.
2. **The minimum of each thread.** For each of its 8 rows, the thread finds the minimum
   of its 8 columns. It does not use columns in the same component as the row. Thus, it
   also does not use the point itself. The result is one `MinLoc{distance, column}` for
   each row.
3. **The 16 threads of a row.** In the tile layout, the 16 threads with the same
   `threadRow` are 16 consecutive lanes of one warp. `groupMinLoc<16>` combines them with
   4 butterfly steps of `__shfl_xor_sync` (offsets 8, 4, 2 and 1). An XOR with an offset
   below 16 stays in the group of 16 lanes. Thus, the two halves of the warp reduce
   independently, at the same time.
4. **The blocks of a row.** The columns of a row are in n / 128 blocks. Lane 0 of each
   group does one `atomicMin` on `nearest[row]`. This is a 64-bit value that contains the
   distance and the column (see below). Thus, each block does 128 atomic operations, not
   128 × 128 writes.

**The edge weight.** Before step 2, the kernel can change each distance into a different
edge weight (the `EdgeWeight` parameter). Single linkage uses the distance itself
(`DistanceWeight`). HDBSCAN uses the mutual reachability distance, and Ward linkage uses
the Ward distance between clusters.

**One integer for (distance, index).** `atomicMin` operates on integers, not on structs.
`packMinLoc` (in `reductions.cuh`) makes `(float bits of distance) << 32 | index`. For
floats that are not negative, a comparison of the bit patterns as unsigned integers gives
the same sequence as a comparison of the floats. Thus, the smallest packed value has the
smallest distance, and equal distances go to the smallest index. The function first sets
negative distances to 0, because rounding can make a cosine distance −1e−8. `nearest[]`
starts with all bits set (`cudaMemset(0xff)`). This is the largest possible value, and it
means "nothing found".

**Threads outside the matrix also do the shuffle.** For `__shfl_xor_sync`, all lanes in
the mask must take part. Threads with a row after the end calculate a dummy value and do
the shuffle. They do not do the atomic operation. If they stop early with `return`, the
reduction can stop or give incorrect values.

---

## 5. Ties

If edges have equal weights, a simple Borůvka implementation can **add a cycle**. For
example, components A, B and C are at the same distance from each other. A selects the
edge to B, B selects C, and C selects A. The algorithm adds all three edges. This makes a
cycle, and the result has one edge less than a tree.

cuhc has two safeguards:

1. **One global sequence of edges.** cuhc compares edges by
   (weight, smaller end point, larger end point) (`edgeLess`). On the GPU, equal weights
   go to the lowest column, which agrees with this sequence for one row. The CPU scan for
   each component uses `edgeLess` directly. With a strict total sequence, all edge
   weights are effectively different, and a cycle cannot occur.
2. **Union-find before an edge is added.** If the two end points of an edge are already
   connected, the algorithm does not add the edge. This also covers two components that
   select the same edge.

`test_many_ties_give_the_same_heights_as_dense` examines this on an integer grid of
20 × 20 points. On this grid, almost all nearest-neighbour distances are exactly 1.

---

## 6. From the MST to the merge tree

`singleLinkageFromMst` operates on the CPU:

1. Sort the n − 1 edges with `edgeLess`.
2. For each edge in sequence, find the two clusters of its end points (union-find).
3. Write a tree row `[smaller id, larger id, weight, size]`. Join the two clusters, and
   give the result the number n + step.

This is O(n log n) for the sort and almost O(n) for the union-find. It operates in C++ in
`cuhc_linkage`, because a Python loop over 1,000,000 merges takes several seconds.

---

## 7. Results

RTX 3080, 32 features, random normal data, full single-linkage tree, `algorithm="mst"`:

| n | MST | scikit-learn | Speed increase | Same labels (k = 10) |
|---|---|---|---|---|
| 2,000 | 0.001 s | 0.11 s | approximately 90 times | yes |
| 10,000 | 0.006 s | 1.6 s | approximately 280 times | yes |
| 20,000 | 0.027 s | 6.7 s | approximately 250 times | yes |
| 50,000 | 0.10 s | — | | |
| 100,000 | 0.40 s | — | | |
| 200,000 | 1.4 s | — | | |
| 500,000 | 8.3 s | — | | |
| 1,000,000 | 33.5 s | — | | |

Random normal data with 32 features does not prune. Thus, these times use the symmetric
pass (`docs/all_pairs_passes.md`). For clustered data with few features, spatial pruning
makes single linkage faster again: 2.4 s at 1,000,000 points in 20 blobs with 2
features.

The time increases as n², because each round is an all-pairs pass, and the number of
rounds increases very little with n. The memory is O(n). At 1,000,000 points, it is
approximately 140 MB of GPU memory (the points plus some arrays of length n). The dense
matrix for the same points is 4 TB.

`cuhc.linkage(X, method="single")` and `AgglomerativeClustering(linkage="single")` use
this algorithm automatically (`algorithm="auto"`).

To measure again, use
`python3 bench/bench_linkage.py --methods single --algorithm mst --sizes ...`.

---

## 8. How the tests compare trees

`tests/test_mst.py` compares the MST algorithm with scikit-learn and with the dense
algorithm of cuhc. It uses sizes near the tile edges, all metrics, 8,000 points, an
integer grid with many ties, and incorrect requests.

**float32 can make ties.** In the test with 8,000 points, scikit-learn has two merges at
the float64 heights 1.782150126 and 1.782150202. In float32 (cuhc), both heights become
1.782150149. Thus, they are a tie, and tied merges can occur in either sequence. If two
merges change places, the numbers of the clusters that they make also change places.
Thus, all subsequent rows that use those clusters are also different. A row-by-row
comparison of `Z` shows a difference, but the dendrogram is the same.

For this reason, `_helpers.assert_same_dendrogram` compares trees as **sets of
clusters**:

- In a hierarchy, two clusters are always nested or separate. Thus,
  (smallest point, size) identifies each cluster.
- Two trees are the same dendrogram if they have the same clusters at the same heights,
  in any merge sequence.

The comparison finds real differences. For example, a complete-linkage tree fails it,
and a tree with one height incorrect by 1% also fails it.

cuhc does not accept input with NaN or infinity, and gives a `ValueError` (the same as
scikit-learn). A NaN distance would become 0 after the clamp, and join points without a
warning.
