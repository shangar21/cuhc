# Dense linkage: exact agglomerative clustering with the distance matrix

This document explains the dense algorithm (`algorithm="dense"`). It calculates exact
single, complete, average and Ward linkage with the full n × n distance matrix. The code
is in these files:

- `src/linkage.cuh`: the GPU algorithm.
- `python/cuhc/_linkage.py`: `linkage`, `fcluster` and the tree cut.
- `python/cuhc/_estimators.py`: `AgglomerativeClustering`.

Complete and average linkage always use this algorithm. Single and Ward linkage have
faster algorithms that do not keep the matrix (`docs/boruvka_mst.md` and
`docs/rnn_ward.md`).

Contents:

1. [The basic algorithm, and why it is too slow](#1-the-basic-algorithm-and-why-it-is-too-slow)
2. [Lance-Williams: the distances after a merge](#2-lance-williams-the-distances-after-a-merge)
3. [The nearest-neighbour cache](#3-the-nearest-neighbour-cache)
4. [Slots: the position of each cluster in the matrix](#4-slots-the-position-of-each-cluster-in-the-matrix)
5. [The three kernels of one merge](#5-the-three-kernels-of-one-merge)
6. [The argmin with warp shuffles](#6-the-argmin-with-warp-shuffles)
7. [From the tree to the labels (CPU)](#7-from-the-tree-to-the-labels-cpu)
8. [Results](#8-results)
9. [Limits](#9-limits)

---

## 1. The basic algorithm, and why it is too slow

```
D = n x n distances
repeat n-1 times:
    (a, b) = the nearest pair of clusters      <- a scan of all of D: O(n^2)
    merge them, change D
```

There are n − 1 merges, and each merge does an O(n²) scan. The total is **O(n³)**. At
50,000 points, this is approximately 10¹⁴ reads of a 10 GB matrix. One scan at the full
GPU memory bandwidth (approximately 760 GB/s) takes approximately 13 ms. Thus, 50,000
scans take more than 10 minutes.

The solution is to never scan the full matrix (section 3). For this, the distances after
a merge must be cheap to calculate (section 2).

---

## 2. Lance-Williams: the distances after a merge

After clusters a and b merge, the algorithm must have the distance from the new cluster
a∪b to each other cluster k. The Lance-Williams formulas calculate it from the known
distances, without the points (`lanceWilliams` in `linkage.cuh`):

| Linkage | d(a∪b, k) |
|---|---|
| single | `min(d_ak, d_bk)` |
| complete | `max(d_ak, d_bk)` |
| average | `(n_a·d_ak + n_b·d_bk) / (n_a + n_b)` |
| ward | `sqrt( ((n_a+n_k)·d_ak² + (n_b+n_k)·d_bk² − n_k·d_ab²) / (n_a+n_b+n_k) )` |

`n_x` is the number of points in cluster x. The Ward formula uses plain Euclidean
distances. scipy and scikit-learn use the same convention for their merge heights.

The calculation is in **double**, and the result is kept as a float. This is only O(n)
work for each merge. It prevents an increase of the rounding errors over tens of
thousands of merges.

---

## 3. The nearest-neighbour cache

For each cluster, the algorithm keeps its nearest other cluster (`nearestDistance[]` and
`nearestSlot[]`). Then the nearest pair of all is the smallest `nearestDistance`. This is
an argmin over **n** values, not over n² values.

After a merge, the cache must stay correct. For this, the algorithm uses a property of
all four linkages, **reducibility**:

> d(a∪b, k) ≥ min(d_ak, d_bk)

Thus, a merge of two clusters never gives a cluster that is nearer to k than the nearer
of the two. All the formulas in the table have this property: min, max, a weighted mean,
and Ward.

After a and b merge, these are the possible cases for a cluster k:

- **The nearest cluster of k was a different cluster j.** Then
  d(a∪b, k) ≥ min(d_ak, d_bk) ≥ d_kj. Thus, j is still the nearest. The cache does not
  change.
- **The nearest cluster of k was a or b, and the merged cluster is at the same distance
  or nearer.** Then the merged cluster is the nearest. The algorithm sets
  `nearestSlot[k] = a`. Single linkage always has this case, because `min(d_ak, d_bk)` is
  the old nearest distance.
- **The nearest cluster of k was a or b, and the merged cluster is farther.** Then k must
  scan its row again, which is O(n).
- **The cluster a itself.** All of its row changed. Thus, it always scans its row again.

Each merge costs O(n) for the change, plus O(n) for each row that scans again. Usually,
only a small number of rows scan again. Thus, a merge is usually O(n), and the full tree
is approximately O(n²), not O(n³). Some inputs cause more scans. The worst case is still
O(n³).

This method operates only for reducible linkages. Centroid linkage and median linkage are
not reducible. For this reason, cuhc does not supply them.

---

## 4. Slots: the position of each cluster in the matrix

The matrix does not become smaller. Each cluster has a **slot**: a row and column index.
When the clusters in slots a < b merge:

- The merged cluster goes into slot a. The Lance-Williams values replace row a and
  column a.
- Slot b stops: `isActive[b] = 0`, `nearestDistance[b] = +inf`, and **column b becomes
  +inf**.

The diagonal is +inf from the start. The columns of stopped slots are also +inf. Thus,
a scan of a row is only "the minimum of the row". It does not have to examine which
slots are active, and it cannot select a stopped cluster.

The slot numbers are not the cluster numbers of the output. `clusterId[slot]` keeps the
number. The points are clusters 0 to n − 1, and merge i makes cluster n + i, the same as
in scipy.

Example: single linkage on points at 0, 1, 5 and 7 on a line (slots 0 to 3):

```
merge 0: nearest pair slots (0,1), d=1  -> row [0, 1, 1, 2]   slot 0 now has cluster 4
merge 1: nearest pair slots (2,3), d=2  -> row [2, 3, 2, 2]   slot 2 now has cluster 5
merge 2: nearest pair slots (0,2), d=4  -> row [4, 5, 4, 4]   slot 0 now has cluster 6
```

---

## 5. The three kernels of one merge

The host loop (`denseLinkageOnMatrix`) starts three kernels for each merge. It **does not
wait for the GPU** in the loop. The first kernel selects the merge, and gives it to the
second kernel in device memory (`*state.merge`). The CPU only puts the kernel starts in
the queue. Thus, the GPU is rarely idle.

Before the loop, the tiled distance kernel calculates the matrix. Then the diagonal
becomes +inf, and each row does one scan to fill the cache.

**Kernel 1, `findClosestPairKernel`** (one block of 1024 threads):

- The argmin over `nearestDistance` (section 6) gives slot s and its nearest slot
  `nearestSlot[s]`.
- The kernel names them a < b, and writes tree row `step` =
  `[smaller id, larger id, distance, size]`.
- It keeps the old sizes in `*state.merge`. Then it changes `clusterSize[a]` and
  `clusterId[a]`, and stops slot b.
- It clears the list of rows to scan again.

**Kernel 2, `mergeKernel`** (one thread for each slot k):

- It calculates d(a∪b, k) with Lance-Williams, and writes it to `D[a][k]` and `D[k][a]`.
  It writes +inf to `D[k][b]`.
- It applies the rules of section 3. It adds the rows that must scan again to
  `rescanSlots` with `atomicAdd`. This is a cheap method for many threads to make one
  list.
- The thread for k = a adds a to the list.

The reads of row a and row b are coalesced. The column writes (`D[k][a]` and `D[k][b]`)
are strided, but there are only 2n of them for each merge.

**Kernel 3, `rescanKernel`** (a fixed grid of 4 × the number of SMs blocks):

- Each block takes rows from `rescanSlots` (`item = blockIdx.x, +gridDim.x, ...`).
- For each row, it does a coalesced min and argmin over n floats with the block
  reduction.

If the list has only one or two rows, the other blocks stop immediately.

Kernel 1 keeps the old sizes in `*state.merge` for this reason: the Ward and average
formulas in kernel 2 use n_a and n_b from *before* the merge. Kernel 1 already changed
`clusterSize[a]`. Thus, kernel 2 reads the kept values. Kernel 2 reads `clusterSize[k]`
only for k ≠ a, b, which did not change.

**Clusters that are larger than one point.** `denseLinkageOnMatrix` can also start from
a matrix of cluster distances, with a start size and a start number for each slot. The
Ward rounds use this to give their last clusters to the dense algorithm
(`docs/rnn_ward.md`).

---

## 6. The argmin with warp shuffles

Kernels 1 and 3 find "the smallest value and its position". The parts are in
`src/reductions.cuh`:

```cpp
struct MinLoc { float distance; int index; };
minLoc(x, y)             // the smaller distance wins; if equal, the lower index wins
groupMinLoc<G>(v)        // log2(G) steps of __shfl_xor_sync; G = 32 is the full warp
blockMinLoc<B>(v)        // each warp reduces; lane 0 of each warp writes to shared memory;
                         // warp 0 reduces those B/32 values; thread 0 has the result
```

`__shfl_xor_sync(mask, value, offset)` lets each lane read `value` from lane
`lane ^ offset`. The value comes directly from the registers of that lane, not through
memory. After the step with offset 16, each lane has combined its value with one other
lane. After offset 8, each lane has the minimum of 4 lanes, and so on. Five steps cover
all 32 lanes. After them, *each* lane has the minimum of the warp (a "butterfly"
reduction).

The rule for equal distances (the lower index wins) makes the result the same on each
run.

The block version uses shared memory only to move one value from each warp. It ends
with a `__syncthreads()`, so that a loop can call it again. Kernel 3 does this.

---

## 7. From the tree to the labels (CPU)

`cut_tree(Z, num_merges)` in `_linkage.py` applies the first `num_merges` merges. Then it
gives each point the label of its cluster:

- Make `parent[c]`: the cluster that c merged into, for the applied merges. A cluster
  that did not merge is its own parent.
- **Pointer jumping:** do `parent = parent[parent]` until nothing changes. Each pass
  doubles the distance up the tree for each entry. Thus, it takes O(log n) NumPy vector
  passes, not a Python loop over the tree.
- The roots of the points are the clusters. `np.unique(..., return_inverse=True)` gives
  them the numbers 0 to k − 1.

The functions use this as follows:

- `AgglomerativeClustering(n_clusters=k)` applies the first n − k merges.
- With `distance_threshold=t`, it uses the scikit-learn rule: it does not apply merges
  with a distance of **t or more**. Thus, `n_clusters_ = count(distances ≥ t) + 1`.
- `fcluster(Z, t, "distance")` uses the scipy rule: it applies merges with a distance of
  **t or less**. The two libraries are different at exactly t. Each function agrees with
  the library that it replaces.

---

## 8. Results

RTX 3080, 32 features, random normal data, `algorithm="dense"`, time for the full tree.
scikit-learn uses scipy, with float64 input. "Same labels" compares the cut at k = 10.

| n | single | complete | average | ward | scikit-learn (average) | Speed increase | Same labels |
|---|---|---|---|---|---|---|---|
| 2,000 | 0.021 s | 0.022 s | 0.021 s | 0.024 s | 0.10 s | approximately 5 times | yes |
| 10,000 | 0.11 s | 0.14 s | 0.15 s | 0.16 s | 3.6 s | approximately 24 times | yes |
| 20,000 | 0.31 s | 0.39 s | 0.41 s | 0.42 s | 18.2 s | approximately 44 times | yes |
| 40,000 | 0.93 s | 1.30 s | 1.34 s | 1.42 s | not measured | | |
| 50,000 | 1.48 s | 1.93 s | 1.99 s | 2.14 s | not measured | | |

At 20,000 points, the speed increase is from 18 times (single) to 44 times (average).
The scipy single linkage is already fast, because it uses a minimum spanning tree. At
60,000 points, the matrix (14.4 GB) does not fit in the GPU memory. Then `cuhc.linkage`
gives a `MemoryError` with this information.

To measure again, use `python3 bench/bench_linkage.py --algorithm dense`.

The tests are `tests/test_linkage.py` (trees compared with scikit-learn, with 2,500
points and with duplicate points), `tests/test_agglomerative.py` (the estimator compared
with scikit-learn) and `tests/test_fcluster.py` (compared with scipy).

---

## 9. Limits

- **Memory:** n² floats. This is approximately 55,000 points on a GPU with 12 GB free.
  Single and Ward linkage have algorithms without the matrix (`docs/boruvka_mst.md`,
  `docs/rnn_ward.md`).
- **The cost of each merge:** 50,000 merges in approximately 2 s is approximately 40 µs
  for each merge. Most of this is the start of three kernels and an argmin over n values
  in one block. A CUDA graph for the loop, or one kernel in place of kernels 1 and 2,
  would decrease this cost.
- **Ties:** the lowest slot wins. With equal distances, the merge sequence can be
  different from scipy, but the tree is still correct. The tests with duplicate points
  examine this.
- **Only symmetric metrics:** the algorithm assumes d(x, y) = d(y, x). All built-in
  metrics are symmetric.
