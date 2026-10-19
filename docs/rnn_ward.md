# Reciprocal-nearest-neighbour rounds: Ward linkage without the matrix

This document explains `src/rnn_ward.cuh` (`algorithm="rnn"`). The algorithm calculates
the exact Ward tree in O(n·d) memory. It is faster than the nearest-neighbour chain
(`docs/nn_chain.md`) and faster than the dense matrix (`docs/dense_linkage.md`) at all
measured sizes. Thus, `algorithm="auto"` uses it for Ward linkage.

Contents:

1. [Why one query at a time is slow](#1-why-one-query-at-a-time-is-slow)
2. [Rounds: all clusters do their query together](#2-rounds-all-clusters-do-their-query-together)
3. [Why a merge of all reciprocal pairs together is exact](#3-why-a-merge-of-all-reciprocal-pairs-together-is-exact)
4. [How much work the rounds do](#4-how-much-work-the-rounds-do)
5. [One round on the GPU](#5-one-round-on-the-gpu)
6. [The bad case, and the fallback](#6-the-bad-case-and-the-fallback)
7. [A fast Ward edge weight](#7-a-fast-ward-edge-weight)
8. [Floating point](#8-floating-point)
9. [Results](#9-results)

---

## 1. Why one query at a time is slow

The Ward distance between two clusters uses only their sizes and centroids:

```
ward(a, b) = sqrt( 2 · n_a · n_b / (n_a + n_b) ) · |c_a − c_b|
```

The nearest-neighbour chain asks "which cluster is nearest to *this* cluster?" in each
step. This compares one cluster with all others. Thus, each centroid comes from memory
and gives only one distance: d floats read (128 bytes at d = 32) for d multiply-adds.
The memory bandwidth is the limit. At 500,000 points, the chain reads centroids at
approximately 540 GB/s. This is approximately 4 billion Ward distances for each second.

A cache of parts of the distance matrix does not help much. Most chain steps ask about a
cluster that was not asked about before. Thus, there is little to use again, at most
approximately 2 times.

The number of queries for each read is the important value. The tiled distance kernel
(`docs/distance_matrix.md`) calculates a block of 128 × 128 distances from 128 + 128
loaded points. Thus, it uses each loaded value 128 times. Its limit is the arithmetic, not
the memory. At d = 32, it calculates approximately 25 times more distances each second.

---

## 2. Rounds: all clusters do their query together

```
each point is a cluster
repeat until one cluster is left:
    each cluster finds its nearest cluster (Ward distance)       <- one all-pairs pass
    each pair (a, b) with nearest(a) = b and nearest(b) = a merges
    the merged clusters replace their parts; the next round uses the remaining clusters
```

Two clusters that are the nearest cluster of each other are a **reciprocal nearest
neighbour (RNN) pair**. The chain finds RNN pairs one at a time, along a path of nearest
neighbours. The rounds find all of them together.

---

## 3. Why a merge of all reciprocal pairs together is exact

The chain uses this fact (`docs/nn_chain.md` section 2): the algorithm that merges the
nearest pair first also merges each RNN pair, at the same height. The fact depends on
**reducibility**. For Ward, a merge of a and b never makes a cluster that is nearer to a
cluster k than both a and b were.

The same fact applies to many pairs together. Take two RNN pairs (a, b) and (c, d) from
the same round. A merge of a and b cannot make a∪b nearer to c than the distance of c to
a or to b. Both of those distances are d(c, d) or more, because d is the nearest cluster
of c. Thus, c and d are still the nearest cluster of each other after the first merge,
and the second merge is still correct. By induction, all RNN pairs of a round can merge
in any sequence.

**Each round merges one pair or more.** The nearest pair of all clusters is always an RNN
pair. Ties need care. The kernel gives ties to the lower index, and the Ward weight of
(i, j) is *identical, bit for bit,* to the weight of (j, i) (section 8). Then, of the
nearest pairs, the pair with the lowest indices are the nearest cluster of each other.

**The merge heights come in the sequence of the rounds, not in sorted sequence.** The
same as for the chain, cuhc sorts them by height at the end (`orderChainMerges`). First,
it increases the height of each merge to the height of its children, because float
rounding can put a parent one ulp below a child.

---

## 4. How much work the rounds do

Each round is one pass over m × m pairs (m = the number of remaining clusters). A round
merges a part of the clusters. A measurement with 6,000 points gives:

| Data | Rounds | Total pairs | Part merged in each round |
|---|---|---|---|
| normal, d = 32 | 60 | 6.0 × n² | 7% to 21% |
| blobs, d = 8 | 35 | 3.4 × n² | 9% to 41% |
| uniform, d = 2 | 34 | 2.1 × n² | 12% to 63% |
| normal, d = 128 | 74 | — | 5% or more |

The chain calculates approximately 3 × n² pairs. Thus, the rounds do approximately the
same work, at most 2 times more. Each pair costs approximately 25 times less, so the
rounds are much faster.

---

## 5. One round on the GPU

The remaining clusters are always **compacted** in slots 0 to m − 1. Thus, the pass in
round r costs exactly m_r² pairs. Two buffers (`current` and `next`) contain the centroid,
the size and a **representative point** of each cluster. The representative point is a
point in the cluster. cuhc records the merges with it, the same as the chain.

```
1. nearestOtherComponent(centroids, m, SqEuclidean, WardWeight)    all-pairs pass
     nearest[i] = (Ward distance², slot) of the nearest cluster of i
     "component" = the slot itself, so only i itself is not a candidate
2. rnnSurvivorsKernel        survives[i] = 0 if i pairs with a lower slot j, else 1
3. cub::DeviceScan           newIndex = exclusive prefix sum of survives
4. rnnMergeKernel            one thread for each (slot, feature):
                               slot i stays             -> copy to next[newIndex[i]]
                               i pairs with a higher j  -> merged centroid to next[newIndex[i]],
                                                           and record (rep[i], rep[j], height)
5. the host reads the number of merges (the only wait in a round); swap the buffers
```

Step 1 is the Borůvka pass (`docs/boruvka_mst.md`). Each cluster is its own component,
and the edge weight is the Ward distance. HDBSCAN uses the same parameter for the mutual
reachability distance. For the symmetric and pruned forms of this pass, refer to
`docs/all_pairs_passes.md`.

**The same result on each run.** In a round, the merges go into the list with an atomic
counter. Thus, their sequence depends on the GPU schedule. After each round, the host
sorts the merges of the round by (height, u, v). Thus, the result is the same on each
run. A test examines this.

---

## 6. The bad case, and the fallback

The rounds are fast only if each round has many RNN pairs. Some data does not. On a line
where each gap is 1% larger than the gap before it, the nearest neighbour of each point
is its left neighbour. Thus, each round has only approximately one reciprocal pair:

```
x:       0   1  2.01  3.03  4.06 ...     (gaps 1, 1.01, 1.0201, ...)
nearest: →1  ←0  ←1    ←2    ←3          only (0, 1) is reciprocal
```

At 5,000 points, this is approximately 2,500 rounds and 840 × n² pairs: O(n³) in total.

Thus, after each round, `rnnWard` examines the part of the clusters that merged. If it is
less than `minProgress` (1%), the remaining m clusters go to a different algorithm:

- The **dense algorithm**, if the m × m matrix fits in 90% of the free GPU memory. The
  matrix contains the Ward distances between the current clusters,
  `sqrt(2·nᵢ·nⱼ/(nᵢ+nⱼ))·|cᵢ − cⱼ|`, and Lance-Williams continues from there. The dense
  code accepts start sizes and start numbers for this (`denseLinkageOnMatrix`).
- Otherwise, the **chain**, which starts from the current centroids and sizes
  (`nnChainWardMerges`).

The threshold is safe in both directions. Usual data merged 5% or more in each round in
all measurements, and the bad case merges approximately 1/m. The point where the rounds
and the fallback have the same cost is approximately 0.5% to 1%.

On the line of 5,000 points, the fallback decreases the time of the rounds from 0.15 s to
0.07 s, the same as the dense algorithm. Without it, the difference increases with n.

**No fallback by size.** A fallback to the dense algorithm below a fixed number of
clusters (1,000 to 16,000) was slower at all measured sizes. The last rounds are cheap,
and the dense loop costs approximately 15 µs for each merge. The option
(`RnnOptions::denseBelow`) is off by default. The tests use it to examine the dense
fallback.

---

## 7. A fast Ward edge weight

The Ward weight must be cheap, because the kernel calculates it for each pair. Two
designs are important:

- **No full division for each pair.** `2·nᵢ·nⱼ/(nᵢ+nⱼ)·d²` has one division. `__fdividef`
  (one reciprocal approximation and one multiplication, 2 ulp) replaces the full IEEE
  division.
- **No loads for each pair.** If the weight reads `size[i]` and `size[j]` for each of the
  64 outputs of a thread, the loads are a large part of the time. The compiler cannot
  move these loads out of the loop, because the `atomicMin` stores of the kernel are
  between them. Thus, the edge weight has two parts. The kernel loads the data of each
  point one time for each row and each column of a thread: 8 + 8 loads, not 128.

```cpp
struct WardWeight {
  const int *size;
  using Value = float;
  Value value(int cluster) const { return (float)size[cluster]; }       // one time for each row/column
  float operator()(float squared, Value sizeI, Value sizeJ) const {    // for each pair
    return __fdividef(2.0f * (sizeI * sizeJ) * squared, sizeI + sizeJ);
  }
};
```

With both designs, the Ward pass at d = 32 operates at almost the speed of the same pass
with plain distances.

---

## 8. Floating point

- **Symmetric weights.** Section 3 requires that weight(i, j) is equal to
  weight(j, i), bit for bit. `sizeI * sizeJ` and `sizeI + sizeJ` are commutative in IEEE
  arithmetic, the factor 2 is exact, and the squared distance adds `(a − b)²` over the
  features in the same sequence for both. The left-to-right form `(2·nᵢ)·nⱼ` is *not*
  symmetric when the product must be rounded (sizes above approximately 4,000).
- **Exact heights.** The approximate division only decides *which* cluster is nearest.
  The kernel calculates the recorded height again with a full division from the two
  centroids, in the same sequence of additions as the chain.
- **Centroids:** the incremental form `c_a + (c_b − c_a)·n_b/(n_a + n_b)`, the same as in
  the chain. Thus, a merge of duplicates leaves the centroid exactly in place, and zero
  heights stay zero.
- **Unstable data.** On some data, no float32 code can give the same tree as scipy. An
  example is 20,000 points at (i/n)² on a line. On this data, a change of 1e−7 in the
  input changes the top merge height of scipy itself (in float64) by 3% to 10%. Very
  small rounding differences change early merge decisions, and the remaining tree
  follows them. Ward is exact for exact arithmetic, but it is not stable. On usual data,
  the heights agree with scipy to approximately 1e−7 (relative).
- **Overflow.** Squared distances above the float32 maximum (3.4e38) become inf, and then
  NaN after a centroid update. The tree is then not correct. cuhc has two guards:
  - Before the calculation, Python examines an exact bound for each feature:
    Σ range_k² ≤ 3.4e38 for Euclidean, times n for Ward, times 1/alpha for HDBSCAN
    (`check_distance_range` in `_distances.py`). Data far from the origin is
    satisfactory. Only a very large *spread* causes a "rescale the data" error.
  - Before C++ makes a tree, it examines the merges (`mergesFormTree`: n − 1 merges,
    finite heights, and each merge joins two separate clusters). If the merges are not
    correct, it returns `kErrorNonFinite`, and it does not write outside the arrays.

---

## 9. Results

RTX 3080, 32 features, random normal data, full Ward tree. Random normal data does not
prune, so this table shows the rounds with the symmetric pass only:

| n | rnn | scikit-learn | Speed increase | Same labels (k = 10) |
|---|---|---|---|---|
| 2,000 | 0.003 s | 0.14 s | approximately 50 times | yes |
| 10,000 | 0.010 s | 4.3 s | approximately 410 times | yes |
| 20,000 | 0.038 s | 24.0 s | approximately 630 times | yes |
| 50,000 | 0.15 s | — | | |
| 100,000 | 0.53 s | — | | |
| 200,000 | 2.0 s | — | | |
| 500,000 | 12.5 s | — | | |
| 1,000,000 | 51.4 s | — | | |

The scikit-learn times change from run to run by up to approximately 40%.
`bench/bench_linkage.py --methods ward` gives these values.

For clustered data, spatial pruning makes the rounds faster again. With 1,000,000 points
in 20 blobs, Ward takes 1.1 s (2 features), 3.4 s (8 features) and 13.4 s (32 features).
Refer to `docs/all_pairs_passes.md`.
