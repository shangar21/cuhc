# Nearest-neighbour chain: Ward without the distance matrix

This document explains `src/nn_chain.cuh`. The nearest-neighbour chain calculates exact
Ward linkage in O(n·d) memory. cuhc uses it in two ways:

- `algorithm="nn_chain"` in `cuhc.linkage` and `AgglomerativeClustering`.
- The fallback of the Ward rounds (`docs/rnn_ward.md`). If the rounds make too little
  progress and the dense matrix does not fit, the chain merges the remaining clusters.

For Ward linkage, `algorithm="auto"` uses the rounds, which are faster (section 5).

Contents:

1. [Ward uses only centroids](#1-ward-uses-only-centroids)
2. [The nearest-neighbour chain](#2-the-nearest-neighbour-chain)
3. [One cooperative kernel for the full chain](#3-one-cooperative-kernel-for-the-full-chain)
4. [Floating-point details](#4-floating-point-details)
5. [Results, and comparison with the other algorithms](#5-results-and-comparison-with-the-other-algorithms)
6. [Limits](#6-limits)

---

## 1. Ward uses only centroids

Ward merges the two clusters whose merge gives the smallest increase of the variance in
the clusters. The distance between clusters a and b depends only on their sizes and
centroids:

```
ward(a, b) = sqrt( 2 · n_a · n_b / (n_a + n_b) ) · |c_a − c_b|
```

For two single points, this is their Euclidean distance. scipy and scikit-learn give the
same heights. The Lance-Williams Ward formula in `docs/dense_linkage.md` gives exactly
these values.

Thus, no matrix of pairs is necessary. The algorithm keeps one centroid and one size for
each cluster (O(n·d) memory). It calculates each cluster distance when necessary, in
O(d).

---

## 2. The nearest-neighbour chain

The dense algorithm finds the nearest pair of all clusters at each merge. For this, it
must have the full matrix, or do a full O(n²) scan each time. The chain only asks "which
cluster is nearest to *this* cluster?". This costs one O(n·d) scan:

```
chain = [any cluster]
repeat until one cluster is left:
    a = top of chain
    b = nearest active cluster to a        (ties: the cluster below a in the chain wins)
    if b is the cluster below a:           a and b are the nearest cluster of each other:
        remove both, merge them               "reciprocal nearest neighbours"
    else:
        add b to the chain
```

**Why reciprocal nearest neighbours give the same tree as the nearest pair.** The
algorithm uses reducibility, which Ward has (`docs/dense_linkage.md` section 3). A merge
of a and b never makes a cluster that is nearer to another cluster than a or b was.
Thus, if a and b are the nearest cluster of each other, no subsequent merge can make a
cluster that is nearer to them. The algorithm that merges the nearest pair first also
merges a and b, at the same height. The chain gives the merges in a different
*sequence*. Thus, cuhc sorts them by height at the end, the same as scipy.

**Why it stops.** The distances along the chain never increase, because each new cluster
is the nearest to the cluster before it. With the tie rule "the cluster below wins", the
chain cannot make a cycle. Thus, it ends at a reciprocal pair. Each cluster goes into the
chain a small number of times. This gives approximately 3n steps in total, and each step
is one scan. The total is O(n²·d) work and O(n·d) memory.

**Not for centroid or median linkage.** These linkages are not reducible. A merge of two
clusters can put the new centroid *nearer* to a third cluster. Then a merge of reciprocal
nearest neighbours is not correct. scipy also does not use the chain for them. cuhc does
not supply them.

---

## 3. One cooperative kernel for the full chain

Each step is "scan all active clusters, then a small quantity of sequential logic". With
a CPU loop, each step must start a kernel and copy the result back. This costs
approximately 10 to 20 µs for each step, which is most of the time at small and medium n.
Thus, the full chain operates in **one kernel** (`nnChainWardKernel`), which starts with
`cudaLaunchCooperativeKernel`:

```
loop:
    each block: copy the centroid of the current cluster to shared memory;
                scan its own contiguous range of slots -> the nearest of the block (blockMinLoc)
    grid.sync()                        <- all blocks wait here
    block 0:    combine the block results -> the nearest of all;
                add it to the chain, or merge (change centroid and size, record the merge,
                select the next current cluster)
    grid.sync()
```

- **`grid.sync()`** is a barrier for *all* blocks of the grid, not only one block. This
  is possible only if all blocks are on the GPU at the same time. A cooperative launch
  makes sure of this. For this reason, the grid size comes from the occupancy API
  (blocks for each SM × number of SMs).
- **Slots** operate as in the dense linkage. A merge of slots a and b puts the cluster in
  slot `min(a, b)`, and stops the other slot (`size = 0`). The cluster in a slot always
  contains the start cluster of that slot. Thus, the merges are recorded by
  representative points. The CPU makes the tree with the same union-find code as the MST
  algorithm (`treeFromOrderedMerges`).
- **Ties go to the previous cluster of the chain.** That cluster has the index −1 in the
  `MinLoc` reduction. Thus, it wins each tie.
- **A new chain:** when the chain is empty, it starts again at the first active slot. The
  scan position only moves forward. Thus, this costs O(n) in total.

`nnChainWardMerges` can also start from clusters that are larger than one point, with a
centroid and a size for each slot. The Ward rounds use this for their fallback.

---

## 4. Floating-point details

- **Duplicate points must merge at exactly 0.** The formula
  `(n_a·c_a + n_b·c_b) / (n_a + n_b)` does not give exactly c in float when
  c_a = c_b = c. The merged duplicates then move by one ulp, and later "zero" distances
  become approximately 1e−8. The incremental form `c_a + (c_b − c_a)·n_b/(n_a + n_b)`
  calculates c_b − c_a = 0 exactly. Thus, the centroid does not move. This form is also
  more accurate in general.
- **A parent must not come before its children.** The tree is made from the merges in
  the sequence of their height. The union-find requires that each merge comes after the
  merges that made its clusters. In exact arithmetic, Ward heights never decrease up the
  tree. In float, rounding can put a parent one ulp below a child. Thus,
  `orderChainMerges` first increases the height of each merge to the height of its
  children (a change of one ulp or less). Then it does a *stable* sort. Equal heights
  keep the chain sequence, which always has the children first.

The heights agree with scikit-learn to 1e-4 relative. The tests use sizes from 2 to
6,000, and 1, 7 and 64 features.

---

## 5. Results, and comparison with the other algorithms

RTX 3080, 32 features, random normal data, full Ward tree:

| n | nn_chain | dense | rnn rounds | scikit-learn |
|---|---|---|---|---|
| 2,000 | 0.053 s | 0.027 s | 0.003 s | 0.16 s |
| 10,000 | 0.28 s | 0.17 s | 0.010 s | 4.3 s |
| 20,000 | 0.68 s | 0.47 s | 0.038 s | 21.5 s |
| 50,000 | 2.45 s | does not fit* | 0.15 s | — |
| 100,000 | 8.5 s | — | 0.53 s | — |
| 200,000 | 29 s | — | 2.0 s | — |
| 500,000 | 177 s | — | 12.5 s | — |

\* 2.3 GB of the 12 GB was used by other processes (the display) during this run. Thus,
the 10 GB matrix did not fit. The rnn column uses the symmetric pass. Random normal data
does not prune. For clustered data, spatial pruning makes the rounds faster again
(`docs/all_pairs_passes.md`).

- **At small n, the barriers are most of the time.** The chain has approximately 3n
  steps, each with two barriers for the full grid. At 20,000 points, this is
  approximately 15 µs for each step.
- **At large n, the memory bandwidth is the limit.** At 500,000 points, each step reads
  the 64 MB centroid array in approximately 118 µs. This is approximately 540 GB/s, and
  the maximum of the GPU is approximately 760 GB/s. A faster scan is not possible. Only a
  *smaller* scan can be faster.
- **The rounds are faster because of the access pattern.** Each chain step compares one
  cluster with all others. Thus, each loaded centroid gives only one distance. The rounds
  find the nearest cluster of all clusters together, with the tiled kernel. That kernel
  uses each loaded centroid 128 times, and its limit is the arithmetic, not the memory
  bandwidth. Refer to `docs/rnn_ward.md`.

---

## 6. Limits

- **Speed:** the chain is limited by the memory bandwidth (section 5). Bounding boxes for
  ranges of slots could let each step skip far ranges. Each block scans one contiguous
  range of slots, so a test for each range is possible. This is not implemented, because
  the rounds are faster.
- **Ward only:** the chain operates only for reducible linkages. cuhc uses it only for
  Ward with the Euclidean metric.
