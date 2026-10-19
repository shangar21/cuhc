# The pairwise distance kernel

This document explains three files:

- `src/distances.cuh`: the metrics.
- `src/distance_matrix.cuh`: the tiled distance kernel.
- `bench/bench_distance.cu`: the benchmark.

It tells what the code does, why it has this design, and how one thread moves through
the tiled kernel.

Contents:

1. [The problem](#1-the-problem)
2. [The accumulator pattern](#2-the-accumulator-pattern)
3. [A simple kernel, and why it is slow](#3-a-simple-kernel-and-why-it-is-slow)
4. [The tiled kernel](#4-the-tiled-kernel)
5. [One thread, with real numbers](#5-one-thread-with-real-numbers)
6. [Banks, coalesced access and edges](#6-banks-coalesced-access-and-edges)
7. [The benchmark](#7-the-benchmark)
8. [Where the time goes](#8-where-the-time-goes)
9. [How to write a metric](#9-how-to-write-a-metric)
10. [Questions and answers](#10-questions-and-answers)

---

## 1. The problem

The input is two arrays of points:

- `pointsA` has `numPointsA` rows and `numFeatures` columns.
- `pointsB` has `numPointsB` rows and `numFeatures` columns.

Both are row-major, with one point in each row. The output is the array `distances`, with
`numPointsA` rows and `numPointsB` columns:

```
distances[i][j] = metric(pointsA[i], pointsB[j])
```

For clustering, `pointsA` and `pointsB` are the same array `X`. Two separate inputs cost
nothing. They also let the kernel calculate a block of rows at a time
(`pointsA = X[r0:r1]`, `pointsB = X`).

The work is `numPointsA · numPointsB · numFeatures` feature steps. For 16,384 points and
64 features, this is 17 billion steps. Thus, the important values are the speed of each
step and the quantity of data that each step moves.

---

## 2. The accumulator pattern

### 2.1 The common shape of the metrics

Compare the formulas of the metrics:

```
sq-euclidean   sum_k (a_k - b_k)^2
euclidean      sqrt( sum_k (a_k - b_k)^2 )
manhattan      sum_k |a_k - b_k|
chebyshev      max_k |a_k - b_k|
minkowski-p    ( sum_k |a_k - b_k|^p )^(1/p)
cosine         1 - (sum_k a_k b_k) / sqrt( (sum_k a_k^2)(sum_k b_k^2) )
dot product    sum_k a_k * b_k          <-- matrix multiplication (GEMM) calculates this
```

All of them have the same structure:

```
dist(a, b) = finalize( combine over k of  f(a_k, b_k) )
```

- A value for each feature: `(a-b)^2`, `|a-b|` or `a*b`.
- An operation that puts the values into a running value: `+` or `max`.
- An optional final operation: `sqrt`, `^(1/p)` or `1 - x/y`.

This structure is a **fold** (also known as a reduce). A fold starts from an initial
value and changes it one time for each element. `std::accumulate` in C++ and
`functools.reduce` in Python are folds. Thus, each metric has three parts:

```
init()                    the initial value of the fold
accumulate(acc, a_k, b_k) one step of the fold
finalize(acc)             the final operation
```

### 2.2 Matrix multiplication is one example of the pattern

GEMM is the fold with `accumulate = acc + a*b`. Many years of work have made GEMM fast on
GPUs. Almost all of that work is about data movement: tiles, shared memory and
registers. It does not depend on the `+` and the `*`. Thus, if a metric has the GEMM
shape, a GEMM kernel can calculate it. Only the line that does `acc += a * b` changes.

Mathematics has a name for matrix multiplication with other pairs of operations: a
**semiring**. For example, shortest paths are matrix multiplication with `(+, min)`
instead of `(*, +)`. GraphBLAS is a library that uses this idea. For GPU distances:

- **RAFT/cuVS** (NVIDIA) calculates pairwise distances with a tiled kernel similar to
  GEMM and a small "distance operation" struct. This is almost the same design as cuhc.
- **KeOps** compiles user formulas into the same type of kernel.

### 2.3 Why the metric is three small functions

A simple interface is one function:

```cpp
float distance(const float* a, const float* b, int numFeatures);   // the loop is inside
```

This interface cannot be fast. If the metric contains the loop over the features, the
kernel cannot change the sequence of that loop. The tiled kernel (section 4) puts the
feature loop on the outside. It loads a slice of features for 128 A points and 128 B
points. Then each thread moves **64 different pairs** one step forward. This is possible
only if the metric gives the kernel one step at a time.

Thus, the metric tells **what one step does**, and the kernel decides **when each step
occurs**. The general rule is:

> Keep the part that changes (the mathematics of the metric, small and cheap) separate
> from the part that is expensive and does not change (the data movement). Make the part
> that changes a small function. Let the part that does not change control the loops.

### 2.4 Why a template, and not a function pointer

A function pointer or a `std::function` causes an indirect call for each step. The
compiler cannot inline the call, cannot unroll the loop around it, and cannot keep values
in registers across it.

The kernel is a template, `template <..., typename Metric>`. Thus, each metric gives a
different compiled kernel, and `metric.accumulate(...)` becomes a few inline
instructions. For sq-euclidean, the inner step is one `FADD` (the subtraction) and one
`FFMA` (fused multiply-add). For each slice of 8 features, the compiled kernel contains
512 of each, and only shared-memory loads in addition. Thus, a metric that you write has
the same speed as a built-in metric. After compilation, there is no difference.

The cost is that each metric must be compiled before use. For a metric that is defined
at run time (for example in Python), cuhc must compile the kernel at run time with NVRTC.
The pattern stays the same.

### 2.5 Details of the interface

- **`using Acc = ...`**: the accumulator can be more than one float. Cosine has three
  running sums (`a·b`, `|a|²`, `|b|²`), thus `Acc = float3`. A Jaccard metric can use
  `int2` (intersection, union).
- **Parameters are struct members.** `Minkowski{3.0f}` contains `p`. The kernel receives
  the struct by value, so the parameter goes into constant memory at no cost.
- **`DIST_HD` is `__host__ __device__ __forceinline__`.** The same struct operates on the
  CPU. The benchmark uses this to compare the GPU results with a CPU reference, without a
  second copy of the mathematics.
- **There is no `combine(acc1, acc2)`.** The fold of each output occurs in one thread, in
  feature sequence. Thus, `accumulate` does not have to be associative, and the kernel
  never merges two partial results. A kernel that divides the feature loop across threads
  ("split-K", for very large `numFeatures`) must also have a `combine` function.

### 2.6 Metrics that do not have this shape

A metric has this shape if step k uses only `a_k`, `b_k` and the running value. Many
metrics have it: Hamming, Canberra, Bray-Curtis (two sums), KL divergence, Jaccard and
correlation (written as sums, the same as cosine).

Some metrics do not have this shape, because their steps are related to each other.
Examples are Dynamic Time Warping (a 2D table for each pair), edit distance and the
Wasserstein distance (which sorts the values). These metrics must have different kernels.

---

## 3. A simple kernel, and why it is slow

The simplest kernel has one thread for each output:

```
thread (i, j):
    acc = init()
    for k in 0..numFeatures-1:  acc = accumulate(acc, A[i][k], B[j][k])
    distances[i][j] = finalize(acc)
```

The 32 threads of a warp have the same `i` and consecutive values of `j`. With 64 or
more features, this kernel is approximately 30 times slower than the tiled kernel. There
are three reasons:

1. **Two loads for each step.** Each step reads two floats from global memory. The
   mathematics is two instructions. The loads cost much more.
2. **A bad access pattern for B.** At one value of k, the 32 threads of a warp read
   `B[j][k], B[j+1][k], ...`. These values are `numFeatures` floats apart. Thus, one load
   instruction touches 32 different cache lines.
3. **No reuse.** Each point of B is necessary for all values of `i`, and each point of A
   for all values of `j`. The simple kernel reads them again each time. Only the L1 and
   L2 caches decrease this cost.

---

## 4. The tiled kernel

The tile loop is in `computeTileDistances`, and the thread layout is in `TileGeometry`
(both in `src/distance_matrix.cuh`). Kernels call `computeTileDistances` and then use
the results in different ways:

- `pairwiseDistanceKernel` writes the distances to memory.
- `nearestOtherComponentKernel` and `nearestOtherComponentSymmetricKernel` keep only the
  nearest point of each row (refer to `docs/boruvka_mst.md` and
  `docs/all_pairs_passes.md`).

`DefaultTile` selects this configuration for float accumulators:

| Template parameter | Value | Meaning |
|---|---|---|
| `TileRows × TileCols` | 128 × 128 | The tile of the output that one block calculates |
| `TileFeatures` | 8 | The features in shared memory at one time (a "slice") |
| `RowsPerThread × ColsPerThread` | 8 × 8 | The outputs of one thread |
| `ThreadGridRows × ThreadGridCols` | 16 × 16 | The threads of a block (128 / 8 in each direction), 256 in total |

The kernel uses three levels of memory. Each level has its own reuse:

```
 global memory (slow, large)    pointsA, pointsB, distances
        |   the block loads each float of A and B one time
        v
 shared memory (fast, 8 KB)     sharedA[8][132], sharedB[8][132]   <- one slice of 8 features
        |                                                             of 128 A and 128 B points
        |   16 threads read each float in shared memory
        v
 registers (fastest)            valuesA[8], valuesB[8], accumulators[8][8]   <- one thread
            each thread uses each valuesA[r] 8 times and each valuesB[c] 8 times
```

### 4.1 Level 1: the grid divides the output into tiles of 128 × 128

```
                         columns of the output (points of B)
              0        128       256       384       512 ...
           0  +---------+---------+---------+---------+
              | (0,0)   | (1,0)   | (2,0)   | (3,0)   |   block (blockIdx.x, blockIdx.y)
 rows of   128+---------+---------+---------+---------+
 the          | (0,1)   | (1,1)   | (2,1)   | (3,1)   |   tileFirstRow = 128 · blockIdx.y
 output   256 +---------+---------+---------+---------+   tileFirstCol = 128 · blockIdx.x
 (points      ...
  of A)
```

Block (bx, by) uses the A points `128·by` to `128·by + 127` and the B points `128·bx`
to `128·bx + 127`, with all their features. If there are many features, this is too much
data for shared memory. Thus, the block moves through the features in slices of 8.

### 4.2 Level 2: the feature loop, one slice of 8 features at a time

```
for firstFeature = 0, 8, 16, ... < numFeatures:
    (1) loadTileSlice two times: all 256 threads load
          sharedA[f][p] = pointsA[tileFirstRow + p][firstFeature + f]   p < 128, f < 8
          sharedB[f][p] = pointsB[tileFirstCol + p][firstFeature + f]   (1024 floats each)
    __syncthreads()        // wait until the slices are fully loaded
    (2) each thread does accumulateFeature(f) for f = 0..7 on its 8 × 8 outputs
    __syncthreads()        // wait until all threads stop the reads, before the next load
```

The slice is transposed: its layout is `[feature][point]`, not `[point][feature]`. In
step (2), a thread reads "feature f of several different points". With this layout, those
values are adjacent in one row of shared memory.

### 4.3 Level 3: each step is an outer product in registers

`accumulateFeature(f)` reads 8 values from `sharedA` and 8 values from `sharedB`:

```
valuesA[r] = sharedA[f][threadRow + 16*r]      r = 0..7     (feature f of 8 A points)
valuesB[c] = sharedB[f][threadCol + 16*c]      c = 0..7     (feature f of 8 B points)
```

Then it changes all 64 of its outputs:

```
               valuesB[0] valuesB[1] ... valuesB[7]
 valuesA[0]  [   step       step     ...   step   ]
 valuesA[1]  [   step       step     ...   step   ]
 ...                                                 step: accumulators[r][c] =
 valuesA[7]  [   step       step     ...   step   ]        accumulate(accumulators[r][c],
                                                                      valuesA[r], valuesB[c])
```

Sixteen reads from shared memory supply 64 steps. In GEMM, this is a "rank-1 update" (an
outer product). It is the most important method in the kernel. Each `valuesA[r]` is used
8 times from a register, which costs almost nothing.

### 4.4 The outputs of each thread

A thread does not own a contiguous square of 8 × 8 outputs. Thread
(threadRow, threadCol) owns **strided** rows and columns:

```
tile rows    threadRow, threadRow+16, threadRow+32, ..., threadRow+112
tile cols    threadCol, threadCol+16, threadCol+32, ..., threadCol+112
```

Think of the 128 × 128 tile as an 8 × 8 grid of patches, each 16 × 16. The 256 threads
cover one patch, one cell each. Each thread takes **the same cell in all 64 patches**.

```
 128 × 128 block tile                    one 16 × 16 patch
 +----+----+----+-- ... --+----+         threadCol: 0 1 2 3 4 5 ...      15
 |  P |  P |  P |         |  P |    threadRow 0   . . . . . . . . . . . . . . . .
 +----+----+----+         +----+              1   . . . . . . . . . . . . . . . .
 |  P |  P |                   |              2   . . . . . X . . . . . . . . . .  <- thread
 +----+----+    8 × 8 patches  |              ...                                  (2, 5)
 |               of 16 × 16    |             15   . . . . . . . . . . . . . . . .
 ...                                    thread (2,5) owns the X cell in all 64 patches
```

Section 6 gives the reasons. This layout prevents bank conflicts in shared memory, and
it makes the writes to `distances` coalesced.

---

## 5. One thread, with real numbers

Use 16,384 points, 64 features, block (blockIdx.x = 3, blockIdx.y = 1), and thread
`threadIdx.x = 37` in that block.

**The position of the thread:**

```
threadCol = 37 % 16 = 5          threadRow = 37 / 16 = 2
tileFirstRow = 128 · 1 = 128     tileFirstCol = 128 · 3 = 384
```

**The outputs of the thread** (`threadRow + 16·r` and `threadCol + 16·c`, plus the start
of the tile):

```
global rows  130, 146, 162, 178, 194, 210, 226, 242
global cols  389, 405, 421, 437, 453, 469, 485, 501
```

These are 64 distances. They are in `accumulators[8][8]`, and they start at `init()`.

**Slice firstFeature = 0, the load.** The A slice has 128 × 8 = 1024 floats, and there
are 256 threads. Thus, each thread loads 4 floats. In `loadTileSlice`, `loadIndex` has
the values 37, 293, 549 and 805. `pointInTile = loadIndex / 8` and
`featureInTile = loadIndex % 8`:

```
loadIndex  37  -> pointInTile =   4, featureInTile = 5   sharedA[5][4]   = pointsA[132][5]
loadIndex 293  -> pointInTile =  36, featureInTile = 5   sharedA[5][36]  = pointsA[164][5]
loadIndex 549  -> pointInTile =  68, featureInTile = 5   sharedA[5][68]  = pointsA[196][5]
loadIndex 805  -> pointInTile = 100, featureInTile = 5   sharedA[5][100] = pointsA[228][5]
```

The thread loads `sharedB` in the same way, from the B points `384 + pointInTile`.
Thread 37 loads mostly values that **other threads** use. For example, point 132 belongs
to the threads with `threadRow = 4`. All threads load together, and the load does not
depend on which thread calculates which output. For this reason, the `__syncthreads()`
is necessary.

**Slice firstFeature = 0, the calculation.** For `featureInTile = 0..7`:

```
valuesA = sharedA[f][2], sharedA[f][18], ..., sharedA[f][114]   // pointsA[130][f] ... pointsA[242][f]
valuesB = sharedB[f][5], sharedB[f][21], ..., sharedB[f][117]   // pointsB[389][f] ... pointsB[501][f]
accumulators[r][c] = accumulate(accumulators[r][c], valuesA[r], valuesB[c])   all 64 (r, c)
```

After this slice, each of the 64 accumulators contains features 0 to 7.

**Slices firstFeature = 8, 16, ..., 56:** the same operation. There are 8 slices for 64
features. Then `accumulators[r][c]` contains the full fold.

**The write:** `distances[130][389] = finalize(accumulators[0][0])`, ...,
`distances[242][501] = finalize(accumulators[7][7])`.

In total, thread 37 does 8 slices × 8 features × 16 = **1024 reads from shared memory**
for 64 outputs × 64 features = **4096 accumulate steps**.

### 5.1 The reuse

| Kernel | Loads for each accumulate step |
|---|---|
| Simple kernel (section 3) | 2 global loads |
| Tiled kernel, from global memory | 2048 floats for each slice of each block, for 128 × 128 × 8 = 131,072 steps: **1/64** |
| Tiled kernel, from shared memory | 16 reads for 64 steps: **1/4** |

### 5.2 Resources (from `cuobjdump --dump-resource-usage`)

- **Approximately 120 registers for each thread:** 64 accumulators, 16 for `valuesA` and
  `valuesB`, plus addresses and indices.
- **8448 bytes of shared memory:** 2 slices × 8 × 132 floats × 4 bytes.
- An RTX 3080 SM has 65,536 registers. 256 threads × 120 registers is approximately
  31,000. Thus, **2 blocks (16 warps) fit on each SM**. This occupancy is low, but it is
  satisfactory. Each thread has 64 independent accumulators, which is sufficient work to
  keep the arithmetic units busy.
- **Cosine** has `Acc = float3`. With 64 outputs, 192 registers are necessary for the
  accumulators only. They do not fit, and they go into slow local memory. Thus,
  `DefaultTile` gives cosine 4 × 4 outputs for each thread (a tile of 64 × 64, 77
  registers). This is the `sizeof(Acc) > sizeof(float)` test.

---

## 6. Banks, coalesced access and edges

### 6.1 Coalesced global loads

In `loadTileSlice`, `pointInTile = loadIndex / 8` and `featureInTile = loadIndex % 8`.
Thus, **consecutive threads read consecutive features of the same point**. Warp 0
(threads 0 to 31) reads points 0 to 3, features 0 to 7 of each. These are 4 contiguous
pieces of 32 bytes: 4 full memory sectors, and all bytes are used. The other mapping
(`pointInTile = loadIndex % 128`) causes each thread to read a different point, with
`numFeatures` floats between them.

The kernel calculates distances between points at all positions of the matrix, but this
does not change the coalesced access. Each block reads only two **contiguous ranges of
points**: 128 consecutive rows of A and 128 of B. In each row, it reads consecutive
features. The rows are different for each block, but each load is a contiguous read.

### 6.2 Shared-memory banks and the `Padding`

Shared memory has 32 **banks** of 4-byte words. Word w is in bank `w % 32`. If two
threads of a warp access different addresses in the same bank, the accesses occur one
after the other. This is a "bank conflict". Reads of the same address do not conflict
(broadcast).

**The stores (the transpose).** Thread t of warp 0 writes `sharedA[f][p]`, with
`f = t % 8` and `p = t / 8`. If the row length is 128, the word address is `128·f + p`,
and the bank is `(128·f + p) % 32 = p`. Thus, 32 threads use only 4 banks, which is an
8-way conflict. With a row length of `128 + Padding = 132`:

```
bank = (132*f + p) % 32 = (4*f + p) % 32,   f = 0..7, p = 0..3   -> 0..31, all different
```

The general form is `Padding = 32 / TileFeatures`. The row stride becomes
`32 / TileFeatures` (modulo 32). Thus, the `TileFeatures` different features are
`32 / TileFeatures` banks apart.

**The reads in the calculation step.** Warp 0 has threads 0 to 31: `threadRow` is 0 or
1, and `threadCol` is 0 to 15.

- `sharedB[f][threadCol + 16·c]`: 16 consecutive addresses, each read by 2 threads
  (broadcast). There is no conflict.
- `sharedA[f][threadRow + 16·r]`: 2 different addresses (broadcast).

With contiguous ownership (`threadCol·8 + c`), the 16 threads read addresses 0, 8, 16,
..., 120. These addresses are in banks 0, 8, 16, 24, 0, 8, .... This is a 4-way conflict
on each read of B. **This is the first reason for the strided ownership.**

### 6.3 Coalesced writes to the output

At the end, for one value of (r, c), warp 0 writes 2 rows (`threadRow` 0 and 1) of 16
consecutive columns. These are two pieces of 64 bytes, which are full sectors. With
contiguous ownership, the 16 threads of a row write addresses 8 floats apart. **This is
the second reason.**

### 6.4 Edges: sizes that are not a multiple of the tile

- **Points after `numPointsA` or `numPointsB`:** the loads write `0.0f` into shared
  memory, and the kernel does not do the final write. The kernel discards the values for
  those positions.
- **Features after `numFeatures`:** the loads also write `0.0f`, but the kernel **must
  not** call `accumulate` for them. For each built-in metric, `accumulate(acc, 0, 0)`
  does not change the value, but this is not true for all metrics. For example, a metric
  that counts the features with a == b counts the padding as matches. Thus, the last
  partial slice uses a plain loop to `numFeatures − firstFeature`, instead of the
  unrolled loop of 8. The full slices keep the fast unrolled loop.

### 6.5 The two `__syncthreads()`

- **After the load:** a thread must not read `sharedA` or `sharedB` until all threads
  have written their part. For example, thread 37 uses values that other threads load.
- **After the calculation:** a thread must not start the load of the next slice while a
  slower thread reads the current slice.

If you remove one of them, the results are sometimes incorrect. The errors depend on the
timing.

### 6.6 `__launch_bounds__(256)`

This tells the compiler the block size. Thus, the compiler can plan the registers for
exactly 256 threads. Without it, the compiler must assume up to 1024 threads, and it can
decrease the registers for each thread.

---

## 7. The benchmark

`./build/bench_distance [numPoints=16384] [numFeatures=64] [metric]` measures each metric
(or only the named metric). It compares 16 rows at equal intervals with the CPU result,
which uses the same metric struct on the host. It also shows two **lower limits**:

- **memset**: a write of `numPoints²` floats, with no mathematics. No kernel that writes
  the full matrix can be faster than this.
- **cuBLAS SGEMM** (FP32, no tensor cores): `X Xᵀ`. This is the same loop with
  `accumulate = acc + a*b`, in the best-tuned NVIDIA code.

Results on an RTX 3080, 16,384 points, in milliseconds. The "simple" column is the kernel
of section 3:

| Features | memset | SGEMM | euclidean, simple → tiled | cosine | minkowski p=3 |
|---|---|---|---|---|---|
| 8 | 1.33 | — | 4.9 → **1.5** | 1.6 | 11.9 |
| 64 | 1.33 | 2.1 | 131 → **4.4** | 7.7 | 36 |
| 512 | — | — | 1077 → **32** | 59 | 239 |

1. **With few features, the output write is the limit.** With 8 features, the kernel
   takes 1.5 ms, and the memset limit is 1.33 ms. The mathematics costs almost nothing.
   The write of a 1 GB matrix costs much time.
2. **Cosine is approximately 2 times slower** than euclidean at 64 features or more. It
   has 3 operations for each step, and it uses the smaller 4 × 4 tile for each thread,
   which gives less reuse. If the norms of the points were calculated one time before
   the kernel, cosine would use 1 FMA for each step.
3. **Minkowski:**
   - The full `powf` function is a long sequence of instructions with many branches, for
     special values. The tiled kernel puts it inline 8 × 8 × 8 = 512 times. The code is
     then too large for the instruction cache. With `powf`, the tiled kernel takes 133 ms
     at 64 features, which is not faster than the simple kernel.
   - Thus, `distances.cuh` calculates `x^p = 2^(p · log2 x)` with the hardware
     special-function units, `__log2f` and `exp2f`, with a guard for x = 0. This is 3.4
     times faster.
   - On Ampere, these units operate at 1/8 of the rate of usual FP32 operations, and each
     step uses 2 of them. Thus, the limit is approximately 8 times slower than euclidean.
     The measurement at 512 features agrees.
   - With 8 features, the simple kernel (5.2 ms) is faster than the tiled kernel
     (approximately 11 ms). A possible cause is that the low occupancy of the tiled
     kernel cannot hide the latency of the special-function units. This is not measured
     with a profiler at this time.

### 7.1 The GPU architecture in CMake

`set(CMAKE_CUDA_ARCHITECTURES ...)` must be before `project()`. If the value is not set,
`project()` uses the nvcc default (sm_75). Then a subsequent `if(NOT DEFINED ...)` has no
effect, and the binary contains only sm_75 code. The RTX 3080 (sm_86) must then compile
the PTX at run time. Driver 580 (CUDA 13.0) does not accept PTX from nvcc 13.2, and gives
the error *"the provided PTX was compiled with an unsupported toolchain"*. To see the code
in a binary, use `cuobjdump --list-elf build/bench_distance`.

---

## 8. Where the time goes

On this computer, only the root user can read the hardware counters. Thus, this section
uses the compiled machine code (SASS) and comparisons with cuBLAS.
`sudo scripts/profile_distance.sh` writes full Nsight Compute reports into
`build/profile/`.

### 8.1 Measurements

**The compiled code** (sq-euclidean, `cuobjdump -sass`): for each slice of 8 features,
512 `FADD`, 512 `FFMA` and 128 `LDS` (shared-memory loads). This is 8 mathematical
instructions for each shared load. The inner loop contains no other instructions. There
are no register spills. All memory instructions are scalar (4-byte `LDS`, `LDG`, `STG`).

**Comparison with the hardware** (8192 points; sq-euclidean is 2 instructions for each
step, SGEMM is 1):

| Features | cuBLAS SGEMM | cuhc | cuhc instructions per second, compared with cuBLAS |
|---|---|---|---|
| 512 | 3.60 ms (19.1 TFLOP/s) | 9.19 ms | 78% |
| 2048 | 16.2 ms (17.0 TFLOP/s) | 34.2 ms | 95% |

cuBLAS SGEMM operates at approximately 19 TFLOP/s on the RTX 3080, which is
approximately 65% of the 29.8 TFLOP/s specification. This is the practical FP32 limit of
this GPU. At 2048 features, the generic kernel of cuhc issues mathematical instructions
at 95% of the cuBLAS rate.

### 8.2 Uncoalesced access

The design prevents uncoalesced access (section 6):

- The global loads use full sectors of 32 bytes.
- The stores are full rows of 64 bytes for each half-warp.
- Shared memory has padding, and the read patterns have no conflicts.

To make sure, examine the Memory Workload Analysis section of a profile. Look at the
sectors for each request and at the shared-memory bank conflicts.

### 8.3 Warp shuffles

**Warp shuffles do not help the accumulation.** A shuffle (`__shfl_sync`) moves values
between the 32 lanes of a warp. It is useful when **several lanes have partial results of
the same output**, which must be combined. In this kernel, the fold of each output is in
the registers of one thread. Thus, there is nothing to combine.

A different kernel could divide the features of one pair across the lanes of a warp, and
combine them with shuffles at the end. This removes the register reuse of the 8 × 8 outer
product, which is the source of the speed. Shuffles in place of shared memory, to move
`valuesA` and `valuesB`, do not help either. A shuffle costs approximately the same as a
shared load, and shared memory is still necessary to move data between warps.

**Shuffles help to find the nearest point of each row.** The 16 threads that share an
output row are in the same warp. Thus, a few `__shfl_xor_sync` steps combine their
minimums. The kernels that do not write the matrix use this method
(`nearestOtherComponentKernel`, refer to `docs/boruvka_mst.md`).

### 8.4 Possible improvements, largest first

1. **Do not write the matrix.** With few features, the write of n² floats is almost all
   of the time. A reduction in the kernel (the minimum of each row, the k nearest points,
   counts below a threshold) removes that time and the n² memory. The single-linkage,
   Ward and HDBSCAN algorithms use this method (`docs/all_pairs_passes.md`).
2. **Fewer instructions for each step.** With many features, the kernel is at the
   practical FP32 rate. Thus, only less work for each step can make it faster:
   - Euclidean and cosine distances can be written as `|a|² + |b|² − 2a·b`, which is a
     real GEMM. A real GEMM can use the **tensor cores** (TF32 or FP16, with the
     `mma.sync` warp-level matrix instructions), at several times the FP32 rate.
   - For other metrics, half precision with packed `half2` operations does 2 steps for
     each instruction.
3. **Load and calculate at the same time** (for a medium number of features, possibly
   5% to 15%). Each slice does load → wait → calculate → wait. On Ampere, `cp.async` can
   load slice s+1 into a second shared buffer while the kernel calculates slice s
   ("double buffering"). Vector loads (`float4`) decrease the number of load instructions
   by 4 times. The comparison with cuBLAS shows that the possible increase is small.

---

## 9. How to write a metric

This example is the Canberra distance, `sum_k |a_k − b_k| / (|a_k| + |b_k|)`:

```cpp
struct Canberra {
  using Acc = float;
  DIST_HD Acc init() const { return 0.0f; }
  DIST_HD Acc accumulate(Acc acc, float a, float b) const {
    const float denom = fabsf(a) + fabsf(b);
    return denom > 0.0f ? acc + fabsf(a - b) / denom : acc;
  }
  DIST_HD float finalize(Acc acc) const { return acc; }
};

dist::pairwiseDistance(devicePoints, devicePoints, deviceDistances, numPoints, numPoints,
                       numFeatures, Canberra{});
```

To measure and examine the metric, add `run("canberra", Canberra{});` to the benchmark.

Rules:

- **Keep `accumulate` short, with few branches.** The kernel puts it inline 512 times.
  Long library functions (`powf`, `logf` and `expf` in full precision) make the code too
  large for the instruction cache. If a small decrease in precision is satisfactory, use
  the hardware versions (`__logf`, `__expf`, `exp2f`, `__fdividef`).
- **Keep `Acc` small.** One float keeps the 8 × 8 tile. A larger `Acc` automatically
  receives the 4 × 4 tile.
- **Move the work for one point out of the loop.** Calculate values that depend on only
  one point (norms, means) one time for each point before the kernel, not one time for
  each pair. Cosine does not obey this rule at this time (section 7).
- **Tell the kernels what the metric permits.** Two traits in `src/distances.cuh` let
  kernels do less work. They are `false` for a new metric. Set them only if they are
  true:
  - `metrics::IsSymmetric`: d(x, y) is equal to d(y, x), bit for bit. Then the kernels
    calculate each pair one time only.
  - `metrics::HasBoxBound`: the distance uses only the per-feature gaps |x_k − y_k|, and
    it does not decrease when a gap increases. Then the kernels can skip far tiles
    (`docs/all_pairs_passes.md`).

---

## 10. Questions and answers

1. Thread `threadIdx.x = 200` in block (0, 0): which rows and columns of the output does
   it own?
2. Which `sharedA` entries does the same thread *store* in `loadTileSlice`?
3. Why must `TileRows` be a multiple of 32 for the `Padding` method?
4. With 20 features, how many slices are there, and which slice uses the slow loop?
5. Why can the tiled kernel not calculate Dynamic Time Warping?
6. Why is the tiled kernel approximately 30 times faster than the simple kernel with 64
   features, but only approximately 3 times faster with 8 features?

<details>
<summary>Answers</summary>

1. `threadCol = 200 % 16 = 8` and `threadRow = 200 / 16 = 12`. The rows are 12, 28, 44,
   ..., 124. The columns are 8, 24, 40, ..., 120.
2. `loadIndex` = 200, 456, 712 and 968. Thus, (pointInTile, featureInTile) = (25, 0),
   (57, 0), (89, 0) and (121, 0). The thread stores `sharedA[0][25]`, `sharedA[0][57]`,
   `sharedA[0][89]` and `sharedA[0][121]`.
3. The bank of `sharedA[f][p]` is `(f · (TileRows + Padding) + p) % 32`. Only if
   `TileRows % 32 == 0` is this equal to `(f · Padding + p) % 32`. This makes the 32
   addresses go to 32 different banks.
4. `firstFeature` = 0, 8 and 16: 3 slices. For the last slice, 16 + 8 > 20. Thus, it uses
   the plain loop over features 16 to 19.
5. In DTW, the value for features (k, l) depends on adjacent cells of a 2D table, not on
   (a_k, b_k) and one running value. Thus, DTW is not a fold over the features.
6. With 8 features, both kernels use most of their time for the output write (the
   1.33 ms limit). The tiled kernel makes only the mathematics and the loads faster, and
   with few features, these are a small part of the time.

</details>
