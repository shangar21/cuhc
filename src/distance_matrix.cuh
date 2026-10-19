#pragma once

#include "distances.cuh"
#include "reductions.cuh"

#include <cstddef>
#include <cuda_runtime.h>

// Tiled pairwise distances between the rows of pointsA (numPointsA x
// numFeatures) and pointsB (numPointsB x numFeatures), both row-major.
//
// The tile loop (computeTileDistances) is shared by kernels that differ only in
// what they do with the distances afterwards (the "epilogue"):
//   pairwiseDistanceKernel        writes the full matrix
//   nearestOtherComponentKernel   keeps, per row, only the nearest point in a
//                                 different component (for Borůvka's MST)
//
// See docs/distance_matrix.md for a walkthrough of what each thread does.

namespace dist {

// Shape of a block's tile and how its threads split it.
//  - each block owns a TileRows x TileCols tile of the output and walks over
//    the features in slices of TileFeatures,
//  - its threads form a ThreadGridRows x ThreadGridCols grid; each computes
//    RowsPerThread x ColsPerThread outputs,
//  - thread (threadRow, threadCol) owns tile rows threadRow + r * ThreadGridRows
//    and tile columns threadCol + c * ThreadGridCols. Strided (rather than
//    contiguous) ownership keeps shared-memory reads conflict-free and the
//    final writes coalesced.
template <int TileRows_, int TileCols_, int TileFeatures_, int RowsPerThread_, int ColsPerThread_>
struct TileGeometry {
  static constexpr int TileRows = TileRows_;
  static constexpr int TileCols = TileCols_;
  static constexpr int TileFeatures = TileFeatures_;
  static constexpr int RowsPerThread = RowsPerThread_;
  static constexpr int ColsPerThread = ColsPerThread_;
  static constexpr int ThreadGridRows = TileRows / RowsPerThread;
  static constexpr int ThreadGridCols = TileCols / ColsPerThread;
  static constexpr int ThreadsPerBlock = ThreadGridRows * ThreadGridCols;
  // Shared rows are padded so that the transposed stores in loadTileSlice hit
  // 32 different banks: the row stride becomes 32/TileFeatures (mod 32).
  static constexpr int Padding = 32 / TileFeatures;
  static_assert(32 % TileFeatures == 0 && TileRows % 32 == 0 && TileCols % 32 == 0,
                "tile shape");

  static __device__ __forceinline__ int threadRow() { return threadIdx.x / ThreadGridCols; }
  static __device__ __forceinline__ int threadCol() { return threadIdx.x % ThreadGridCols; }
  // Position within the tile of this thread's r-th row / c-th column.
  static __device__ __forceinline__ int rowInTile(int r) { return threadRow() + r * ThreadGridRows; }
  static __device__ __forceinline__ int colInTile(int c) { return threadCol() + c * ThreadGridCols; }
  // For kernels with one tile per block (grid from grid()): the block's tile
  // and the global output row/column of this thread's r-th row / c-th column.
  static __device__ __forceinline__ int tileFirstRow() { return blockIdx.y * TileRows; }
  static __device__ __forceinline__ int tileFirstCol() { return blockIdx.x * TileCols; }
  static __device__ __forceinline__ int outputRow(int r) { return tileFirstRow() + rowInTile(r); }
  static __device__ __forceinline__ int outputCol(int c) { return tileFirstCol() + colInTile(c); }

  static dim3 grid(int numPointsA, int numPointsB) {
    return dim3((numPointsB + TileCols - 1) / TileCols, (numPointsA + TileRows - 1) / TileRows);
  }
};

// 16 x 16 threads. Metrics with a wider accumulator get fewer outputs per
// thread so the accumulators still fit in registers.
template <typename Metric> struct DefaultTile {
  static constexpr int OutputsPerThreadSide = sizeof(typename Metric::Acc) > sizeof(float) ? 4 : 8;
  using type = TileGeometry<16 * OutputsPerThreadSide, 16 * OutputsPerThreadSide, 8,
                            OutputsPerThreadSide, OutputsPerThreadSide>;
};

// Copies features [firstFeature, firstFeature + TileFeatures) of points
// [firstPoint, firstPoint + TilePoints) into shared memory, transposed to
// sharedTile[feature][point]. All threads of the block take part.
//
// Consecutive threads read consecutive features of the same point, so global
// reads are coalesced. Out-of-range entries are filled with 0 (never used).
template <int TilePoints, int TileFeatures, int RowLength, int ThreadsPerBlock>
__device__ __forceinline__ void loadTileSlice(float (*sharedTile)[RowLength],
                                              const float *__restrict__ points, int numPoints,
                                              int numFeatures, int firstPoint, int firstFeature) {
#pragma unroll
  for (int loadIndex = threadIdx.x; loadIndex < TilePoints * TileFeatures;
       loadIndex += ThreadsPerBlock) {
    const int pointInTile = loadIndex / TileFeatures;
    const int featureInTile = loadIndex % TileFeatures;
    const int point = firstPoint + pointInTile;
    const int feature = firstFeature + featureInTile;
    const bool inRange = point < numPoints && feature < numFeatures;
    sharedTile[featureInTile][pointInTile] =
        inRange ? points[(size_t)point * numFeatures + feature] : 0.0f;
  }
}

// Computes the distances for this thread's outputs of the block's tile, same
// structure as a classic SGEMM:
//  - each slice of A points and B points is staged in shared memory once and
//    reused by every thread in the block,
//  - each thread folds features into RowsPerThread x ColsPerThread
//    accumulators in registers, so every value it reads from shared memory is
//    used several times.
// The only difference from GEMM is that "a*b + acc" is replaced by
// metric.accumulate. The tile starts at (tileFirstRow, tileFirstCol);
// tileDistances[r][c] receives the distance for row tileFirstRow +
// Tile::rowInTile(r) and column tileFirstCol + Tile::colInTile(c). Entries
// outside the matrix are garbage and must be ignored by the caller.
// Contains __syncthreads(): every thread of the block must call it.
template <typename Tile, typename Metric>
__device__ __forceinline__ void
computeTileDistances(const float *__restrict__ pointsA, const float *__restrict__ pointsB,
                     int numPointsA, int numPointsB, int numFeatures, const Metric &metric,
                     int tileFirstRow, int tileFirstCol,
                     float (&tileDistances)[Tile::RowsPerThread][Tile::ColsPerThread]) {
  constexpr int RowsPerThread = Tile::RowsPerThread, ColsPerThread = Tile::ColsPerThread;
  constexpr int TileFeatures = Tile::TileFeatures;
  using Accumulator = typename Metric::Acc;

  // One slice of features for the tile's A points and B points, feature-major.
  __shared__ float sharedA[TileFeatures][Tile::TileRows + Tile::Padding];
  __shared__ float sharedB[TileFeatures][Tile::TileCols + Tile::Padding];

  const int threadRow = Tile::threadRow(), threadCol = Tile::threadCol();

  Accumulator accumulators[RowsPerThread][ColsPerThread];
#pragma unroll
  for (int r = 0; r < RowsPerThread; ++r)
#pragma unroll
    for (int c = 0; c < ColsPerThread; ++c)
      accumulators[r][c] = metric.init();

  // Folds one feature into all of this thread's outputs: an "outer product"
  // of RowsPerThread A values with ColsPerThread B values.
  auto accumulateFeature = [&](int featureInTile) {
    float valuesA[RowsPerThread], valuesB[ColsPerThread];
#pragma unroll
    for (int r = 0; r < RowsPerThread; ++r)
      valuesA[r] = sharedA[featureInTile][threadRow + r * Tile::ThreadGridRows];
#pragma unroll
    for (int c = 0; c < ColsPerThread; ++c)
      valuesB[c] = sharedB[featureInTile][threadCol + c * Tile::ThreadGridCols];
#pragma unroll
    for (int r = 0; r < RowsPerThread; ++r)
#pragma unroll
      for (int c = 0; c < ColsPerThread; ++c)
        accumulators[r][c] = metric.accumulate(accumulators[r][c], valuesA[r], valuesB[c]);
  };

  for (int firstFeature = 0; firstFeature < numFeatures; firstFeature += TileFeatures) {
    loadTileSlice<Tile::TileRows, TileFeatures, Tile::TileRows + Tile::Padding,
                  Tile::ThreadsPerBlock>(sharedA, pointsA, numPointsA, numFeatures, tileFirstRow,
                                         firstFeature);
    loadTileSlice<Tile::TileCols, TileFeatures, Tile::TileCols + Tile::Padding,
                  Tile::ThreadsPerBlock>(sharedB, pointsB, numPointsB, numFeatures, tileFirstCol,
                                         firstFeature);
    __syncthreads(); // slice fully loaded before anyone reads it

    // Don't accumulate the zero padding past numFeatures: a user metric need
    // not treat (0, 0) as a no-op.
    if (firstFeature + TileFeatures <= numFeatures) {
#pragma unroll
      for (int featureInTile = 0; featureInTile < TileFeatures; ++featureInTile)
        accumulateFeature(featureInTile);
    } else {
      for (int featureInTile = 0; featureInTile < numFeatures - firstFeature; ++featureInTile)
        accumulateFeature(featureInTile);
    }
    __syncthreads(); // everyone done reading before the next slice overwrites it
  }

#pragma unroll
  for (int r = 0; r < RowsPerThread; ++r)
#pragma unroll
    for (int c = 0; c < ColsPerThread; ++c)
      tileDistances[r][c] = metric.finalize(accumulators[r][c]);
}

// distances[i][j] = metric(pointsA[i], pointsB[j]), a numPointsA x numPointsB
// row-major matrix.
template <typename Tile, typename Metric>
__global__ void __launch_bounds__(Tile::ThreadsPerBlock)
    pairwiseDistanceKernel(const float *__restrict__ pointsA, const float *__restrict__ pointsB,
                           float *__restrict__ distances, int numPointsA, int numPointsB,
                           int numFeatures, Metric metric) {
  float tileDistances[Tile::RowsPerThread][Tile::ColsPerThread];
  computeTileDistances<Tile>(pointsA, pointsB, numPointsA, numPointsB, numFeatures, metric,
                             Tile::tileFirstRow(), Tile::tileFirstCol(), tileDistances);

#pragma unroll
  for (int r = 0; r < Tile::RowsPerThread; ++r) {
    const int row = Tile::outputRow(r);
    if (row >= numPointsA)
      continue;
#pragma unroll
    for (int c = 0; c < Tile::ColsPerThread; ++c) {
      const int col = Tile::outputCol(c);
      if (col < numPointsB)
        distances[(size_t)row * numPointsB + col] = tileDistances[r][c];
    }
  }
}

// An edge weight turns the distance between points i and j into the weight
// of edge (i, j). It's split in two so the kernel can load per-point data once
// per row/column a thread owns rather than once per pair:
//   using Value = ...;
//   Value value(int point) const;                         per-point data
//   float operator()(float distance, Value valueI, Value valueJ) const;
// The weight must be symmetric in (i, j): weight(d, value(i), value(j)) ==
// weight(d, value(j), value(i)). The symmetric kernel below computes each
// pair once and uses the result for both directions.

// Edge weight = the metric distance itself (plain single linkage).
struct DistanceWeight {
  using Value = int;
  __device__ __forceinline__ Value value(int) const { return 0; }
  __device__ __forceinline__ float operator()(float distance, Value, Value) const {
    return distance;
  }
};

// For every point i: the point j with component[j] != component[i] that
// minimises the edge weight of (i, j) (see above), combined into nearest[i] (packed
// with hc::packMinLoc) by atomicMin, so nearest[] must start at
// hc::kEmptyPackedMinLoc. Ties go to the lower j. edgeWeight lets callers turn
// distances into other edge weights (e.g. HDBSCAN's mutual reachability).
// The distance matrix is never stored: each block reduces its tile in
// registers and issues one atomic per row.
template <typename Tile, typename Metric, typename EdgeWeight>
__global__ void __launch_bounds__(Tile::ThreadsPerBlock)
    nearestOtherComponentKernel(const float *__restrict__ points, int numPoints, int numFeatures,
                                Metric metric, EdgeWeight edgeWeight,
                                const int *__restrict__ component,
                                unsigned long long *__restrict__ nearest) {
  // The threads sharing an output row are ThreadGridCols consecutive lanes of
  // one warp, so a group shuffle can combine them.
  static_assert(32 % Tile::ThreadGridCols == 0, "a row's threads must fit in one warp");

  float tileDistances[Tile::RowsPerThread][Tile::ColsPerThread];
  computeTileDistances<Tile>(points, points, numPoints, numPoints, numFeatures, metric,
                             Tile::tileFirstRow(), Tile::tileFirstCol(), tileDistances);

  int colComponent[Tile::ColsPerThread];
  typename EdgeWeight::Value colValue[Tile::ColsPerThread];
#pragma unroll
  for (int c = 0; c < Tile::ColsPerThread; ++c) {
    const int col = Tile::outputCol(c);
    colComponent[c] = col < numPoints ? component[col] : -1;
    colValue[c] = edgeWeight.value(col < numPoints ? col : 0);
  }

#pragma unroll
  for (int r = 0; r < Tile::RowsPerThread; ++r) {
    const int row = Tile::outputRow(r);
    const int rowComponent = row < numPoints ? component[row] : -1;
    const typename EdgeWeight::Value rowValue = edgeWeight.value(row < numPoints ? row : 0);

    hc::MinLoc best = hc::noMinLoc();
#pragma unroll
    for (int c = 0; c < Tile::ColsPerThread; ++c) {
      const int col = Tile::outputCol(c);
      if (row < numPoints && colComponent[c] >= 0 && colComponent[c] != rowComponent)
        best = hc::minLoc(best,
                          hc::MinLoc{edgeWeight(tileDistances[r][c], rowValue, colValue[c]), col});
    }

    // Every lane takes part in the shuffle, including those with no valid row.
    best = hc::groupMinLoc<Tile::ThreadGridCols>(best);
    if (Tile::threadCol() == 0 && row < numPoints && best.index != hc::noMinLoc().index)
      atomicMin(&nearest[row], hc::packMinLoc(best.distance, best.index));
  }
}

// The same result as nearestOtherComponentKernel for a symmetric metric and
// edge weight, with about half the work: only tiles (I, J) with I <= J are
// launched (a 1D grid over the upper triangle). An off-diagonal tile updates
// its rows' minima, as above, and its columns' minima too, because pair
// (row, col) is also pair (col, row). A diagonal tile covers both directions
// on its own. Each pair's weight is computed once and used for both.
//
// Ties still go to the lower index: a column's candidates are packed as
// (weight, row), and atomicMin on the packed value prefers the lower row.
//
// tilePairs, if not null, lists the tile pairs to compute instead of the whole
// triangle (spatial.cuh prunes pairs that can't contain a row's nearest):
// block b computes pair tilePairs[b] = (I << 16) | J with I <= J.
template <typename Tile, typename Metric, typename EdgeWeight>
__global__ void __launch_bounds__(Tile::ThreadsPerBlock)
    nearestOtherComponentSymmetricKernel(const float *__restrict__ points, int numPoints,
                                         int numFeatures, Metric metric, EdgeWeight edgeWeight,
                                         const int *__restrict__ component,
                                         unsigned long long *__restrict__ nearest,
                                         const unsigned int *__restrict__ tilePairs = nullptr) {
  static_assert(Tile::TileRows == Tile::TileCols, "the triangle needs square tiles");
  static_assert(32 % Tile::ThreadGridCols == 0, "a row's threads must fit in one warp");
  constexpr int R = Tile::RowsPerThread, C = Tile::ColsPerThread;

  int tileI;
  long long tileJ;
  if (tilePairs) {
    tileI = (int)(tilePairs[blockIdx.x] >> 16);
    tileJ = tilePairs[blockIdx.x] & 0xffffu;
  } else {
    // blockIdx.x -> (I, J) with I <= J: k = J * (J + 1) / 2 + I.
    const long long k = blockIdx.x;
    tileJ = (long long)((sqrt(8.0 * (double)k + 1.0) - 1.0) / 2.0);
    while (tileJ * (tileJ + 1) / 2 > k) // fix double rounding, either way
      --tileJ;
    while ((tileJ + 1) * (tileJ + 2) / 2 <= k)
      ++tileJ;
    tileI = (int)(k - tileJ * (tileJ + 1) / 2);
  }
  const int tileFirstRow = tileI * Tile::TileRows, tileFirstCol = (int)tileJ * Tile::TileCols;
  const bool diagonal = tileI == (int)tileJ;

  // Column minima of this tile, combined in shared memory first so that each
  // column costs one global atomic. Initialised here; computeTileDistances
  // synchronises the block before anyone uses it.
  __shared__ unsigned long long colBest[Tile::TileCols];
  for (int i = threadIdx.x; i < Tile::TileCols; i += Tile::ThreadsPerBlock)
    colBest[i] = hc::kEmptyPackedMinLoc;

  float weights[R][C];
  computeTileDistances<Tile>(points, points, numPoints, numPoints, numFeatures, metric,
                             tileFirstRow, tileFirstCol, weights);

  // Distances -> edge weights, in place; +inf marks pairs that don't count
  // (outside the matrix, or both points in the same component).
  int cols[C];
  int colComponent[C];
  typename EdgeWeight::Value colValue[C];
#pragma unroll
  for (int c = 0; c < C; ++c) {
    cols[c] = tileFirstCol + Tile::colInTile(c);
    colComponent[c] = cols[c] < numPoints ? component[cols[c]] : -1;
    colValue[c] = edgeWeight.value(cols[c] < numPoints ? cols[c] : 0);
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int row = tileFirstRow + Tile::rowInTile(r);
    const int rowComponent = row < numPoints ? component[row] : -1;
    const typename EdgeWeight::Value rowValue = edgeWeight.value(row < numPoints ? row : 0);
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const bool counts = row < numPoints && colComponent[c] >= 0 && colComponent[c] != rowComponent;
      weights[r][c] = counts ? edgeWeight(weights[r][c], rowValue, colValue[c]) : INFINITY;
    }
  }

  // Rows: as in nearestOtherComponentKernel.
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int row = tileFirstRow + Tile::rowInTile(r);
    hc::MinLoc best = hc::noMinLoc();
#pragma unroll
    for (int c = 0; c < C; ++c)
      if (weights[r][c] != INFINITY)
        best = hc::minLoc(best, hc::MinLoc{weights[r][c], cols[c]});
    best = hc::groupMinLoc<Tile::ThreadGridCols>(best);
    if (Tile::threadCol() == 0 && row < numPoints && best.index != hc::noMinLoc().index)
      atomicMin(&nearest[row], hc::packMinLoc(best.distance, best.index));
  }
  if (diagonal)
    return; // uniform per block, so no thread is left waiting at a barrier

  // Columns: this thread's rows, then the other threads of the warp that own
  // the same columns (lanes ThreadGridCols apart), then shared memory.
#pragma unroll
  for (int c = 0; c < C; ++c) {
    hc::MinLoc best = hc::noMinLoc();
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (weights[r][c] != INFINITY)
        best = hc::minLoc(best, hc::MinLoc{weights[r][c], tileFirstRow + Tile::rowInTile(r)});
#pragma unroll
    for (int offset = Tile::ThreadGridCols; offset < 32; offset *= 2) {
      hc::MinLoc other;
      other.distance = __shfl_xor_sync(0xffffffffu, best.distance, offset);
      other.index = __shfl_xor_sync(0xffffffffu, best.index, offset);
      best = hc::minLoc(best, other);
    }
    if (threadIdx.x % 32 < Tile::ThreadGridCols && best.index != hc::noMinLoc().index)
      atomicMin(&colBest[Tile::colInTile(c)], hc::packMinLoc(best.distance, best.index));
  }
  __syncthreads();
  for (int i = threadIdx.x; i < Tile::TileCols; i += Tile::ThreadsPerBlock) {
    const int col = tileFirstCol + i;
    if (col < numPoints && colBest[i] != hc::kEmptyPackedMinLoc)
      atomicMin(&nearest[col], colBest[i]);
  }
}

// All pointers are device pointers.
template <typename Metric>
void pairwiseDistance(const float *pointsA, const float *pointsB, float *distances,
                      int numPointsA, int numPointsB, int numFeatures, Metric metric,
                      cudaStream_t stream = 0) {
  using Tile = typename DefaultTile<Metric>::type;
  pairwiseDistanceKernel<Tile><<<Tile::grid(numPointsA, numPointsB), Tile::ThreadsPerBlock, 0,
                                 stream>>>(pointsA, pointsB, distances, numPointsA, numPointsB,
                                           numFeatures, metric);
}

// All pointers are device pointers. nearest[] must be filled with
// hc::kEmptyPackedMinLoc beforehand (e.g. cudaMemset to 0xff).
template <typename Metric, typename EdgeWeight>
void nearestOtherComponent(const float *points, int numPoints, int numFeatures, Metric metric,
                           EdgeWeight edgeWeight, const int *component,
                           unsigned long long *nearest, cudaStream_t stream = 0) {
  using Tile = typename DefaultTile<Metric>::type;
  if constexpr (metrics::IsSymmetric<Metric>::value) {
    // Edge weights are symmetric too: they combine per-point values with
    // commutative operations (see the edge-weight comment above).
    const long long numTiles = (numPoints + Tile::TileRows - 1) / Tile::TileRows;
    nearestOtherComponentSymmetricKernel<Tile>
        <<<(unsigned)(numTiles * (numTiles + 1) / 2), Tile::ThreadsPerBlock, 0, stream>>>(
            points, numPoints, numFeatures, metric, edgeWeight, component, nearest);
  } else {
    nearestOtherComponentKernel<Tile><<<Tile::grid(numPoints, numPoints), Tile::ThreadsPerBlock,
                                        0, stream>>>(points, numPoints, numFeatures, metric,
                                                     edgeWeight, component, nearest);
  }
}

} // namespace dist
