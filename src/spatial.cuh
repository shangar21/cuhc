#pragma once

#include "distance_matrix.cuh"
#include "distances.cuh"
#include "reductions.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>

// Exact spatial pruning for the all-pairs nearest-neighbour passes.
//
//  1. spatialOrder sorts the points into k-d tree order, so that each tile of
//     128 consecutive points is a compact region.
//  2. buildTileBoxes gives each tile an axis-aligned bounding box (plus the
//     smallest per-point edge-weight value and its range of components).
//  3. nearestOtherComponentPruned: one block per row tile walks the column
//     tiles, nearest first, and keeps its rows' best edges in shared memory.
//     Before it computes a tile, it compares a lower bound for every pair in
//     the two boxes with the largest current best of its rows. If the bound
//     is larger, no pair in the tile can be a row's nearest, and the block
//     skips the tile.
//
// The bound comes from the per-feature gaps between the boxes. For a metric
// with metrics::HasBoxBound, every pair is at least
//     finalize(accumulate(... accumulate(init(), gap_k, 0) ...)).
// Edge weights never decrease when the distance or the per-point values grow
// (DistanceWeight, MutualReachabilityWeight, WardWeight), so the weight of
// that distance with each tile's smallest values bounds every edge. A small
// relative slack covers float rounding (boxSlack). Skips need the bound to be
// strictly larger, so a skipped pair can't even tie with a row's best: for
// the points in spatial order, the result is the same as without pruning,
// bit for bit. (Callers renumber the points back afterwards. Between edges
// of exactly equal weight, ties go to the lower index in spatial order, so
// the order of equal-height merges can differ from an unpruned run.)
//
// See docs/all_pairs_passes.md.

namespace spatial {

constexpr int kTilePoints = 128; // must match DefaultTile's TileRows / TileCols

// Point order for spatial pruning: order[i] is the original index of the
// point that goes to position i. k-d tree order: split each range at a tile
// boundary near its middle, along the feature with the largest spread
// (estimated from a sample), until each range is one tile.
inline std::vector<int> spatialOrder(const float *points, int numPoints, int numFeatures) {
  std::vector<int> order(numPoints);
  std::iota(order.begin(), order.end(), 0);
  struct Range {
    int begin, end;
  };
  std::vector<Range> stack{{0, numPoints}};
  constexpr int kSample = 64;
  while (!stack.empty()) {
    const Range range = stack.back();
    stack.pop_back();
    const int length = range.end - range.begin;
    if (length <= kTilePoints)
      continue;

    int splitFeature = 0;
    float widest = -1.0f;
    const int step = std::max(1, length / kSample);
    for (int feature = 0; feature < numFeatures; ++feature) {
      float low = INFINITY, high = -INFINITY;
      for (int i = range.begin; i < range.end; i += step) {
        const float value = points[(size_t)order[i] * numFeatures + feature];
        low = std::min(low, value);
        high = std::max(high, value);
      }
      if (high - low > widest) {
        widest = high - low;
        splitFeature = feature;
      }
    }

    // Left side: half of the range's tiles, so leaves line up with tiles.
    const int numTiles = (length + kTilePoints - 1) / kTilePoints;
    const int middle = range.begin + numTiles / 2 * kTilePoints;
    std::nth_element(order.begin() + range.begin, order.begin() + middle,
                     order.begin() + range.end, [&](int a, int b) {
                       return points[(size_t)a * numFeatures + splitFeature] <
                              points[(size_t)b * numFeatures + splitFeature];
                     });
    stack.push_back({range.begin, middle});
    stack.push_back({middle, range.end});
  }
  return order;
}

// destination[i] = source[order[i]], rows of numFeatures floats.
__global__ void gatherRowsKernel(const float *__restrict__ source, const int *__restrict__ order,
                                 int numPoints, int numFeatures, float *__restrict__ destination) {
  const size_t total = (size_t)numPoints * numFeatures;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < total;
       e += (size_t)gridDim.x * blockDim.x) {
    const size_t row = e / numFeatures, feature = e % numFeatures;
    destination[e] = source[(size_t)order[row] * numFeatures + feature];
  }
}

// Per-tile summary for the pruning test. All arrays are device memory.
template <typename Value> struct TileBoxes {
  float *low = nullptr, *high = nullptr; // numTiles x numFeatures
  Value *minValue = nullptr;             // smallest edgeWeight.value(i) in the tile
  int *minComponent = nullptr, *maxComponent = nullptr;
  int numTiles = 0;

  cudaError_t allocate(int tiles, int numFeatures) {
    numTiles = tiles;
    cudaError_t status = cudaMalloc(&low, (size_t)tiles * numFeatures * sizeof(float));
    if (status == cudaSuccess)
      status = cudaMalloc(&high, (size_t)tiles * numFeatures * sizeof(float));
    if (status == cudaSuccess)
      status = cudaMalloc(&minValue, tiles * sizeof(Value));
    if (status == cudaSuccess)
      status = cudaMalloc(&minComponent, tiles * sizeof(int));
    if (status == cudaSuccess)
      status = cudaMalloc(&maxComponent, tiles * sizeof(int));
    return status;
  }
  void release() {
    cudaFree(low);
    cudaFree(high);
    cudaFree(minValue);
    cudaFree(minComponent);
    cudaFree(maxComponent);
    *this = TileBoxes{};
  }
};

// One block per tile. component may be null (then the component range is
// left unset and must not be used).
template <typename EdgeWeight>
__global__ void buildTileBoxesKernel(const float *__restrict__ points, int numPoints,
                                     int numFeatures, EdgeWeight edgeWeight,
                                     const int *__restrict__ component,
                                     TileBoxes<typename EdgeWeight::Value> boxes) {
  const int tile = blockIdx.x;
  const int first = tile * kTilePoints, last = min(numPoints, first + kTilePoints);
  for (int feature = threadIdx.x; feature < numFeatures; feature += blockDim.x) {
    float low = INFINITY, high = -INFINITY;
    for (int point = first; point < last; ++point) {
      const float value = points[(size_t)point * numFeatures + feature];
      low = fminf(low, value);
      high = fmaxf(high, value);
    }
    boxes.low[(size_t)tile * numFeatures + feature] = low;
    boxes.high[(size_t)tile * numFeatures + feature] = high;
  }
  if (threadIdx.x == 0) {
    typename EdgeWeight::Value minValue = edgeWeight.value(first);
    int minComponent = component ? component[first] : 0, maxComponent = minComponent;
    for (int point = first + 1; point < last; ++point) {
      const typename EdgeWeight::Value value = edgeWeight.value(point);
      minValue = value < minValue ? value : minValue;
      if (component) {
        minComponent = min(minComponent, component[point]);
        maxComponent = max(maxComponent, component[point]);
      }
    }
    boxes.minValue[tile] = minValue;
    boxes.minComponent[tile] = minComponent;
    boxes.maxComponent[tile] = maxComponent;
  }
}

template <typename EdgeWeight>
cudaError_t buildTileBoxes(const float *points, int numPoints, int numFeatures,
                           EdgeWeight edgeWeight, const int *component,
                           TileBoxes<typename EdgeWeight::Value> &boxes) {
  buildTileBoxesKernel<<<boxes.numTiles, 64>>>(points, numPoints, numFeatures, edgeWeight,
                                               component, boxes);
  return cudaGetLastError();
}

// Relative slack for the bound: the computed distance of a pair and the
// computed bound can each be off by about numFeatures rounding steps (2^-24),
// more for Minkowski's fast powers. The slack is far larger than that and far
// too small to cost any pruning.
__host__ __device__ __forceinline__ float boxSlack(int numFeatures) {
  return 1e-5f + (numFeatures + 8) * 5e-7f;
}

// Lower bound for metric(x, y) over x in box A and y in box B.
template <typename Metric>
__device__ __forceinline__ float boxLowerBound(const Metric &metric, const float *lowA,
                                               const float *highA, const float *lowB,
                                               const float *highB, int numFeatures) {
  typename Metric::Acc acc = metric.init();
  for (int k = 0; k < numFeatures; ++k) {
    const float gap = fmaxf(0.0f, fmaxf(lowB[k] - highA[k], lowA[k] - highB[k]));
    acc = metric.accumulate(acc, gap, 0.0f);
  }
  return metric.finalize(acc);
}

// The column tile a row tile visits at a given step: itself, then outwards
// by index (I, I+1, I-1, I+2, ...). In k-d order, near indices are near in
// space, so good candidates come first and the bound tightens early. Returns
// -1 for steps that fall outside 0..numTiles-1.
__device__ __forceinline__ int walkTile(int rowTile, int step, int numTiles) {
  const int offset = (step + 1) / 2;
  const int tile = step % 2 == 1 ? rowTile + offset : rowTile - offset;
  return tile >= 0 && tile < numTiles ? tile : -1;
}

// Same result as dist::nearestOtherComponent (see there), with pruning. One
// block per row tile; the block owns its rows' results, so it keeps them in
// shared memory and writes them once at the end. Dynamic shared memory:
// 2 * numFeatures floats (the row tile's box). Counts computed tiles in
// *tilesComputed (for statistics) if it isn't null.
template <typename Tile, typename Metric, typename EdgeWeight>
__global__ void __launch_bounds__(Tile::ThreadsPerBlock)
    nearestOtherComponentWalkKernel(const float *__restrict__ points, int numPoints,
                                    int numFeatures, Metric metric, EdgeWeight edgeWeight,
                                    const int *__restrict__ component,
                                    TileBoxes<typename EdgeWeight::Value> boxes,
                                    unsigned long long *__restrict__ nearest,
                                    unsigned long long *tilesComputed, int rowTileStride = 1) {
  static_assert(Tile::TileRows == kTilePoints && Tile::TileCols == kTilePoints, "tile size");
  static_assert(32 % Tile::ThreadGridCols == 0, "a row's threads must fit in one warp");
  constexpr int R = Tile::RowsPerThread, C = Tile::ColsPerThread;
  using Value = typename EdgeWeight::Value;

  extern __shared__ float rowBox[]; // low[numFeatures], high[numFeatures]
  __shared__ unsigned long long rowBest[Tile::TileRows];
  __shared__ float bound;           // largest current best weight of the rows
  __shared__ int skip;

  // rowTileStride > 1 runs only every stride-th row tile (estimatePrunedShare).
  const int rowTile = blockIdx.x * rowTileStride, numTiles = boxes.numTiles;
  const int tileFirstRow = rowTile * Tile::TileRows;
  for (int i = threadIdx.x; i < Tile::TileRows; i += Tile::ThreadsPerBlock)
    rowBest[i] = hc::kEmptyPackedMinLoc;
  for (int k = threadIdx.x; k < numFeatures; k += Tile::ThreadsPerBlock) {
    rowBox[k] = boxes.low[(size_t)rowTile * numFeatures + k];
    rowBox[numFeatures + k] = boxes.high[(size_t)rowTile * numFeatures + k];
  }
  if (threadIdx.x == 0)
    bound = INFINITY;
  const bool rowTilePure = boxes.minComponent[rowTile] == boxes.maxComponent[rowTile];
  const float slack = 1.0f - boxSlack(numFeatures);

  int rowComponent[R];
  Value rowValue[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int row = tileFirstRow + Tile::rowInTile(r);
    rowComponent[r] = row < numPoints ? component[row] : -1;
    rowValue[r] = edgeWeight.value(row < numPoints ? row : 0);
  }
  unsigned long long computed = 0;
  __syncthreads();

  for (int step = 0; step < 2 * numTiles; ++step) {
    const int colTile = walkTile(rowTile, step, numTiles);
    if (colTile < 0)
      continue; // uniform across the block

    // The test, by one thread; the block follows its decision.
    if (threadIdx.x == 0) {
      bool canSkip = rowTilePure && boxes.minComponent[colTile] == boxes.maxComponent[colTile] &&
                     boxes.minComponent[colTile] == boxes.minComponent[rowTile];
      if (!canSkip && bound != INFINITY) {
        const float distanceBound =
            boxLowerBound(metric, rowBox, rowBox + numFeatures,
                          boxes.low + (size_t)colTile * numFeatures,
                          boxes.high + (size_t)colTile * numFeatures, numFeatures);
        const float weightBound =
            edgeWeight(distanceBound, boxes.minValue[rowTile], boxes.minValue[colTile]) * slack;
        canSkip = weightBound > bound;
      }
      skip = canSkip;
    }
    __syncthreads();
    if (skip) {
      __syncthreads(); // nobody may overwrite `skip` before everyone has read it
      continue;
    }
    ++computed;

    const int tileFirstCol = colTile * Tile::TileCols;
    float weights[R][C];
    dist::computeTileDistances<Tile>(points, points, numPoints, numPoints, numFeatures, metric,
                                     tileFirstRow, tileFirstCol, weights);
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const int col = tileFirstCol + Tile::colInTile(c);
      const int colComponent = col < numPoints ? component[col] : -1;
      const Value colValue = edgeWeight.value(col < numPoints ? col : 0);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const bool counts = rowComponent[r] >= 0 && colComponent >= 0 &&
                            colComponent != rowComponent[r];
        weights[r][c] = counts ? edgeWeight(weights[r][c], rowValue[r], colValue) : INFINITY;
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      hc::MinLoc best = hc::noMinLoc();
#pragma unroll
      for (int c = 0; c < C; ++c)
        if (weights[r][c] != INFINITY)
          best = hc::minLoc(best, hc::MinLoc{weights[r][c], tileFirstCol + Tile::colInTile(c)});
      best = hc::groupMinLoc<Tile::ThreadGridCols>(best);
      // One group of threads owns each row, so its first thread can update
      // the row's best without an atomic.
      if (Tile::threadCol() == 0 && best.index != hc::noMinLoc().index) {
        const unsigned long long packed = hc::packMinLoc(best.distance, best.index);
        unsigned long long &current = rowBest[Tile::rowInTile(r)];
        current = packed < current ? packed : current;
      }
    }
    __syncthreads();

    // New bound: the largest best among the valid rows (inf if any row has
    // no candidate yet). Warp 0 reduces the 128 values.
    if (threadIdx.x < 32) {
      float largest = 0.0f;
      for (int i = threadIdx.x; i < Tile::TileRows; i += 32) {
        if (tileFirstRow + i >= numPoints)
          continue;
        const unsigned long long packed = rowBest[i];
        largest = fmaxf(largest, packed == hc::kEmptyPackedMinLoc ? INFINITY
                                                                   : hc::unpackDistance(packed));
      }
      for (int offset = 16; offset > 0; offset /= 2)
        largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, offset));
      if (threadIdx.x == 0)
        bound = largest;
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < Tile::TileRows; i += Tile::ThreadsPerBlock)
    if (tileFirstRow + i < numPoints && rowBest[i] != hc::kEmptyPackedMinLoc)
      atomicMin(&nearest[tileFirstRow + i], rowBest[i]);
  if (tilesComputed && threadIdx.x == 0)
    atomicAdd(tilesComputed, computed);
}

// When pruning pays. Boxes stop separating points in many dimensions, and
// small inputs are fast anyway. Set from measurements (docs/all_pairs_passes.md).
constexpr int kPruneMinPoints = 16384;
constexpr int kPruneMaxFeatures = 64;
// Hard limit, also for PruneMode::Always: each block keeps its row tile's box
// (2 * numFeatures floats) in shared memory, next to the core-distance lists
// (up to 32 KB) and the tile loop's 8.4 KB, within 48 KB.
constexpr int kPruneHardMaxFeatures = 256;

enum class PruneMode : int { Auto = 0, Always = 1, Never = 2 };

template <typename Metric>
bool shouldPrune(PruneMode mode, int numPoints, int numFeatures) {
  if (!metrics::HasBoxBound<Metric>::value || mode == PruneMode::Never ||
      (numPoints + kTilePoints - 1) / kTilePoints < 2 || numFeatures > kPruneHardMaxFeatures)
    return false;
  return mode == PruneMode::Always ||
         (numPoints >= kPruneMinPoints && numFeatures <= kPruneMaxFeatures);
}

// Statistics of the last pruned passes, for benchmarks and docs.
struct PruneStats {
  unsigned long long tilesComputed = 0, tilesTotal = 0;
};

// The walk kernel computes each pair of tiles twice (once from each side),
// where the symmetric kernel computes it once. So if a pass computes more than
// this share of all tiles, pruning costs more than it saves, and callers stop
// pruning for their remaining passes.
constexpr double kPruneGiveUpShare = 0.5;

// Same contract as dist::nearestOtherComponent (nearest[] must start at
// hc::kEmptyPackedMinLoc), for points in spatial order. Needs a metric with
// HasBoxBound. *computedShare receives the share of tiles it computed; stats
// (if not null) accumulates the counts. Returns a cudaError_t.
template <typename Metric, typename EdgeWeight>
cudaError_t nearestOtherComponentPruned(const float *points, int numPoints, int numFeatures,
                                        Metric metric, EdgeWeight edgeWeight,
                                        const int *component, unsigned long long *nearest,
                                        double *computedShare, PruneStats *stats = nullptr) {
  static_assert(metrics::HasBoxBound<Metric>::value, "pruning needs a box bound");
  using Tile = typename dist::DefaultTile<Metric>::type;
  const int numTiles = (numPoints + kTilePoints - 1) / kTilePoints;
  TileBoxes<typename EdgeWeight::Value> boxes;
  unsigned long long *deviceComputed = nullptr;
  cudaError_t status = boxes.allocate(numTiles, numFeatures);
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceComputed, sizeof(unsigned long long));
  if (status == cudaSuccess)
    status = cudaMemset(deviceComputed, 0, sizeof(unsigned long long));
  if (status == cudaSuccess)
    status = buildTileBoxes(points, numPoints, numFeatures, edgeWeight, component, boxes);
  if (status == cudaSuccess) {
    nearestOtherComponentWalkKernel<Tile>
        <<<numTiles, Tile::ThreadsPerBlock, 2 * numFeatures * sizeof(float)>>>(
            points, numPoints, numFeatures, metric, edgeWeight, component, boxes, nearest,
            deviceComputed);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) {
    unsigned long long computed = 0;
    const unsigned long long total = (unsigned long long)numTiles * numTiles;
    status = cudaMemcpy(&computed, deviceComputed, sizeof(computed), cudaMemcpyDeviceToHost);
    *computedShare = (double)computed / (double)total;
    if (stats) {
      stats->tilesComputed += computed;
      stats->tilesTotal += total;
    }
  }
  boxes.release();
  cudaFree(deviceComputed);
  return status;
}

// Share of tiles a pruned pass would compute, measured on a sample of row
// tiles (every stride-th one, about sampleTiles of them): a few percent of the
// cost of one pass. Plain distances, every point its own component: that is
// Borůvka's first pass and the shape of the other passes. Callers in "auto"
// mode don't prune if the share is above kPruneGiveUpShare.
template <typename Metric>
cudaError_t estimatePrunedShare(const float *points, int numPoints, int numFeatures,
                                Metric metric, double *share, int sampleTiles = 32) {
  using Tile = typename dist::DefaultTile<Metric>::type;
  const int numTiles = (numPoints + kTilePoints - 1) / kTilePoints;
  const int stride = std::max(1, numTiles / sampleTiles);
  const int numSampled = (numTiles + stride - 1) / stride;
  TileBoxes<int> boxes;
  int *component = nullptr;
  unsigned long long *nearest = nullptr, *computed = nullptr;
  cudaError_t status = boxes.allocate(numTiles, numFeatures);
  if (status == cudaSuccess)
    status = cudaMalloc(&component, numPoints * sizeof(int));
  if (status == cudaSuccess)
    status = cudaMalloc(&nearest, numPoints * sizeof(unsigned long long));
  if (status == cudaSuccess)
    status = cudaMalloc(&computed, sizeof(unsigned long long));
  if (status == cudaSuccess) {
    std::vector<int> iota(numPoints);
    std::iota(iota.begin(), iota.end(), 0);
    status = cudaMemcpy(component, iota.data(), numPoints * sizeof(int), cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess)
    status = cudaMemset(nearest, 0xff, numPoints * sizeof(unsigned long long));
  if (status == cudaSuccess)
    status = cudaMemset(computed, 0, sizeof(unsigned long long));
  if (status == cudaSuccess)
    status = buildTileBoxes(points, numPoints, numFeatures, dist::DistanceWeight{}, component,
                            boxes);
  if (status == cudaSuccess) {
    nearestOtherComponentWalkKernel<Tile>
        <<<numSampled, Tile::ThreadsPerBlock, 2 * numFeatures * sizeof(float)>>>(
            points, numPoints, numFeatures, metric, dist::DistanceWeight{}, component, boxes,
            nearest, computed, stride);
    status = cudaGetLastError();
  }
  unsigned long long count = 0;
  if (status == cudaSuccess)
    status = cudaMemcpy(&count, computed, sizeof(count), cudaMemcpyDeviceToHost);
  *share = (double)count / ((double)numSampled * numTiles);
  boxes.release();
  cudaFree(component);
  cudaFree(nearest);
  cudaFree(computed);
  return status;
}

} // namespace spatial
