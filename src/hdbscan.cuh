#pragma once

#include "distance_matrix.cuh"
#include "mst.cuh"
#include "spatial.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <vector>

// GPU part of HDBSCAN: core distances, then the single-linkage tree of the
// mutual reachability distance (via the Borůvka MST in mst.cuh). Condensing the
// tree and choosing clusters happen on the CPU (python/cuhc/_hdbscan.py).
//
// Core distance: for every point, the distance to its k-th nearest point,
// counting the point itself as the 1st (so k = 1 gives 0). This is sklearn's
// convention for min_samples. Two ways to compute it, both exact:
//
//   fused (k <= kMaxFusedK, the usual case): each block owns a tile of rows and
//   walks over all columns; each row's k smallest distances are kept in shared
//   memory. No distances are written to global memory.
//
//   blocked + radix select (any k): the distance matrix is computed a block of
//   rows at a time into a bounded buffer; for each row, a radix select finds
//   the k-th smallest value one byte at a time (non-negative floats order like
//   their bit patterns) with a 256-bin histogram per byte.
//
// condenseTree (host) is HDBSCAN's tree condensing step; the rest of the
// cluster extraction is in python/cuhc/_hdbscan.py.
//
// See docs/hdbscan.md.

namespace hc {

// Adds 1 to histogram[bucket] for every lane that calls it, with one shared
// atomic per distinct bucket in the warp instead of one per lane. Lanes that
// pass the same bucket find each other with __match_any_sync, and the lowest
// of them adds the group's count. Matters because nearby distances share their
// high bytes, so many lanes often hit the same bin.
__device__ __forceinline__ void warpAggregatedIncrement(unsigned int *histogram, unsigned int bucket,
                                                        bool active) {
  const unsigned int activeLanes = __ballot_sync(0xffffffffu, active);
  if (!active)
    return;
  const unsigned int sameBucket = __match_any_sync(activeLanes, bucket);
  const int lane = threadIdx.x % 32;
  if (lane == __ffs(sameBucket) - 1)
    atomicAdd(&histogram[bucket], __popc(sameBucket));
}

// One block per row of `distances` (rowsInChunk x numCols, row-major), which
// holds rows firstRow.. of the full matrix. core[firstRow + row] receives the
// k-th smallest value of the row (1-based), with the row's own point counted
// as distance 0.
template <int BlockSize>
__global__ void kthSmallestPerRowKernel(const float *__restrict__ distances, int numCols,
                                        int firstRow, int k, float *__restrict__ core) {
  __shared__ unsigned int histogram[256];
  __shared__ unsigned int prefix, prefixMask, rankRemaining;

  const int row = blockIdx.x;
  const int self = firstRow + row;
  const float *rowDistances = distances + (size_t)row * numCols;
  if (threadIdx.x == 0) {
    prefix = 0;
    prefixMask = 0;
    rankRemaining = k;
  }

  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int bin = threadIdx.x; bin < 256; bin += BlockSize)
      histogram[bin] = 0;
    __syncthreads();

    // Count, among the values that match the bytes chosen so far, how many
    // have each possible value of the current byte. The loop bound is rounded
    // up so all lanes of a warp reach the warp-wide intrinsics together.
    const unsigned int currentPrefix = prefix, currentMask = prefixMask;
    for (int base = 0; base < numCols; base += BlockSize) {
      const int col = base + threadIdx.x;
      unsigned int bits = 0;
      if (col < numCols) {
        const float value = col == self ? 0.0f : fmaxf(rowDistances[col], 0.0f);
        bits = __float_as_uint(value);
      }
      const bool matches = col < numCols && (bits & currentMask) == currentPrefix;
      warpAggregatedIncrement(histogram, (bits >> shift) & 0xffu, matches);
    }
    __syncthreads();

    // The k-th smallest is in the first bin where the running count reaches k.
    if (threadIdx.x == 0) {
      unsigned int below = 0, bin = 0;
      while (below + histogram[bin] < rankRemaining)
        below += histogram[bin++];
      rankRemaining -= below;
      prefix |= bin << shift;
      prefixMask |= 0xffu << shift;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
    core[self] = __uint_as_float(prefix);
}

// ---------------------------------------------------------------------------
// Fused k-nearest: no distances written to global memory

// Largest k for the fused kernel: its per-row lists take TileRows * k floats of
// shared memory, which must fit next to the tile loop's 8.4 KB in 48 KB.
constexpr int kMaxFusedK = 64;

// Inserts value into the ascending list[0..k), dropping the largest, if it is
// smaller than the current k-th.
__device__ __forceinline__ void insertSorted(float *list, int k, float value) {
  if (!(value < list[k - 1]))
    return;
  int position = k - 1;
  while (position > 0 && list[position - 1] > value) {
    list[position] = list[position - 1];
    --position;
  }
  list[position] = value;
}

// One block per tile of rows; the block walks over every column tile. For
// each of its rows it keeps the k smallest distances seen so far, ascending,
// in shared memory (dynamic: TileRows * k floats); core[row] = the k-th.
//
// A row's list is only touched by the ThreadGridCols threads that compute that
// row, which are consecutive lanes of one warp. After each tile, each of those
// threads marks which of its distances beat the row's current k-th, and the
// group inserts them one at a time: __ballot_sync finds lanes with work, the
// lowest of them inserts one value. Insertions are rare after the first few
// tiles (about k * ln(n / k) per row in total), so this costs little.
//
// The block visits the column tiles nearest first by index (spatial::walkTile).
// boxes (spatial.cuh), if numTiles > 0, enable pruning for points in spatial
// order: the block skips a tile if the lower bound for its box is larger than
// every row's current k-th. The k-th smallest doesn't depend on the visit
// order or on skipped tiles, so the result is the same.
// Pruning adds 2 * numFeatures floats of dynamic shared memory.
template <typename Tile, typename Metric>
__global__ void __launch_bounds__(Tile::ThreadsPerBlock)
    kthNearestFusedKernel(const float *__restrict__ points, int numPoints, int numFeatures,
                          Metric metric, int k, float *__restrict__ core,
                          spatial::TileBoxes<int> boxes = {}) {
  constexpr int GroupSize = Tile::ThreadGridCols;
  static_assert(32 % GroupSize == 0 && Tile::ColsPerThread <= 32, "row group must fit in a warp");
  extern __shared__ float nearest[]; // [TileRows][k], then the row tile's box if pruning
  __shared__ int skip;

  const int tileFirstRow = blockIdx.x * Tile::TileRows;
  for (int i = threadIdx.x; i < Tile::TileRows * k; i += Tile::ThreadsPerBlock)
    nearest[i] = INFINITY;
  const bool prune = boxes.numTiles > 0;
  float *rowBox = nearest + Tile::TileRows * k;
  if (prune)
    for (int f = threadIdx.x; f < numFeatures; f += Tile::ThreadsPerBlock) {
      rowBox[f] = boxes.low[(size_t)blockIdx.x * numFeatures + f];
      rowBox[numFeatures + f] = boxes.high[(size_t)blockIdx.x * numFeatures + f];
    }
  __syncthreads();
  const float slack = 1.0f - spatial::boxSlack(numFeatures);
  const int numTiles = (numPoints + Tile::TileCols - 1) / Tile::TileCols;

  const int lane = threadIdx.x % 32;
  const unsigned int groupMask =
      (GroupSize == 32 ? 0xffffffffu : ((1u << GroupSize) - 1u)) << (lane / GroupSize * GroupSize);

  // Always nearest tiles first (by index), pruning or not: after the first
  // few tiles a row's list rarely changes, so insertions stay rare. Visiting
  // tiles 0, 1, 2, ... instead would be the worst case for spatially ordered
  // points: each tile nearer than the last, so most distances get inserted.
  for (int step = 0; step < 2 * numTiles; ++step) {
    const int colTile = spatial::walkTile(blockIdx.x, step, numTiles);
    if (colTile < 0)
      continue;
    if (prune) {
      __syncthreads(); // the previous tile's insertions are done
      // Warp 0: the largest current k-th among the valid rows, then the test.
      if (threadIdx.x < 32) {
        float largest = 0.0f;
        for (int i = threadIdx.x; i < Tile::TileRows; i += 32)
          if (tileFirstRow + i < numPoints)
            largest = fmaxf(largest, nearest[i * k + k - 1]);
        for (int offset = 16; offset > 0; offset /= 2)
          largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, offset));
        if (threadIdx.x == 0)
          skip = largest != INFINITY &&
                 spatial::boxLowerBound(metric, rowBox, rowBox + numFeatures,
                                        boxes.low + (size_t)colTile * numFeatures,
                                        boxes.high + (size_t)colTile * numFeatures,
                                        numFeatures) * slack > largest;
      }
      __syncthreads();
      const bool skipTile = skip;
      __syncthreads(); // everyone has read `skip` before the next step writes it
      if (skipTile)
        continue;
    }
    const int tileFirstCol = colTile * Tile::TileCols;
    float tileDistances[Tile::RowsPerThread][Tile::ColsPerThread];
    dist::computeTileDistances<Tile>(points, points, numPoints, numPoints, numFeatures, metric,
                                     tileFirstRow, tileFirstCol, tileDistances);

#pragma unroll
    for (int r = 0; r < Tile::RowsPerThread; ++r) {
      const int rowInTile = Tile::rowInTile(r);
      const int row = tileFirstRow + rowInTile;
      float *list = nearest + rowInTile * k;

      // Which of this thread's distances for the row beat the current k-th.
      unsigned int pending = 0;
      if (row < numPoints) {
        const float kth = list[k - 1];
#pragma unroll
        for (int c = 0; c < Tile::ColsPerThread; ++c) {
          const int col = tileFirstCol + Tile::colInTile(c);
          // The point itself counts as distance 0 (removes float noise such
          // as cosine giving 1e-7 for d(x, x)).
          tileDistances[r][c] = col == row ? 0.0f : fmaxf(tileDistances[r][c], 0.0f);
          if (col < numPoints && tileDistances[r][c] < kth)
            pending |= 1u << c;
        }
      }

      // Insert them, one value per group per step. The loop runs until no
      // lane in the whole warp has work, so all lanes stay converged for the
      // ballot (the warp's two groups work on different rows at once).
      while (true) {
        const unsigned int lanesWithWork = __ballot_sync(0xffffffffu, pending != 0);
        if (lanesWithWork == 0)
          break;
        const unsigned int groupWork = lanesWithWork & groupMask;
        if (groupWork != 0 && lane == __ffs(groupWork) - 1) {
          const int c = __ffs(pending) - 1;
          pending &= pending - 1;
          float value = 0.0f; // select without dynamic register indexing
#pragma unroll
          for (int cc = 0; cc < Tile::ColsPerThread; ++cc)
            if (cc == c)
              value = tileDistances[r][cc];
          insertSorted(list, k, value);
        }
        __syncwarp();
      }
    }
  }

  __syncthreads();
  for (int rowInTile = threadIdx.x; rowInTile < Tile::TileRows; rowInTile += Tile::ThreadsPerBlock) {
    const int row = tileFirstRow + rowInTile;
    if (row < numPoints)
      core[row] = nearest[rowInTile * k + k - 1];
  }
}

// prune: the points are in spatial order; skip far tiles (needs a metric
// with metrics::HasBoxBound, else ignored).
template <typename Metric>
cudaError_t coreDistancesFused(const float *devicePoints, int numPoints, int numFeatures,
                               Metric metric, int k, float *deviceCore, bool prune = false) {
  using Tile = typename dist::DefaultTile<Metric>::type;
  const int blocks = (numPoints + Tile::TileRows - 1) / Tile::TileRows;
  size_t sharedBytes = (size_t)Tile::TileRows * k * sizeof(float);
  spatial::TileBoxes<int> boxes;
  cudaError_t status = cudaSuccess;
  if (prune && metrics::HasBoxBound<Metric>::value) {
    status = boxes.allocate(blocks, numFeatures);
    if (status == cudaSuccess)
      status = spatial::buildTileBoxes(devicePoints, numPoints, numFeatures,
                                       dist::DistanceWeight{}, nullptr, boxes);
    sharedBytes += 2 * numFeatures * sizeof(float);
  }
  if (status == cudaSuccess) {
    kthNearestFusedKernel<Tile><<<blocks, Tile::ThreadsPerBlock, sharedBytes>>>(
        devicePoints, numPoints, numFeatures, metric, k, deviceCore, boxes);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaDeviceSynchronize();
  boxes.release();
  return status;
}

// ---------------------------------------------------------------------------
// Blocked rows + radix select

// Same result as the fused kernel for any k. Uses a buffer of about
// maxBufferBytes (at least one tile of rows).
template <typename Metric>
cudaError_t coreDistancesRadixSelect(const float *devicePoints, int numPoints, int numFeatures,
                                     Metric metric, int k, float *deviceCore,
                                     size_t maxBufferBytes = (size_t)512 << 20) {
  constexpr int kSelectBlock = 256;
  using Tile = typename dist::DefaultTile<Metric>::type;
  const size_t rowBytes = (size_t)numPoints * sizeof(float);
  // Whole tiles of rows, so the distance kernel doesn't compute unused rows.
  int rowsPerChunk = (int)std::min<size_t>(maxBufferBytes / rowBytes, numPoints);
  rowsPerChunk = std::max(Tile::TileRows, rowsPerChunk / Tile::TileRows * Tile::TileRows);
  rowsPerChunk = std::min(rowsPerChunk, numPoints);

  float *chunk = nullptr;
  cudaError_t status = cudaMalloc(&chunk, (size_t)rowsPerChunk * rowBytes);
  for (int firstRow = 0; status == cudaSuccess && firstRow < numPoints; firstRow += rowsPerChunk) {
    const int rows = std::min(rowsPerChunk, numPoints - firstRow);
    dist::pairwiseDistance(devicePoints + (size_t)firstRow * numFeatures, devicePoints, chunk, rows,
                           numPoints, numFeatures, metric);
    kthSmallestPerRowKernel<kSelectBlock><<<rows, kSelectBlock>>>(chunk, numPoints, firstRow, k,
                                                                  deviceCore);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaDeviceSynchronize();
  cudaFree(chunk);
  return status;
}

// ---------------------------------------------------------------------------
// Entry point

enum class CoreMethod : int { Auto = 0, Fused = 1, RadixSelect = 2 };

// core[i] = distance from point i to its k-th nearest point (itself = 1st).
// devicePoints is numPoints x numFeatures row-major; deviceCore has numPoints
// floats. Auto uses the fused kernel when k <= kMaxFusedK.
template <typename Metric>
cudaError_t coreDistances(const float *devicePoints, int numPoints, int numFeatures, Metric metric,
                          int k, float *deviceCore, CoreMethod method = CoreMethod::Auto,
                          bool prune = false) {
  if (method == CoreMethod::Fused && k > kMaxFusedK)
    return cudaErrorInvalidValue;
  if (method == CoreMethod::Fused || (method == CoreMethod::Auto && k <= kMaxFusedK))
    return coreDistancesFused(devicePoints, numPoints, numFeatures, metric, k, deviceCore, prune);
  return coreDistancesRadixSelect(devicePoints, numPoints, numFeatures, metric, k, deviceCore);
}

// mreach(i, j) = max(core[i], core[j], d(i, j) / alpha), as in sklearn's
// default (feature input) path and the original hdbscan library.
struct MutualReachabilityWeight {
  const float *core; // device pointer
  float alpha;
  using Value = float;
  __device__ __forceinline__ Value value(int point) const { return core[point]; }
  __device__ __forceinline__ float operator()(float distance, Value coreI, Value coreJ) const {
    // A division per pair is expensive; alpha is almost always 1.
    return fmaxf(fmaxf(coreI, coreJ), alpha == 1.0f ? distance : distance / alpha);
  }
};

// Single-linkage tree of the mutual reachability distance, in scipy's linkage
// format: tree receives (numPoints - 1) x 4 doubles (host memory). Also
// returns the core distances in hostCore (numPoints floats) if not null.
//
// prune: the points are in spatial order; originalIndex (host, numPoints)
// gives each one's original index, and the tree and the core distances are
// returned for the original order. Same result as without pruning.
template <typename Metric>
int hdbscanSingleLinkage(const float *devicePoints, int numPoints, int numFeatures, Metric metric,
                         int minSamples, float alpha, double *tree, float *hostCore,
                         CoreMethod coreMethod = CoreMethod::Auto, bool prune = false,
                         const int *originalIndex = nullptr,
                         spatial::PruneStats *stats = nullptr) {
  float *deviceCore = nullptr;
  int status = cudaMalloc(&deviceCore, numPoints * sizeof(float));
  if (status == cudaSuccess)
    status = coreDistances(devicePoints, numPoints, numFeatures, metric, minSamples, deviceCore,
                           coreMethod, prune);

  std::vector<MstEdge> edges;
  if (status == cudaSuccess)
    status = boruvkaMst(devicePoints, numPoints, numFeatures, metric, edges,
                        MutualReachabilityWeight{deviceCore, alpha}, prune, stats);
  if (status == cudaSuccess && hostCore != nullptr) {
    std::vector<float> core(numPoints);
    status = cudaMemcpy(core.data(), deviceCore, numPoints * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numPoints; ++i)
      hostCore[originalIndex ? originalIndex[i] : i] = core[i];
  }
  if (status == cudaSuccess && originalIndex)
    for (MstEdge &edge : edges) {
      const int u = originalIndex[edge.u], v = originalIndex[edge.v];
      edge = MstEdge{std::min(u, v), std::max(u, v), edge.weight};
    }
  if (status == cudaSuccess)
    singleLinkageFromMst(std::move(edges), numPoints, tree);
  cudaFree(deviceCore);
  return status;
}

// ---------------------------------------------------------------------------
// Condensing the tree (host)

// One row of the condensed tree: child (a point < n, or a cluster >= n) leaves
// or splits off from cluster `parent` at lambda = 1 / distance.
struct CondensedRow {
  long long parent, child;
  double lambda;
  long long size;
};

// HDBSCAN's condensing step, a line-for-line port of condense_tree in
// python/cuhc/_hdbscan.py (itself a port of sklearn's _condense_tree), so the
// rows come out identical and in the same order. tree is the (n-1) x 4
// single-linkage tree in scipy format. Walks it breadth-first from the root:
//   both sides >= minClusterSize: a real split, two new clusters;
//   one side smaller:   the big side continues the parent cluster, the small
//                       side's points fall out of it;
//   both sides smaller: all points fall out; the parent cluster ends.
inline std::vector<CondensedRow> condenseTree(const double *tree, long long numPoints,
                                              long long minClusterSize) {
  const long long n = numPoints;
  const long long root = 2 * (n - 1);
  auto left = [&](long long node) { return (long long)tree[(node - n) * 4 + 0]; };
  auto right = [&](long long node) { return (long long)tree[(node - n) * 4 + 1]; };
  auto distance = [&](long long node) { return tree[(node - n) * 4 + 2]; };
  auto count = [&](long long node) { return node >= n ? (long long)tree[(node - n) * 4 + 3] : 1; };

  // Breadth-first order below `from`, children left before right, as in the
  // Python version. Appends to `out`.
  std::vector<long long> queue;
  auto bfs = [&](long long from, std::vector<long long> &out) {
    queue.assign(1, from);
    for (size_t head = 0; head < queue.size(); ++head) {
      const long long node = queue[head];
      out.push_back(node);
      if (node >= n) {
        queue.push_back(left(node));
        queue.push_back(right(node));
      }
    }
  };

  std::vector<long long> nodeList;
  nodeList.reserve(root + 1);
  bfs(root, nodeList);

  std::vector<long long> relabel(root + 1, 0);
  std::vector<char> ignore(root + 1, 0);
  relabel[root] = n;
  long long nextLabel = n + 1;
  std::vector<CondensedRow> rows;
  rows.reserve(n + 2 * (n / std::max(minClusterSize, 1LL)) + 2);

  std::vector<long long> subtree;
  auto dropSubtree = [&](long long parentLabel, long long subtreeRoot, double lambda) {
    subtree.clear();
    bfs(subtreeRoot, subtree);
    for (long long subNode : subtree) {
      if (subNode < n)
        rows.push_back({parentLabel, subNode, lambda, 1});
      ignore[subNode] = 1;
    }
  };

  for (long long node : nodeList) {
    if (ignore[node] || node < n)
      continue;
    const long long l = left(node), r = right(node);
    const double d = distance(node);
    const double lambda = d > 0.0 ? 1.0 / d : INFINITY;
    const long long leftCount = count(l), rightCount = count(r);

    if (leftCount >= minClusterSize && rightCount >= minClusterSize) {
      relabel[l] = nextLabel++;
      rows.push_back({relabel[node], relabel[l], lambda, leftCount});
      relabel[r] = nextLabel++;
      rows.push_back({relabel[node], relabel[r], lambda, rightCount});
    } else if (leftCount < minClusterSize && rightCount < minClusterSize) {
      dropSubtree(relabel[node], l, lambda);
      dropSubtree(relabel[node], r, lambda);
    } else if (leftCount < minClusterSize) {
      relabel[r] = relabel[node];
      dropSubtree(relabel[node], l, lambda);
    } else {
      relabel[l] = relabel[node];
      dropSubtree(relabel[node], r, lambda);
    }
  }
  return rows;
}

} // namespace hc
