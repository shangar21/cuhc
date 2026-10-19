#pragma once

#include "distance_matrix.cuh"
#include "reductions.cuh"

#include <cuda_runtime.h>

// Exact agglomerative clustering on a dense distance matrix.
//
// Every cluster lives in a "slot" (a row/column of the matrix). Merging the
// clusters in slots a < b puts the merged cluster in slot a and retires slot b.
// For each active slot we keep its nearest active neighbour (distance + slot),
// so finding the closest pair is an argmin over n values instead of n^2.
//
// One merge = three kernels:
//   1. findClosestPairKernel  argmin of nearestDistance; records the merge.
//   2. mergeKernel            Lance-Williams update of row/column a; retires
//                             column b; works out which rows lost their nearest
//                             neighbour.
//   3. rescanKernel           recomputes the nearest neighbour of those rows.
//
// Step 2 relies on all supported linkages being "reducible": the distance from
// a merged cluster to k is never smaller than both old distances, so a row
// whose nearest neighbour wasn't a or b still has the same nearest neighbour.
//
// Output is scipy's linkage format: tree[i] = {id1, id2, distance, size}, where
// leaves are 0..n-1 and the cluster made by merge i is n + i.

namespace hc {

enum class Linkage : int { Single = 0, Complete = 1, Average = 2, Ward = 3 };

// Distance from the merged cluster (a ∪ b) to cluster k, from the old
// distances alone. Done in double: it's only O(n) work per merge, and rounding
// would otherwise compound over thousands of merges.
__device__ __forceinline__ double lanceWilliams(Linkage linkage, double distAK, double distBK,
                                                double distAB, double sizeA, double sizeB,
                                                double sizeK) {
  switch (linkage) {
  case Linkage::Single:
    return fmin(distAK, distBK);
  case Linkage::Complete:
    return fmax(distAK, distBK);
  case Linkage::Average:
    return (sizeA * distAK + sizeB * distBK) / (sizeA + sizeB);
  case Linkage::Ward: {
    // Ward on (non-squared) Euclidean distances, as in scipy and sklearn.
    const double total = sizeA + sizeB + sizeK;
    const double squared = ((sizeA + sizeK) * distAK * distAK + (sizeB + sizeK) * distBK * distBK -
                            sizeK * distAB * distAB) /
                           total;
    return sqrt(fmax(squared, 0.0));
  }
  }
  return 0.0;
}

// ---------------------------------------------------------------------------
// State

struct Merge {
  int slotA, slotB; // slotA < slotB; the merged cluster lives in slotA
  float distance;
  int sizeA, sizeB; // sizes before the merge
};

// All pointers are device pointers.
struct DenseLinkageState {
  float *distances;        // n x n row-major; diagonal and retired columns hold +inf
  float *nearestDistance;  // per slot: distance to its nearest active slot (+inf if retired)
  int *nearestSlot;        // per slot: that nearest active slot
  int *clusterSize;        // per slot: number of points in the cluster
  int *clusterId;          // per slot: the cluster's id in the output tree
  unsigned char *isActive; // per slot
  int *rescanSlots;        // slots whose nearest neighbour must be recomputed
  int *rescanCount;
  Merge *merge;            // the merge being performed in this step
  double *tree;            // (n-1) x 4 output
  const int *initialSize;  // per slot: starting cluster size (null = all 1)
  const int *initialId;    // per slot: starting cluster id in the tree (null = slot)
  int firstNewId;          // id of the cluster made by merge 0 (scipy: numPoints)
  int numPoints;
  Linkage linkage;
};

// ---------------------------------------------------------------------------
// Kernels

__global__ void initKernel(DenseLinkageState state) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= state.numPoints)
    return;
  state.distances[(size_t)slot * state.numPoints + slot] = INFINITY;
  state.clusterSize[slot] = state.initialSize ? state.initialSize[slot] : 1;
  state.clusterId[slot] = state.initialId ? state.initialId[slot] : slot;
  state.isActive[slot] = 1;
  state.rescanSlots[slot] = slot; // every row needs its first nearest neighbour
  if (slot == 0)
    *state.rescanCount = state.numPoints;
}

// Recomputes nearestDistance/nearestSlot for every slot in rescanSlots, one
// block per row at a time. Retired columns hold +inf, so no activity check is
// needed while scanning.
template <int BlockSize> __global__ void rescanKernel(DenseLinkageState state) {
  const int count = *state.rescanCount;
  const int n = state.numPoints;
  for (int item = blockIdx.x; item < count; item += gridDim.x) {
    const int row = state.rescanSlots[item];
    const float *rowDistances = state.distances + (size_t)row * n;

    MinLoc best = noMinLoc();
    for (int col = threadIdx.x; col < n; col += BlockSize)
      best = minLoc(best, MinLoc{rowDistances[col], col});
    best = blockMinLoc<BlockSize>(best);

    if (threadIdx.x == 0) {
      state.nearestDistance[row] = best.distance;
      state.nearestSlot[row] = best.index;
    }
  }
}

// Single block: finds the closest pair of active clusters, writes tree row
// `step`, and updates the bookkeeping for the merged cluster.
template <int BlockSize> __global__ void findClosestPairKernel(DenseLinkageState state, int step) {
  MinLoc best = noMinLoc();
  for (int slot = threadIdx.x; slot < state.numPoints; slot += BlockSize)
    best = minLoc(best, MinLoc{state.nearestDistance[slot], slot});
  best = blockMinLoc<BlockSize>(best);
  if (threadIdx.x != 0)
    return;

  const int other = state.nearestSlot[best.index];
  const int slotA = min(best.index, other), slotB = max(best.index, other);
  const int sizeA = state.clusterSize[slotA], sizeB = state.clusterSize[slotB];
  const int idA = state.clusterId[slotA], idB = state.clusterId[slotB];
  *state.merge = Merge{slotA, slotB, best.distance, sizeA, sizeB};

  double *row = state.tree + (size_t)step * 4;
  row[0] = min(idA, idB);
  row[1] = max(idA, idB);
  row[2] = best.distance;
  row[3] = sizeA + sizeB;

  // mergeKernel reads the old sizes from *state.merge, so these can change now.
  state.clusterSize[slotA] = sizeA + sizeB;
  state.clusterId[slotA] = state.firstNewId + step;
  state.isActive[slotB] = 0;
  state.nearestDistance[slotB] = INFINITY;
  *state.rescanCount = 0;
}

// One thread per slot k: distance from the merged cluster to k (Lance-Williams),
// written to row and column a; column b is retired. Then decides whether k's
// nearest neighbour is still known or k must be rescanned.
__global__ void mergeKernel(DenseLinkageState state) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = state.numPoints;
  if (k >= n || !state.isActive[k])
    return;
  const Merge merge = *state.merge;
  const int a = merge.slotA, b = merge.slotB;
  float *distances = state.distances;

  if (k == a) {
    distances[(size_t)a * n + b] = INFINITY;
    state.rescanSlots[atomicAdd(state.rescanCount, 1)] = a; // row a changed entirely
    return;
  }

  const float newDistance = (float)lanceWilliams(
      state.linkage, distances[(size_t)a * n + k], distances[(size_t)b * n + k], merge.distance,
      merge.sizeA, merge.sizeB, state.clusterSize[k]);
  distances[(size_t)a * n + k] = newDistance;
  distances[(size_t)k * n + a] = newDistance;
  distances[(size_t)k * n + b] = INFINITY;

  const int nearest = state.nearestSlot[k];
  if (nearest == a || nearest == b) {
    // k's nearest neighbour was merged. If the merged cluster is at least as
    // close as before it is still the nearest (nothing else got closer);
    // otherwise k has to look again.
    if (newDistance <= state.nearestDistance[k]) {
      state.nearestDistance[k] = newDistance;
      state.nearestSlot[k] = a;
    } else {
      state.rescanSlots[atomicAdd(state.rescanCount, 1)] = k;
    }
  } else if (newDistance < state.nearestDistance[k]) {
    // Impossible in exact arithmetic for reducible linkages, but rounding can
    // land a hair below; keep the bookkeeping consistent anyway.
    state.nearestDistance[k] = newDistance;
    state.nearestSlot[k] = a;
  }
}

// ---------------------------------------------------------------------------
// Host entry point

// Merge tree from a distance matrix that's already on the GPU: distances is
// numPoints x numPoints row-major (overwritten). Clusters may start out larger
// than one point: initialSize gives each slot's starting size and initialId
// its id in the tree (both device arrays; null means size 1 and id = slot).
// The cluster made by merge `step` gets id firstNewId + step (scipy's
// numbering: firstNewId = numPoints), so starting ids must not collide with
// those. deviceTree receives (numPoints-1) x 4 doubles.
inline cudaError_t denseLinkageOnMatrix(float *distances, const int *initialSize,
                                        const int *initialId, int firstNewId, int numPoints,
                                        Linkage linkage, double *deviceTree) {
  if (numPoints < 2)
    return cudaSuccess;
  const int n = numPoints;

  DenseLinkageState state{};
  state.distances = distances;
  state.numPoints = n;
  state.linkage = linkage;
  state.tree = deviceTree;
  state.initialSize = initialSize;
  state.initialId = initialId;
  state.firstNewId = firstNewId;

  cudaError_t status = cudaSuccess;
  auto check = [&](cudaError_t result) {
    if (status == cudaSuccess)
      status = result;
  };
  check(cudaMalloc(&state.nearestDistance, n * sizeof(float)));
  check(cudaMalloc(&state.nearestSlot, n * sizeof(int)));
  check(cudaMalloc(&state.clusterSize, n * sizeof(int)));
  check(cudaMalloc(&state.clusterId, n * sizeof(int)));
  check(cudaMalloc(&state.isActive, n * sizeof(unsigned char)));
  check(cudaMalloc(&state.rescanSlots, n * sizeof(int)));
  check(cudaMalloc(&state.rescanCount, sizeof(int)));
  check(cudaMalloc(&state.merge, sizeof(Merge)));

  if (status == cudaSuccess) {
    constexpr int kPerSlotBlock = 256, kReduceBlock = 1024, kRescanBlock = 256;
    const int perSlotBlocks = (n + kPerSlotBlock - 1) / kPerSlotBlock;
    int numSMs = 0;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    const int rescanBlocks = 4 * numSMs;

    initKernel<<<perSlotBlocks, kPerSlotBlock>>>(state);
    rescanKernel<kRescanBlock><<<rescanBlocks, kRescanBlock>>>(state);

    // No host synchronisation inside the loop: the merge to perform is decided
    // on the GPU and passed between kernels through *state.merge.
    for (int step = 0; step < n - 1; ++step) {
      findClosestPairKernel<kReduceBlock><<<1, kReduceBlock>>>(state, step);
      mergeKernel<<<perSlotBlocks, kPerSlotBlock>>>(state);
      rescanKernel<kRescanBlock><<<rescanBlocks, kRescanBlock>>>(state);
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
  }

  cudaFree(state.nearestDistance);
  cudaFree(state.nearestSlot);
  cudaFree(state.clusterSize);
  cudaFree(state.clusterId);
  cudaFree(state.isActive);
  cudaFree(state.rescanSlots);
  cudaFree(state.rescanCount);
  cudaFree(state.merge);
  return status;
}

// Builds the full merge tree for numPoints points. devicePoints is
// numPoints x numFeatures row-major; deviceTree receives (numPoints-1) x 4
// doubles. Needs numPoints^2 floats of GPU memory for the matrix.
template <typename Metric>
cudaError_t denseLinkage(const float *devicePoints, int numPoints, int numFeatures, Metric metric,
                         Linkage linkage, double *deviceTree) {
  if (numPoints < 2)
    return cudaSuccess;
  const int n = numPoints;
  float *distances = nullptr;
  cudaError_t status = cudaMalloc(&distances, (size_t)n * n * sizeof(float));
  if (status == cudaSuccess) {
    dist::pairwiseDistance(devicePoints, devicePoints, distances, n, n, numFeatures, metric);
    status = denseLinkageOnMatrix(distances, nullptr, nullptr, n, n, linkage, deviceTree);
  }
  cudaFree(distances);
  return status;
}

} // namespace hc
