#pragma once

#include "mst.cuh"
#include "reductions.cuh"

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <vector>

// Exact Ward linkage without the n x n distance matrix: the nearest-neighbour
// chain algorithm on cluster centroids. Memory is O(n * numFeatures).
//
// Ward's distance between clusters a and b depends only on their sizes and
// centroids:
//     ward(a, b) = sqrt(2 * n_a * n_b / (n_a + n_b)) * |c_a - c_b|
// (equal to the Euclidean distance for two single points; the same heights as
// scipy and sklearn).
//
// Nearest-neighbour chain:
//     chain = [any cluster]
//     repeat until one cluster is left:
//         a = top of chain;  b = nearest active cluster to a
//         if b is the cluster below a in the chain: pop both, merge them
//         else: push b
// Valid for "reducible" linkages (merging two clusters never brings the result
// closer to a third), which Ward is. Ties prefer the cluster below a in the
// chain, which guarantees progress. Merges come out in chain order, not sorted
// by height; they are sorted (stably) at the end, as scipy does.
//
// Each step is one "nearest active cluster to a" query over all slots. All
// ~3n steps run inside one cooperative kernel: every block finds the nearest
// slot in its range, grid.sync(), block 0 combines the block results and
// updates the chain (merging if needed), grid.sync(), next step. That avoids
// a round trip to the CPU per step.
//
// Clusters live in slots like the dense linkage: merging the clusters in slots
// a and b puts the merged cluster in slot min(a, b), and size[] = 0 marks a
// retired slot. A slot's cluster always contains the point with the slot's
// index, which is how merges are recorded (by representative point).
//
// See docs/nn_chain.md.

namespace hc {

struct NnChainState {
  float *centroids;   // numPoints x numFeatures; slot s = centroid of the cluster in slot s
  int *size;          // points in the cluster in each slot; 0 = retired
  int *chain;         // the chain of slots, bottom first
  int *chainLength;
  int *current;       // slot whose nearest neighbour the next step looks for; -1 = done
  int *nextStart;     // scan position for an active slot to start a new chain
  MinLoc *blockBest;  // one per block
  MstEdge *merges;    // (slot, slot, Ward distance) in the order they happen
  int *numMerges;
  int numPoints, numFeatures;
};

// Ward distance squared; comparing squares avoids a sqrt per candidate.
__device__ __forceinline__ float wardSquared(const float *centroidA, const float *centroidB,
                                             int numFeatures, float sizeA, float sizeB) {
  float squared = 0.0f;
  for (int k = 0; k < numFeatures; ++k) {
    const float diff = centroidA[k] - centroidB[k];
    squared = fmaf(diff, diff, squared);
  }
  return 2.0f * sizeA * sizeB / (sizeA + sizeB) * squared;
}

template <int BlockSize>
__global__ void __launch_bounds__(BlockSize) nnChainWardKernel(NnChainState state) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  extern __shared__ float centroidA[]; // numFeatures floats: the current slot's centroid
  __shared__ int currentSlot, previousSlot, mergeWith;

  const int n = state.numPoints, d = state.numFeatures;
  // Each block owns one contiguous range of slots (so a per-range bounding
  // box test could skip a whole range later; see docs/nn_chain.md).
  const int slotsPerBlock = (n + gridDim.x - 1) / gridDim.x;
  const int firstSlot = blockIdx.x * slotsPerBlock;
  const int endSlot = min(n, firstSlot + slotsPerBlock);

  while (true) {
    if (threadIdx.x == 0) {
      currentSlot = *state.current;
      const int length = *state.chainLength;
      previousSlot = length >= 2 ? state.chain[length - 2] : -1;
    }
    __syncthreads();
    const int a = currentSlot, previous = previousSlot;
    if (a < 0)
      return; // every block sees the same value, so all leave together
    for (int k = threadIdx.x; k < d; k += BlockSize)
      centroidA[k] = state.centroids[(size_t)a * d + k];
    __syncthreads();

    // 1. Nearest active slot to a within this block's range. The previous
    //    slot in the chain gets index -1 so that it wins ties.
    const float sizeA = (float)state.size[a];
    MinLoc best = noMinLoc();
    for (int slot = firstSlot + threadIdx.x; slot < endSlot; slot += BlockSize) {
      const int sizeSlot = state.size[slot];
      if (sizeSlot == 0 || slot == a)
        continue;
      const float distance = wardSquared(centroidA, state.centroids + (size_t)slot * d, d, sizeA,
                                         (float)sizeSlot);
      best = minLoc(best, MinLoc{distance, slot == previous ? -1 : slot});
    }
    best = blockMinLoc<BlockSize>(best);
    if (threadIdx.x == 0)
      state.blockBest[blockIdx.x] = best;
    grid.sync();

    // 2. Block 0: overall nearest; push it, or merge it with a.
    if (blockIdx.x == 0) {
      MinLoc overall = noMinLoc();
      for (int block = threadIdx.x; block < gridDim.x; block += BlockSize)
        overall = minLoc(overall, state.blockBest[block]);
      overall = blockMinLoc<BlockSize>(overall);
      if (threadIdx.x == 0) {
        const int nearest = overall.index == -1 ? previous : overall.index;
        if (nearest == previous) {
          mergeWith = previous;
        } else {
          mergeWith = -1;
          state.chain[(*state.chainLength)++] = nearest;
          *state.current = nearest;
        }
      }
      __syncthreads();

      if (mergeWith >= 0) {
        const int b = mergeWith;
        const int keep = min(a, b), retire = max(a, b);
        const float sizeB = (float)state.size[b];
        // Merged centroid = size-weighted mean, written into slot `keep`, in
        // the incremental form c_a + (c_b - c_a) * n_b / (n_a + n_b): when the
        // centroids are equal (duplicate points) the result is exactly c_a,
        // which (n_a * c_a + n_b * c_b) / (n_a + n_b) isn't in float.
        // centroidA is a shared copy, so overwriting slot a is safe.
        const float weightB = sizeB / (sizeA + sizeB);
        for (int k = threadIdx.x; k < d; k += BlockSize)
          state.centroids[(size_t)keep * d + k] =
              fmaf(state.centroids[(size_t)b * d + k] - centroidA[k], weightB, centroidA[k]);
        if (threadIdx.x == 0) {
          state.merges[(*state.numMerges)++] = MstEdge{keep, retire, sqrtf(overall.distance)};
          state.size[keep] = state.size[a] + state.size[b];
          state.size[retire] = 0;
          *state.chainLength -= 2;
          if (*state.numMerges == n - 1) {
            *state.current = -1; // done
          } else if (*state.chainLength > 0) {
            *state.current = state.chain[*state.chainLength - 1];
          } else {
            // Start a new chain at the first active slot. Slots only retire,
            // so the scan position never has to move back.
            int start = *state.nextStart;
            while (state.size[start] == 0)
              ++start;
            *state.nextStart = start;
            state.chain[0] = start;
            *state.chainLength = 1;
            *state.current = start;
          }
        }
      }
    }
    grid.sync();
  }
}

// Merges in chain order -> merges in tree order. Heights are made
// non-decreasing up the tree first (a merge's height becomes at least its
// children's): exact arithmetic guarantees that for Ward, but float rounding
// can leave a parent an ulp below a child, and the stable sort by height must
// never put a parent before its children.
inline void orderChainMerges(std::vector<MstEdge> &merges, int numPoints) {
  UnionFind sets(numPoints);
  std::vector<float> height(numPoints, 0.0f); // per set root: height of the cluster
  for (MstEdge &merge : merges) {
    const int rootU = sets.find(merge.u), rootV = sets.find(merge.v);
    merge.weight = std::max({merge.weight, height[rootU], height[rootV]});
    height[sets.unite(rootU, rootV)] = merge.weight;
  }
  std::stable_sort(merges.begin(), merges.end(),
                   [](const MstEdge &x, const MstEdge &y) { return x.weight < y.weight; });
}

// Runs the chain on numClusters clusters with the given starting centroids
// (numClusters x numFeatures, device) and sizes (device; null = all 1); both
// are copied, not modified. merges receives the numClusters - 1 merges in
// chain order (a valid child-first order, heights not sorted), naming
// clusters by slot: the cluster in slot s always contains starting cluster s.
// Returns a cudaError_t.
inline int nnChainWardMerges(const float *deviceCentroids, const int *deviceSizes,
                             int numClusters, int numFeatures, std::vector<MstEdge> &merges) {
  const int n = numClusters;
  merges.assign(n > 1 ? n - 1 : 0, MstEdge{});
  if (n < 2)
    return cudaSuccess;
  constexpr int kBlockSize = 256;

  int supportsCooperative = 0;
  cudaDeviceGetAttribute(&supportsCooperative, cudaDevAttrCooperativeLaunch, 0);
  if (!supportsCooperative)
    return cudaErrorNotSupported;
  const size_t sharedBytes = (size_t)numFeatures * sizeof(float);
  int blocksPerSM = 0, numSMs = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, nnChainWardKernel<kBlockSize>,
                                                kBlockSize, sharedBytes);
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  // Every block must be resident at once for grid.sync(); more blocks than
  // slots would only add idle blocks.
  const int numBlocks = std::min(blocksPerSM * numSMs, (n + kBlockSize - 1) / kBlockSize);
  if (numBlocks < 1)
    return cudaErrorInvalidConfiguration;

  NnChainState state{};
  state.numPoints = n;
  state.numFeatures = numFeatures;
  int status = cudaSuccess;
  auto check = [&](cudaError_t result) {
    if (status == cudaSuccess)
      status = result;
  };
  check(cudaMalloc(&state.centroids, (size_t)n * numFeatures * sizeof(float)));
  check(cudaMalloc(&state.size, n * sizeof(int)));
  check(cudaMalloc(&state.chain, n * sizeof(int)));
  check(cudaMalloc(&state.chainLength, sizeof(int)));
  check(cudaMalloc(&state.current, sizeof(int)));
  check(cudaMalloc(&state.nextStart, sizeof(int)));
  check(cudaMalloc(&state.blockBest, numBlocks * sizeof(MinLoc)));
  check(cudaMalloc(&state.merges, (n - 1) * sizeof(MstEdge)));
  check(cudaMalloc(&state.numMerges, sizeof(int)));

  if (status == cudaSuccess) {
    // The chain starts at slot 0.
    const int zero = 0, one = 1;
    check(cudaMemcpy(state.centroids, deviceCentroids, (size_t)n * numFeatures * sizeof(float),
                     cudaMemcpyDeviceToDevice));
    if (deviceSizes) {
      check(cudaMemcpy(state.size, deviceSizes, n * sizeof(int), cudaMemcpyDeviceToDevice));
    } else {
      const std::vector<int> ones(n, 1);
      check(cudaMemcpy(state.size, ones.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    }
    check(cudaMemcpy(state.chain, &zero, sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(state.chainLength, &one, sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(state.current, &zero, sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(state.nextStart, &zero, sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(state.numMerges, &zero, sizeof(int), cudaMemcpyHostToDevice));
  }
  if (status == cudaSuccess) {
    void *args[] = {&state};
    check(cudaLaunchCooperativeKernel((void *)nnChainWardKernel<kBlockSize>, numBlocks, kBlockSize,
                                      args, sharedBytes, 0));
    check(cudaDeviceSynchronize());
  }
  check(cudaMemcpy(merges.data(), state.merges, (n - 1) * sizeof(MstEdge),
                   cudaMemcpyDeviceToHost));

  cudaFree(state.centroids);
  cudaFree(state.size);
  cudaFree(state.chain);
  cudaFree(state.chainLength);
  cudaFree(state.current);
  cudaFree(state.nextStart);
  cudaFree(state.blockBest);
  cudaFree(state.merges);
  cudaFree(state.numMerges);
  return status;
}

// Full Ward merge tree for numPoints points (devicePoints, numPoints x
// numFeatures row-major on the GPU). tree receives (numPoints - 1) x 4 doubles
// (host memory) in scipy's linkage format. Returns a cudaError_t, or
// kErrorNonFinite if distances overflowed float32.
inline int nnChainWard(const float *devicePoints, int numPoints, int numFeatures, double *tree) {
  std::vector<MstEdge> merges;
  int status = nnChainWardMerges(devicePoints, nullptr, numPoints, numFeatures, merges);
  if (status == cudaSuccess && numPoints > 1 && !mergesFormTree(merges, numPoints))
    status = kErrorNonFinite;
  if (status == cudaSuccess && numPoints > 1) {
    orderChainMerges(merges, numPoints);
    treeFromOrderedMerges(merges, numPoints, tree);
  }
  return status;
}

} // namespace hc
