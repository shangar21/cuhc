#pragma once

#include "distance_matrix.cuh"
#include "linkage.cuh"
#include "mst.cuh"
#include "nn_chain.cuh"
#include "reductions.cuh"
#include "spatial.cuh"

#include <algorithm>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <utility>
#include <vector>

// Exact Ward linkage in rounds of reciprocal nearest neighbours, without the
// n x n distance matrix. Memory is O(n * numFeatures).
//
// Each round:
//   1. every active cluster finds its nearest active cluster under Ward's
//      distance: one all-pairs pass with the tiled kernel
//      (dist::nearestOtherComponent, every cluster its own "component",
//      WardWeight turning squared distances into Ward distances),
//   2. every pair of clusters that are each other's nearest merges, all at
//      once (correct for Ward for the same reason as the chain: reducibility,
//      see docs/rnn_ward.md),
//   3. the surviving clusters are compacted to slots 0..m'-1, so the next
//      pass is over m' x m' pairs.
//
// The all-pairs pass reuses every loaded centroid across a 128 x 128 tile, so
// it's limited by arithmetic, where the chain's one-against-all scans are
// limited by memory bandwidth. Rounds do 2-6 x n^2 Ward evaluations in total,
// similar to the chain's ~3 x n^2, but each costs far less.
//
// If a round merges too few clusters, the rest is handed to the dense
// algorithm on the m x m Ward matrix of the remaining clusters (if it fits),
// else to the chain. See RnnOptions.
//
// See docs/rnn_ward.md.

namespace hc {

// Ward distance squared from the squared Euclidean distance between two
// centroids: 2 * n_i * n_j / (n_i + n_j) * |c_i - c_j|^2.
__host__ __device__ __forceinline__ float wardFromSquared(float squaredDistance, float sizeI,
                                                          float sizeJ) {
  return 2.0f * (sizeI * sizeJ) / (sizeI + sizeJ) * squaredDistance;
}

// The same as an edge weight for the all-pairs pass, with the fast
// approximate division (__fdividef, 2 ulp): a full division per pair costs
// ~25% of the pass. Only which cluster is nearest is taken from it; merge
// heights are recomputed exactly (rnnMergeKernel). It's symmetric bit for bit
// (n_i * n_j and n_i + n_j are commutative), which every round relies on: the
// overall closest pair must see each other as nearest.
struct WardWeight {
  const int *size;
  using Value = float;
  __device__ __forceinline__ Value value(int cluster) const { return (float)size[cluster]; }
  __device__ __forceinline__ float operator()(float squaredDistance, Value sizeI,
                                              Value sizeJ) const {
    return __fdividef(2.0f * (sizeI * sizeJ) * squaredDistance, sizeI + sizeJ);
  }
};

struct RnnOptions {
  // Leave the rounds when one merges fewer than this share of its clusters:
  // each round costs a full pass, so rounds that merge little are wasted. On
  // ordinary data every round merges 5-60%; the bad case (e.g. points on a
  // line with growing gaps) merges one pair per round, ~1/m.
  float minProgress = 0.01f;
  // Also hand over once at most this many clusters are left (and their
  // matrix fits). Off by default: running the rounds to the end was faster at
  // every size measured. Tests use it to exercise the dense tail.
  int denseBelow = 0;
  // Whether a tail may use the dense algorithm at all (else always the
  // chain). Tests turn it off to exercise the chain tail.
  bool allowDense = true;
  // The points are in spatial order: prune the passes (spatial.cuh).
  bool prune = false;
  // Original index of each point (host, numPoints), if the points were
  // reordered; merges are recorded with these. Null = the identity.
  const int *originalIndex = nullptr;
  // Receives the pruned passes' statistics, if not null.
  spatial::PruneStats *stats = nullptr;
};

// One cluster set: centroids, sizes, and a representative point per slot
// (any point inside the cluster, used to record merges).
struct RnnClusters {
  float *centroids = nullptr;
  int *size = nullptr;
  int *rep = nullptr;
};

// The round's merge decision for slot i: the slot it pairs with, or -1.
__device__ __forceinline__ int reciprocalPartner(const unsigned long long *nearest, int i) {
  const unsigned long long packed = nearest[i];
  if (packed == kEmptyPackedMinLoc)
    return -1;
  const int j = unpackIndex(packed);
  return unpackIndex(nearest[j]) == i ? j : -1;
}

// survives[i] = 1 unless slot i is absorbed into a lower slot this round.
__global__ void rnnSurvivorsKernel(const unsigned long long *nearest, int numClusters,
                                   int *survives) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numClusters)
    return;
  const int partner = reciprocalPartner(nearest, i);
  survives[i] = !(partner >= 0 && partner < i);
}

// Writes the next round's clusters: surviving slot i goes to newIndex[i];
// if i pairs with a higher slot j, the merged cluster goes there instead and
// the merge is recorded. One thread per (slot, feature) so that centroid
// reads and writes are coalesced.
__global__ void rnnMergeKernel(RnnClusters current, RnnClusters next,
                               const unsigned long long *nearest, const int *newIndex,
                               int numClusters, int numFeatures, MstEdge *merges,
                               int *numMerges) {
  const size_t total = (size_t)numClusters * numFeatures;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < total;
       e += (size_t)gridDim.x * blockDim.x) {
    const int i = (int)(e / numFeatures), k = (int)(e % numFeatures);
    const int j = reciprocalPartner(nearest, i);
    if (j >= 0 && j < i)
      continue; // merged into slot j by j's threads
    const int to = newIndex[i];
    const float centroidI = current.centroids[e];
    float centroid = centroidI;
    if (j >= 0) {
      // Incremental form, as in the chain: exact for duplicate centroids.
      const float sizeI = (float)current.size[i], sizeJ = (float)current.size[j];
      centroid = fmaf(current.centroids[(size_t)j * numFeatures + k] - centroidI,
                      sizeJ / (sizeI + sizeJ), centroidI);
    }
    next.centroids[(size_t)to * numFeatures + k] = centroid;
    if (k == 0) {
      next.size[to] = current.size[i] + (j >= 0 ? current.size[j] : 0);
      next.rep[to] = current.rep[i];
      if (j >= 0) {
        // Exact height (the pass used an approximate division), summed in the
        // same order as metrics::SqEuclidean.
        const float *centroidsI = current.centroids + (size_t)i * numFeatures;
        const float *centroidsJ = current.centroids + (size_t)j * numFeatures;
        float squared = 0.0f;
        for (int f = 0; f < numFeatures; ++f) {
          const float diff = centroidsI[f] - centroidsJ[f];
          squared = fmaf(diff, diff, squared);
        }
        const int u = current.rep[i], v = current.rep[j];
        merges[atomicAdd(numMerges, 1)] = MstEdge{
            min(u, v), max(u, v),
            sqrtf(wardFromSquared(squared, (float)current.size[i], (float)current.size[j]))};
      }
    }
  }
}

// distances[i][j] (squared Euclidean between centroids) -> Ward distance.
__global__ void wardMatrixKernel(float *distances, const int *size, int numClusters) {
  const size_t total = (size_t)numClusters * numClusters;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < total;
       e += (size_t)gridDim.x * blockDim.x) {
    const int i = (int)(e / numClusters), j = (int)(e % numClusters);
    distances[e] = sqrtf(wardFromSquared(distances[e], (float)size[i], (float)size[j]));
  }
}

inline bool wardMatrixFits(int numClusters) {
  size_t freeBytes = 0, totalBytes = 0;
  if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess)
    return false;
  return 4.0 * numClusters * numClusters <= 0.9 * freeBytes;
}

// Dense tail: Ward matrix of the remaining clusters, then the dense
// algorithm. Appends the merges, named by representative point.
inline int rnnDenseTail(const RnnClusters &clusters, int numClusters, int numFeatures,
                        int numPoints, std::vector<MstEdge> &merges) {
  const int m = numClusters;
  float *distances = nullptr;
  double *deviceTree = nullptr;
  int status = cudaMalloc(&distances, (size_t)m * m * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceTree, (size_t)(m - 1) * 4 * sizeof(double));
  if (status == cudaSuccess) {
    dist::pairwiseDistance(clusters.centroids, clusters.centroids, distances, m, m, numFeatures,
                           metrics::SqEuclidean{});
    wardMatrixKernel<<<1024, 256>>>(distances, clusters.size, m);
    status = cudaGetLastError();
  }
  // Starting ids are representative points (< numPoints); new clusters are
  // numbered from numPoints, so the two can't collide.
  if (status == cudaSuccess)
    status = denseLinkageOnMatrix(distances, clusters.size, clusters.rep, numPoints, m,
                                  Linkage::Ward, deviceTree);
  std::vector<double> tree((size_t)(m - 1) * 4);
  if (status == cudaSuccess)
    status = cudaMemcpy(tree.data(), deviceTree, tree.size() * sizeof(double),
                        cudaMemcpyDeviceToHost);
  if (status == cudaSuccess) {
    // Tree ids -> representative points: a new cluster is represented by
    // the representative of its first child.
    std::vector<int> repOfStep(m - 1);
    // Ids outside that range only come from non-finite distances; they become
    // -1 and mergesFormTree rejects the result.
    auto repOf = [&](double id, int step) {
      const int cluster = id >= 0 && id < numPoints + step ? (int)id : -1;
      return cluster < numPoints ? cluster : repOfStep[cluster - numPoints];
    };
    for (int step = 0; step < m - 1; ++step) {
      const double *row = &tree[(size_t)step * 4];
      const int u = repOf(row[0], step), v = repOf(row[1], step);
      merges.push_back(MstEdge{std::min(u, v), std::max(u, v), (float)row[2]});
      repOfStep[step] = u;
    }
  }
  cudaFree(distances);
  cudaFree(deviceTree);
  return status;
}

// Chain tail on the remaining clusters. Appends the merges, named by
// representative point.
inline int rnnChainTail(const RnnClusters &clusters, int numClusters, int numFeatures,
                        std::vector<MstEdge> &merges) {
  std::vector<MstEdge> tail;
  int status = nnChainWardMerges(clusters.centroids, clusters.size, numClusters, numFeatures, tail);
  std::vector<int> rep(numClusters);
  if (status == cudaSuccess)
    status = cudaMemcpy(rep.data(), clusters.rep, numClusters * sizeof(int),
                        cudaMemcpyDeviceToHost);
  if (status == cudaSuccess)
    for (const MstEdge &merge : tail) {
      const int u = rep[merge.u], v = rep[merge.v];
      merges.push_back(MstEdge{std::min(u, v), std::max(u, v), merge.weight});
    }
  return status;
}

// Full Ward merge tree for numPoints points (devicePoints, numPoints x
// numFeatures row-major on the GPU). tree receives (numPoints - 1) x 4 doubles
// (host memory) in scipy's linkage format. Returns a cudaError_t, or
// kErrorNonFinite if distances overflowed float32.
inline int rnnWard(const float *devicePoints, int numPoints, int numFeatures, double *tree,
                   RnnOptions options = {}) {
  const int n = numPoints, d = numFeatures;
  if (n < 2)
    return cudaSuccess;

  RnnClusters current, next;
  int *component = nullptr, *survives = nullptr, *newIndex = nullptr, *numMerges = nullptr;
  unsigned long long *nearest = nullptr;
  MstEdge *deviceMerges = nullptr;
  void *scanTemp = nullptr;
  size_t scanTempBytes = 0;

  int status = cudaSuccess;
  auto check = [&](int result) {
    if (status == cudaSuccess)
      status = result;
  };
  for (RnnClusters *clusters : {&current, &next}) {
    check(cudaMalloc(&clusters->centroids, (size_t)n * d * sizeof(float)));
    check(cudaMalloc(&clusters->size, n * sizeof(int)));
    check(cudaMalloc(&clusters->rep, n * sizeof(int)));
  }
  check(cudaMalloc(&component, n * sizeof(int)));
  check(cudaMalloc(&survives, n * sizeof(int)));
  check(cudaMalloc(&newIndex, n * sizeof(int)));
  check(cudaMalloc(&numMerges, sizeof(int)));
  check(cudaMalloc(&nearest, n * sizeof(unsigned long long)));
  check(cudaMalloc(&deviceMerges, (n - 1) * sizeof(MstEdge)));
  check(cub::DeviceScan::ExclusiveSum(nullptr, scanTempBytes, survives, newIndex, n));
  check(cudaMalloc(&scanTemp, scanTempBytes));

  if (status == cudaSuccess) {
    // Every point is its own cluster, represented by itself; each slot is its
    // own "component" for the nearest-other-component pass.
    std::vector<int> iota(n);
    for (int i = 0; i < n; ++i)
      iota[i] = i;
    const std::vector<int> ones(n, 1);
    const int zero = 0;
    check(cudaMemcpy(current.centroids, devicePoints, (size_t)n * d * sizeof(float),
                     cudaMemcpyDeviceToDevice));
    check(cudaMemcpy(current.size, ones.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(current.rep, options.originalIndex ? options.originalIndex : iota.data(),
                     n * sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(component, iota.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(numMerges, &zero, sizeof(int), cudaMemcpyHostToDevice));
  }

  // Round r's merges are deviceMerges[roundEnd[r-1], roundEnd[r]).
  std::vector<int> roundEnd;
  int m = n, merged = 0;
  bool pruneThisPass = options.prune; // off for good once a pass skips too little
  enum class Tail { None, Dense, Chain } tail = Tail::None;
  while (status == cudaSuccess && m > 1) {
    if (options.allowDense && m <= options.denseBelow && wardMatrixFits(m)) {
      tail = Tail::Dense;
      break;
    }
    check(cudaMemset(nearest, 0xff, m * sizeof(unsigned long long)));
    if (pruneThisPass) {
      double computedShare = 0.0;
      check(spatial::nearestOtherComponentPruned(current.centroids, m, d, metrics::SqEuclidean{},
                                                 WardWeight{current.size}, component, nearest,
                                                 &computedShare, options.stats));
      pruneThisPass = computedShare <= spatial::kPruneGiveUpShare;
    } else
      dist::nearestOtherComponent(current.centroids, m, d, metrics::SqEuclidean{},
                                  WardWeight{current.size}, component, nearest);
    rnnSurvivorsKernel<<<(m + 255) / 256, 256>>>(nearest, m, survives);
    check(cub::DeviceScan::ExclusiveSum(scanTemp, scanTempBytes, survives, newIndex, m));
    rnnMergeKernel<<<1024, 256>>>(current, next, nearest, newIndex, m, d, deviceMerges, numMerges);
    check(cudaGetLastError());

    int total = 0;
    check(cudaMemcpy(&total, numMerges, sizeof(int), cudaMemcpyDeviceToHost));
    const int roundMerges = total - merged;
    merged = total;
    roundEnd.push_back(total);
    std::swap(current, next);
    const int before = m;
    m -= roundMerges;
    if (m > 1 && (roundMerges == 0 || roundMerges < options.minProgress * before)) {
      tail = options.allowDense && wardMatrixFits(m) ? Tail::Dense : Tail::Chain;
      break;
    }
  }

  std::vector<MstEdge> merges(merged);
  check(cudaMemcpy(merges.data(), deviceMerges, merged * sizeof(MstEdge),
                   cudaMemcpyDeviceToHost));
  // Within a round the merges arrive in atomic order; sort each round so the
  // result doesn't depend on scheduling. Rounds stay in order (child first).
  for (size_t r = 0; r < roundEnd.size(); ++r)
    std::sort(merges.begin() + (r == 0 ? 0 : roundEnd[r - 1]), merges.begin() + roundEnd[r],
              edgeLess);
  if (status == cudaSuccess && tail == Tail::Dense)
    check(rnnDenseTail(current, m, d, n, merges));
  if (status == cudaSuccess && tail == Tail::Chain)
    check(rnnChainTail(current, m, d, merges));
  if (status == cudaSuccess && !mergesFormTree(merges, n))
    status = kErrorNonFinite;
  if (status == cudaSuccess) {
    orderChainMerges(merges, n);
    treeFromOrderedMerges(merges, n, tree);
  }

  for (RnnClusters *clusters : {&current, &next}) {
    cudaFree(clusters->centroids);
    cudaFree(clusters->size);
    cudaFree(clusters->rep);
  }
  cudaFree(component);
  cudaFree(survives);
  cudaFree(newIndex);
  cudaFree(numMerges);
  cudaFree(nearest);
  cudaFree(deviceMerges);
  cudaFree(scanTemp);
  return status;
}

} // namespace hc
