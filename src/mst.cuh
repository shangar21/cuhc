#pragma once

#include "distance_matrix.cuh"
#include "reductions.cuh"
#include "spatial.cuh"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>

// Exact minimum spanning tree (MST) of the complete graph on the points, with
// edge weight = metric distance, without storing the distance matrix.
// Single linkage is the MST with its edges merged in order of weight.
//
// Borůvka's algorithm:
//   every point starts as its own component
//   each round:
//     GPU: for every point, its nearest point in a *different* component
//          (one fused pass over all pairs, O(n) memory)
//     CPU: for every component, the shortest of its points' edges; add those
//          edges and merge the components they join
//   the number of components at least halves per round -> <= log2(n) rounds.
//
// Each round's edges are safe to add together because every one of them is
// the shortest edge leaving its component (the "cut property"). Ties are
// broken with one global order -- (weight, smaller endpoint, larger endpoint)
// -- which keeps the result a tree; union-find also skips any edge whose
// endpoints are already connected (e.g. two components picking the same edge).
//
// See docs/boruvka_mst.md.

namespace hc {

struct MstEdge {
  int u, v; // u < v
  float weight;
};

// The global edge order used for tie-breaking.
inline bool edgeLess(const MstEdge &x, const MstEdge &y) {
  if (x.weight != y.weight)
    return x.weight < y.weight;
  if (x.u != y.u)
    return x.u < y.u;
  return x.v < y.v;
}

// Disjoint sets with path halving and union by size.
class UnionFind {
public:
  explicit UnionFind(int n) : parent_(n), size_(n, 1) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }
  int find(int x) {
    while (parent_[x] != x) {
      parent_[x] = parent_[parent_[x]];
      x = parent_[x];
    }
    return x;
  }
  // Joins the sets of x and y; returns the new root, or -1 if already joined.
  int unite(int x, int y) {
    x = find(x);
    y = find(y);
    if (x == y)
      return -1;
    if (size_[x] < size_[y])
      std::swap(x, y);
    parent_[y] = x;
    size_[x] += size_[y];
    return x;
  }

private:
  std::vector<int> parent_, size_;
};

constexpr int kErrorDisconnected = -5;
constexpr int kErrorNonFinite = -6;

// Fills edges with the n-1 MST edges (in the order they were found).
// devicePoints is numPoints x numFeatures row-major on the GPU. Edge weights
// come from edgeWeight (see distance_matrix.cuh): the distance itself by
// default, or e.g. HDBSCAN's mutual reachability. Returns a cudaError_t, or
// kErrorDisconnected if some points have no finite edge to the rest (NaN or
// infinite weights).
//
// prune: the points are in spatial order (spatial::spatialOrder), so the
// passes can skip tile pairs that can't hold a nearest edge. Same result.
// Ignored for metrics without metrics::HasBoxBound. If a pass skips too few
// tiles (spatial::kPruneGiveUpShare), the remaining passes don't prune.
// stats, if not null, receives how many tiles the pruned passes computed.
template <typename Metric, typename EdgeWeight = dist::DistanceWeight>
int boruvkaMst(const float *devicePoints, int numPoints, int numFeatures, Metric metric,
               std::vector<MstEdge> &edges, EdgeWeight edgeWeight = {}, bool prune = false,
               spatial::PruneStats *stats = nullptr) {
  const int n = numPoints;
  edges.clear();
  edges.reserve(n > 0 ? n - 1 : 0);
  if (n < 2)
    return cudaSuccess;

  std::vector<int> component(n);
  std::iota(component.begin(), component.end(), 0);
  std::vector<unsigned long long> nearest(n);
  UnionFind sets(n);

  int *deviceComponent = nullptr;
  unsigned long long *deviceNearest = nullptr;
  int status = cudaMalloc(&deviceComponent, n * sizeof(int));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceNearest, n * sizeof(unsigned long long));

  int numComponents = n;
  bool pruneThisPass = prune;
  while (status == cudaSuccess && numComponents > 1) {
    // GPU: nearest point in another component, for every point.
    status = cudaMemcpy(deviceComponent, component.data(), n * sizeof(int),
                        cudaMemcpyHostToDevice);
    if (status == cudaSuccess)
      status = cudaMemset(deviceNearest, 0xff, n * sizeof(unsigned long long));
    if (status == cudaSuccess) {
      if constexpr (metrics::HasBoxBound<Metric>::value) {
        if (pruneThisPass) {
          double computedShare = 0.0;
          status = spatial::nearestOtherComponentPruned(devicePoints, n, numFeatures, metric,
                                                        edgeWeight, deviceComponent,
                                                        deviceNearest, &computedShare, stats);
          pruneThisPass = computedShare <= spatial::kPruneGiveUpShare;
        } else {
          dist::nearestOtherComponent(devicePoints, n, numFeatures, metric, edgeWeight,
                                      deviceComponent, deviceNearest);
          status = cudaGetLastError();
        }
      } else {
        dist::nearestOtherComponent(devicePoints, n, numFeatures, metric, edgeWeight,
                                    deviceComponent, deviceNearest);
        status = cudaGetLastError();
      }
    }
    if (status == cudaSuccess)
      status = cudaMemcpy(nearest.data(), deviceNearest, n * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
      break;

    // CPU: shortest edge leaving each component.
    std::vector<MstEdge> shortest(numComponents, MstEdge{-1, -1, 0.0f});
    for (int point = 0; point < n; ++point) {
      if (nearest[point] == kEmptyPackedMinLoc)
        continue;
      const int other = unpackIndex(nearest[point]);
      const MstEdge edge{std::min(point, other), std::max(point, other),
                         unpackDistance(nearest[point])};
      MstEdge &best = shortest[component[point]];
      if (best.u < 0 || edgeLess(edge, best))
        best = edge;
    }

    // Add them; two components may have chosen the same edge.
    const size_t edgesBefore = edges.size();
    for (const MstEdge &edge : shortest)
      if (edge.u >= 0 && sets.unite(edge.u, edge.v) >= 0)
        edges.push_back(edge);
    if (edges.size() == edgesBefore) {
      status = kErrorDisconnected;
      break;
    }

    // Relabel components as 0..numComponents-1.
    std::vector<int> label(n, -1);
    numComponents = 0;
    for (int point = 0; point < n; ++point) {
      const int root = sets.find(point);
      if (label[root] < 0)
        label[root] = numComponents++;
      component[point] = label[root];
    }
  }

  cudaFree(deviceComponent);
  cudaFree(deviceNearest);
  return status;
}

// Whether merges can form a tree of numPoints points: numPoints - 1 merges,
// finite heights, each joining two clusters not yet joined. Non-finite
// distances (float32 overflow, NaN) break that, and the tree-building code
// below would then index out of bounds.
inline bool mergesFormTree(const std::vector<MstEdge> &merges, int numPoints) {
  if ((int)merges.size() != numPoints - 1)
    return false;
  UnionFind sets(numPoints);
  for (const MstEdge &merge : merges) {
    if (!std::isfinite(merge.weight) || merge.u < 0 || merge.u >= numPoints || merge.v < 0 ||
        merge.v >= numPoints || sets.unite(merge.u, merge.v) < 0)
      return false;
  }
  return true;
}

// Merge tree in scipy's linkage format from merges already in tree order
// (every merge after the merges that created its two clusters, heights
// non-decreasing). Each merge names its clusters by any point inside them;
// union-find finds the clusters. tree receives (n-1) x 4 doubles.
inline void treeFromOrderedMerges(const std::vector<MstEdge> &merges, int numPoints,
                                  double *tree) {
  UnionFind sets(numPoints);
  std::vector<int> clusterId(numPoints), clusterSize(numPoints, 1); // indexed by set root
  std::iota(clusterId.begin(), clusterId.end(), 0);

  for (size_t step = 0; step < merges.size(); ++step) {
    const int rootU = sets.find(merges[step].u), rootV = sets.find(merges[step].v);
    const int idU = clusterId[rootU], idV = clusterId[rootV];
    const int size = clusterSize[rootU] + clusterSize[rootV];
    double *row = tree + step * 4;
    row[0] = std::min(idU, idV);
    row[1] = std::max(idU, idV);
    row[2] = merges[step].weight;
    row[3] = size;

    const int root = sets.unite(rootU, rootV);
    clusterId[root] = numPoints + (int)step;
    clusterSize[root] = size;
  }
}

// Single-linkage merge tree from an MST: merge along the edges in increasing
// weight. tree receives (n-1) x 4 doubles in scipy's linkage format.
inline void singleLinkageFromMst(std::vector<MstEdge> edges, int numPoints, double *tree) {
  std::sort(edges.begin(), edges.end(), edgeLess);
  treeFromOrderedMerges(edges, numPoints, tree);
}

} // namespace hc
