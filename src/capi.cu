// C interface to the CUDA code, loaded from Python with ctypes
// (python/cuhc/_lib.py). Takes and returns host arrays; device memory is
// managed here.

#include "distance_matrix.cuh"
#include "hdbscan.cuh"
#include "linkage.cuh"
#include "mst.cuh"
#include "nn_chain.cuh"
#include "rnn_ward.cuh"

#include <cuda_runtime.h>

namespace {

// Must match METRIC_IDS in python/cuhc/_distances.py.
enum MetricId : int {
  kEuclidean = 0,
  kSqEuclidean = 1,
  kManhattan = 2,
  kChebyshev = 3,
  kMinkowski = 4,
  kCosine = 5,
};

constexpr int kErrorInvalidMetric = -1;
constexpr int kErrorInvalidShape = -2;
constexpr int kErrorInvalidLinkage = -3;
constexpr int kErrorInvalidAlgorithm = -4;
// hc::kErrorDisconnected = -5, hc::kErrorNonFinite = -6

// Must match ALGORITHM_IDS in python/cuhc/_linkage.py.
enum AlgorithmId : int {
  kDense = 0, // full distance matrix + Lance-Williams, any linkage
  kMst = 1,   // Borůvka minimum spanning tree, single linkage only
  kNnChain = 2, // nearest-neighbour chain on centroids, Ward (euclidean) only
  kRnn = 3,     // rounds of reciprocal nearest neighbours, Ward (euclidean) only
};

// Calls run(metric) with the metric struct for metricId; returns what run
// returns, or kErrorInvalidMetric.
template <typename Run> int withMetric(int metricId, float metricParam, Run run) {
  switch (metricId) {
  case kEuclidean:
    return run(metrics::Euclidean{});
  case kSqEuclidean:
    return run(metrics::SqEuclidean{});
  case kManhattan:
    return run(metrics::Manhattan{});
  case kChebyshev:
    return run(metrics::Chebyshev{});
  case kMinkowski:
    return run(metrics::Minkowski{metricParam});
  case kCosine:
    return run(metrics::Cosine{});
  default:
    return kErrorInvalidMetric;
  }
}

// Set by cuhc_set_pruning; Auto decides per call (spatial::shouldPrune).
spatial::PruneMode gPruneMode = spatial::PruneMode::Auto;
spatial::PruneStats gPruneStats;

// The points on the GPU, in spatial order if `prune`: then order[i] is the
// original index of row i. Without pruning the rows keep their order.
// (upload can't turn pruning off; decidePruning does, keeping the order.)
struct DevicePoints {
  float *points = nullptr;
  std::vector<int> order;
  bool ordered() const { return !order.empty(); }
  int upload(const float *hostPoints, int numPoints, int numFeatures, bool prune) {
    const size_t bytes = (size_t)numPoints * numFeatures * sizeof(float);
    int status = cudaMalloc(&points, bytes);
    if (status != cudaSuccess || !prune)
      return status == cudaSuccess ? cudaMemcpy(points, hostPoints, bytes, cudaMemcpyHostToDevice)
                                   : status;
    order = spatial::spatialOrder(hostPoints, numPoints, numFeatures);
    float *unordered = nullptr;
    int *deviceOrder = nullptr;
    status = cudaMalloc(&unordered, bytes);
    if (status == cudaSuccess)
      status = cudaMalloc(&deviceOrder, numPoints * sizeof(int));
    if (status == cudaSuccess)
      status = cudaMemcpy(unordered, hostPoints, bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess)
      status = cudaMemcpy(deviceOrder, order.data(), numPoints * sizeof(int),
                          cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
      spatial::gatherRowsKernel<<<1024, 256>>>(unordered, deviceOrder, numPoints, numFeatures,
                                               points);
      status = cudaGetLastError();
    }
    cudaFree(unordered);
    cudaFree(deviceOrder);
    return status;
  }
  ~DevicePoints() { cudaFree(points); }
};

// Whether to prune, for points in spatial order: in Auto mode, only if a
// sample of a pruned pass skips enough tiles. Sets `prune`; returns a
// cudaError_t.
template <typename Metric>
int decidePruning(const DevicePoints &ordered, int numPoints, int numFeatures, Metric metric,
                  bool &prune) {
  if constexpr (!metrics::HasBoxBound<Metric>::value) {
    prune = false; // shouldPrune already says so; this keeps the kernels uninstantiated
    return cudaSuccess;
  } else {
    if (!prune || gPruneMode == spatial::PruneMode::Always)
      return cudaSuccess;
    double share = 1.0;
    const int status =
        spatial::estimatePrunedShare(ordered.points, numPoints, numFeatures, metric, &share);
    prune = status == cudaSuccess && share <= spatial::kPruneGiveUpShare;
    return status;
  }
}

// Runs a sequence of CUDA calls, keeping the first error.
struct Status {
  int code = 0;
  void operator()(int result) {
    if (code == 0)
      code = result;
  }
  bool ok() const { return code == 0; }
};

} // namespace

extern "C" {

// distances (numPointsA x numPointsB) = metric(pointsA[i], pointsB[j]).
// pointsB may be null, meaning pointsB = pointsA (copied to the GPU once).
// Returns 0 on success, a cudaError_t value, or one of the negative codes above.
int cuhc_pairwise_distances(const float *hostPointsA, int numPointsA, const float *hostPointsB,
                            int numPointsB, int numFeatures, int metricId, float metricParam,
                            float *hostDistances) {
  if (numPointsA <= 0 || numPointsB <= 0 || numFeatures <= 0)
    return kErrorInvalidShape;

  const bool sameInput = hostPointsB == nullptr;
  const size_t bytesA = (size_t)numPointsA * numFeatures * sizeof(float);
  const size_t bytesB = (size_t)numPointsB * numFeatures * sizeof(float);
  const size_t bytesOut = (size_t)numPointsA * numPointsB * sizeof(float);

  float *devicePointsA = nullptr, *devicePointsB = nullptr, *deviceDistances = nullptr;
  Status status;
  status(cudaMalloc(&devicePointsA, bytesA));
  status(cudaMemcpy(devicePointsA, hostPointsA, bytesA, cudaMemcpyHostToDevice));
  if (sameInput) {
    devicePointsB = devicePointsA;
  } else {
    status(cudaMalloc(&devicePointsB, bytesB));
    status(cudaMemcpy(devicePointsB, hostPointsB, bytesB, cudaMemcpyHostToDevice));
  }
  status(cudaMalloc(&deviceDistances, bytesOut));
  if (status.ok())
    status(withMetric(metricId, metricParam, [&](auto metric) {
      dist::pairwiseDistance(devicePointsA, devicePointsB, deviceDistances, numPointsA,
                             numPointsB, numFeatures, metric);
      return (int)cudaGetLastError();
    }));
  status(cudaMemcpy(hostDistances, deviceDistances, bytesOut, cudaMemcpyDeviceToHost));

  cudaFree(devicePointsA);
  if (!sameInput)
    cudaFree(devicePointsB);
  cudaFree(deviceDistances);
  return status.code;
}

// Full merge tree in scipy's linkage format: hostTree receives
// (numPoints - 1) x 4 doubles {id1, id2, distance, size}. linkageId is
// hc::Linkage (must match LINKAGE_IDS in python/cuhc/_linkage.py);
// algorithmId is AlgorithmId.
int cuhc_linkage(const float *hostPoints, int numPoints, int numFeatures, int metricId,
                 float metricParam, int linkageId, int algorithmId, double *hostTree) {
  if (numPoints <= 0 || numFeatures <= 0)
    return kErrorInvalidShape;
  if (linkageId < (int)hc::Linkage::Single || linkageId > (int)hc::Linkage::Ward)
    return kErrorInvalidLinkage;
  const bool validAlgorithm =
      algorithmId == kDense ||
      (algorithmId == kMst && linkageId == (int)hc::Linkage::Single) ||
      ((algorithmId == kNnChain || algorithmId == kRnn) &&
       linkageId == (int)hc::Linkage::Ward && metricId == kEuclidean);
  if (!validAlgorithm)
    return kErrorInvalidAlgorithm;
  if (numPoints == 1)
    return 0; // nothing to merge

  const size_t pointBytes = (size_t)numPoints * numFeatures * sizeof(float);
  const size_t treeBytes = (size_t)(numPoints - 1) * 4 * sizeof(double);

  float *devicePoints = nullptr;
  double *deviceTree = nullptr;
  Status status;
  // rnn and mst upload their own copy (maybe in spatial order).
  if (algorithmId == kNnChain || algorithmId == kDense) {
    status(cudaMalloc(&devicePoints, pointBytes));
    status(cudaMemcpy(devicePoints, hostPoints, pointBytes, cudaMemcpyHostToDevice));
  }

  if (algorithmId == kNnChain) {
    if (status.ok())
      status(hc::nnChainWard(devicePoints, numPoints, numFeatures, hostTree));
  } else if (algorithmId == kRnn) {
    // Ward is Euclidean only, and the rounds work on squared distances.
    bool prune = spatial::shouldPrune<metrics::SqEuclidean>(gPruneMode, numPoints, numFeatures);
    DevicePoints ordered;
    status(ordered.upload(hostPoints, numPoints, numFeatures, prune));
    if (status.ok())
      status(decidePruning(ordered, numPoints, numFeatures, metrics::SqEuclidean{}, prune));
    hc::RnnOptions options;
    options.prune = prune;
    options.originalIndex = ordered.ordered() ? ordered.order.data() : nullptr;
    options.stats = &gPruneStats;
    if (status.ok())
      status(hc::rnnWard(ordered.points, numPoints, numFeatures, hostTree, options));
  } else if (algorithmId == kMst) {
    status(withMetric(metricId, metricParam, [&](auto metric) {
      bool prune = spatial::shouldPrune<decltype(metric)>(gPruneMode, numPoints, numFeatures);
      DevicePoints ordered;
      int result = ordered.upload(hostPoints, numPoints, numFeatures, prune);
      if (result == cudaSuccess)
        result = decidePruning(ordered, numPoints, numFeatures, metric, prune);
      std::vector<hc::MstEdge> edges;
      if (result == cudaSuccess)
        result = hc::boruvkaMst(ordered.points, numPoints, numFeatures, metric, edges,
                                dist::DistanceWeight{}, prune, &gPruneStats);
      if (result == cudaSuccess) {
        if (ordered.ordered())
          for (hc::MstEdge &edge : edges) {
            const int u = ordered.order[edge.u], v = ordered.order[edge.v];
            edge = hc::MstEdge{std::min(u, v), std::max(u, v), edge.weight};
          }
        hc::singleLinkageFromMst(std::move(edges), numPoints, hostTree);
      }
      return result;
    }));
  } else {
    status(cudaMalloc(&deviceTree, treeBytes));
    if (status.ok())
      status(withMetric(metricId, metricParam, [&](auto metric) {
        return (int)hc::denseLinkage(devicePoints, numPoints, numFeatures, metric,
                                     (hc::Linkage)linkageId, deviceTree);
      }));
    status(cudaMemcpy(hostTree, deviceTree, treeBytes, cudaMemcpyDeviceToHost));
  }

  cudaFree(devicePoints);
  cudaFree(deviceTree);
  return status.code;
}

// Ward through rnn rounds with explicit hand-over settings (hc::RnnOptions),
// so tests can force the dense and chain tails. Same output as cuhc_linkage.
int cuhc_ward_rnn_with_options(const float *hostPoints, int numPoints, int numFeatures,
                               float minProgress, int denseBelow, int allowDense,
                               double *hostTree) {
  if (numPoints <= 0 || numFeatures <= 0)
    return kErrorInvalidShape;
  const size_t pointBytes = (size_t)numPoints * numFeatures * sizeof(float);
  float *devicePoints = nullptr;
  Status status;
  status(cudaMalloc(&devicePoints, pointBytes));
  status(cudaMemcpy(devicePoints, hostPoints, pointBytes, cudaMemcpyHostToDevice));
  hc::RnnOptions options;
  options.minProgress = minProgress;
  options.denseBelow = denseBelow;
  options.allowDense = allowDense != 0;
  if (status.ok())
    status(hc::rnnWard(devicePoints, numPoints, numFeatures, hostTree, options));
  cudaFree(devicePoints);
  return status.code;
}

// HDBSCAN's single-linkage tree of the mutual reachability distance.
// hostTree receives (numPoints - 1) x 4 doubles (scipy linkage format);
// hostCore receives the numPoints core distances (may be null).
// coreMethod is hc::CoreMethod (0 = auto, 1 = fused, 2 = radix select).
int cuhc_hdbscan_tree(const float *hostPoints, int numPoints, int numFeatures, int metricId,
                      float metricParam, int minSamples, float alpha, int coreMethod,
                      double *hostTree, float *hostCore) {
  if (numPoints <= 0 || numFeatures <= 0 || minSamples < 1 || minSamples > numPoints)
    return kErrorInvalidShape;
  if (coreMethod < 0 || coreMethod > 2 ||
      (coreMethod == (int)hc::CoreMethod::Fused && minSamples > hc::kMaxFusedK))
    return kErrorInvalidAlgorithm;
  Status status;
  status(withMetric(metricId, metricParam, [&](auto metric) {
    bool prune = spatial::shouldPrune<decltype(metric)>(gPruneMode, numPoints, numFeatures);
    DevicePoints ordered;
    int result = ordered.upload(hostPoints, numPoints, numFeatures, prune);
    if (result == cudaSuccess)
      result = decidePruning(ordered, numPoints, numFeatures, metric, prune);
    if (result == cudaSuccess)
      result = hc::hdbscanSingleLinkage(ordered.points, numPoints, numFeatures, metric,
                                        minSamples, alpha, hostTree, hostCore,
                                        (hc::CoreMethod)coreMethod, prune,
                                        ordered.ordered() ? ordered.order.data() : nullptr,
                                        &gPruneStats);
    return result;
  }));
  return status.code;
}

// HDBSCAN's condensed tree from the (numPoints - 1) x 4 single-linkage tree.
// Writes up to `capacity` rows into the four output arrays and returns the
// number of rows, or -(rows needed) if capacity is too small.
long long cuhc_condense_tree(const double *tree, long long numPoints, long long minClusterSize,
                             long long capacity, long long *parent, long long *child,
                             double *lambda, long long *size) {
  const std::vector<hc::CondensedRow> rows = hc::condenseTree(tree, numPoints, minClusterSize);
  if ((long long)rows.size() > capacity)
    return -(long long)rows.size();
  for (size_t i = 0; i < rows.size(); ++i) {
    parent[i] = rows[i].parent;
    child[i] = rows[i].child;
    lambda[i] = rows[i].lambda;
    size[i] = rows[i].size;
  }
  return (long long)rows.size();
}

// Spatial pruning for the matrix-free paths (mst, rnn, HDBSCAN): 0 = auto
// (by size and dimension), 1 = always, 2 = never. Results are the same; only
// the speed changes. Returns kErrorInvalidAlgorithm for other values.
int cuhc_set_pruning(int mode) {
  if (mode < 0 || mode > 2)
    return kErrorInvalidAlgorithm;
  gPruneMode = (spatial::PruneMode)mode;
  return 0;
}

// Tiles computed and tiles in total by the pruned passes since the last call
// (then resets them).
void cuhc_prune_stats(unsigned long long *computed, unsigned long long *total) {
  *computed = gPruneStats.tilesComputed;
  *total = gPruneStats.tilesTotal;
  gPruneStats = spatial::PruneStats{};
}

const char *cuhc_error_string(int code) {
  switch (code) {
  case kErrorInvalidMetric:
    return "invalid metric id";
  case kErrorInvalidShape:
    return "array dimensions must be positive";
  case kErrorInvalidLinkage:
    return "invalid linkage id";
  case kErrorInvalidAlgorithm:
    return "invalid algorithm id for this linkage";
  case hc::kErrorNonFinite:
    return "distances overflowed float32 or are NaN; rescale the data";
  case hc::kErrorDisconnected:
    return "some points have no finite distance to the others (NaN or infinite values?)";
  default:
    return cudaGetErrorString((cudaError_t)code);
  }
}

} // extern "C"
