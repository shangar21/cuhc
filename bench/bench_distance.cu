// Benchmarks the pairwise distance kernels on random data.
// Usage: bench_distance [numPoints=16384] [numFeatures=64] [metric=all]

#include "distance_matrix.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t err_ = (call);                                                                     \
    if (err_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,         \
                   __LINE__);                                                                      \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

// Average milliseconds per call of f, after one warm-up call.
template <typename F> float timeMs(F f, int iters) {
  f();
  CUDA_CHECK(cudaGetLastError());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int it = 0; it < iters; ++it)
    f();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

// Same metric functor, run on the host.
template <typename Metric>
float cpuDistance(const float *a, const float *b, int numFeatures, const Metric &metric) {
  auto acc = metric.init();
  for (int k = 0; k < numFeatures; ++k)
    acc = metric.accumulate(acc, a[k], b[k]);
  return metric.finalize(acc);
}

int main(int argc, char **argv) {
  const int numPoints = argc > 1 ? std::atoi(argv[1]) : 16384;
  const int numFeatures = argc > 2 ? std::atoi(argv[2]) : 64;
  const char *onlyMetric = argc > 3 ? argv[3] : nullptr;
  const size_t outputBytes = (size_t)numPoints * numPoints * sizeof(float);

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("%s | numPoints = %d, numFeatures = %d | distance matrix is %.2f GB\n\n", prop.name,
              numPoints, numFeatures, outputBytes / 1e9);

  std::vector<float> hostPoints((size_t)numPoints * numFeatures);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  for (auto &v : hostPoints)
    v = uni(rng);

  float *devicePoints, *deviceDistances;
  CUDA_CHECK(cudaMalloc(&devicePoints, hostPoints.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceDistances, outputBytes));
  CUDA_CHECK(cudaMemcpy(devicePoints, hostPoints.data(), hostPoints.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Reference points for "how fast could this be":
  //  - memset: just writing numPoints*numPoints floats, no math at all.
  //  - SGEMM:  X * X^T in FP32 (no tensor cores): the same inner loop as
  //            our kernels with the best-tuned code NVIDIA ships.
  const float memsetMs = timeMs([&] { cudaMemsetAsync(deviceDistances, 0, outputBytes); }, 10);
  cublasHandle_t handle;
  cublasCreate(&handle);
  const float one = 1.0f, zero = 0.0f;
  // Row-major X (numPoints x numFeatures) is column-major X^T (numFeatures x numPoints), so C =
  // (X^T)^T (X^T) = X X^T.
  const float gemmMs = timeMs(
      [&] {
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, numPoints, numPoints, numFeatures, &one,
                    devicePoints, numFeatures, devicePoints, numFeatures, &zero, deviceDistances,
                    numPoints);
      },
      10);
  cublasDestroy(handle);
  std::printf("floor: write-only memset %8.2f ms (%.0f GB/s)\n", memsetMs,
              outputBytes / memsetMs / 1e6);
  std::printf("floor: cuBLAS SGEMM      %8.2f ms (%.1f TFLOP/s)\n\n", gemmMs,
              2.0 * numPoints * numPoints * numFeatures / gemmMs / 1e9);

  std::printf("%-12s %10s %12s %10s\n", "metric", "ms", "Gpairs/s", "max err");

  // Rows checked against the CPU (a full check would take minutes for large numPoints).
  const int numRowsToCheck = 16;
  std::vector<float> gpuRow(numPoints);

  auto run = [&](const char *name, auto metric) {
    if (onlyMetric && std::strcmp(onlyMetric, name) != 0)
      return;
    const float kernelMs = timeMs(
        [&] {
          dist::pairwiseDistance(devicePoints, devicePoints, deviceDistances, numPoints, numPoints,
                                 numFeatures, metric);
        },
        10);

    float maxRelativeError = 0.0f;
    for (int check = 0; check < numRowsToCheck; ++check) {
      const int row = (int)((long long)check * (numPoints - 1) / (numRowsToCheck - 1));
      CUDA_CHECK(cudaMemcpy(gpuRow.data(), deviceDistances + (size_t)row * numPoints,
                            numPoints * sizeof(float), cudaMemcpyDeviceToHost));
      for (int col = 0; col < numPoints; ++col) {
        const float expected =
            cpuDistance(&hostPoints[(size_t)row * numFeatures],
                        &hostPoints[(size_t)col * numFeatures], numFeatures, metric);
        maxRelativeError = std::max(maxRelativeError, std::fabs(gpuRow[col] - expected) /
                                                          std::max(1.0f, std::fabs(expected)));
      }
    }

    std::printf("%-12s %10.2f %12.2f %10.1e %s\n", name, kernelMs,
                (double)numPoints * numPoints / kernelMs / 1e6, maxRelativeError,
                maxRelativeError < 1e-4f ? "" : "  <-- MISMATCH");
  };

  run("sqeuclidean", metrics::SqEuclidean{});
  run("euclidean", metrics::Euclidean{});
  run("manhattan", metrics::Manhattan{});
  run("chebyshev", metrics::Chebyshev{});
  run("minkowski3", metrics::Minkowski{3.0f});
  run("cosine", metrics::Cosine{});

  CUDA_CHECK(cudaFree(devicePoints));
  CUDA_CHECK(cudaFree(deviceDistances));
  return 0;
}
