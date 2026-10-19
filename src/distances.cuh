#pragma once

#include <cuda_runtime.h>
#include <math.h>

#define DIST_HD __host__ __device__ __forceinline__

// A distance metric is a small struct with:
//   using Acc = ...;                                  accumulator type
//   Acc   init() const;                               starting value
//   Acc   accumulate(Acc acc, float a, float b) const; fold in one feature
//   float finalize(Acc acc) const;                    accumulator -> distance
//
// d(x, y) = finalize(accumulate(...accumulate(init(), x0, y0)..., xd, yd))
//
// Kernels are templated on the metric, so every call is inlined: a user-written
// metric in this form compiles to the same code as a built-in one.

namespace metrics {

struct SqEuclidean {
  using Acc = float;
  DIST_HD Acc init() const { return 0.0f; }
  DIST_HD Acc accumulate(Acc acc, float a, float b) const {
    const float diff = a - b;
    return fmaf(diff, diff, acc);
  }
  DIST_HD float finalize(Acc acc) const { return acc; }
};

struct Euclidean : SqEuclidean {
  DIST_HD float finalize(Acc acc) const { return sqrtf(acc); }
};

struct Manhattan {
  using Acc = float;
  DIST_HD Acc init() const { return 0.0f; }
  DIST_HD Acc accumulate(Acc acc, float a, float b) const { return acc + fabsf(a - b); }
  DIST_HD float finalize(Acc acc) const { return acc; }
};

struct Chebyshev {
  using Acc = float;
  DIST_HD Acc init() const { return 0.0f; }
  DIST_HD Acc accumulate(Acc acc, float a, float b) const { return fmaxf(acc, fabsf(a - b)); }
  DIST_HD float finalize(Acc acc) const { return acc; }
};

// Metrics can carry runtime parameters as members.
struct Minkowski {
  float p;
  using Acc = float;
  DIST_HD Acc init() const { return 0.0f; }
  DIST_HD Acc accumulate(Acc acc, float a, float b) const {
    const float x = fabsf(a - b);
#ifdef __CUDA_ARCH__
    // |x|^p = 2^(p*log2|x|) on the hardware special-function units. Full powf
    // is a long branchy sequence; inlined into the tiled kernel's unrolled loop
    // it thrashes the instruction cache and runs ~30x slower.
    return x > 0.0f ? acc + exp2f(p * __log2f(x)) : acc;
#else
    return acc + powf(x, p);
#endif
  }
  DIST_HD float finalize(Acc acc) const { return powf(acc, 1.0f / p); }
};

// Accumulators don't have to be a single float: cosine tracks (a.b, |a|^2, |b|^2).
struct Cosine {
  using Acc = float3;
  DIST_HD Acc init() const { return make_float3(0.0f, 0.0f, 0.0f); }
  DIST_HD Acc accumulate(Acc acc, float a, float b) const {
    acc.x = fmaf(a, b, acc.x);
    acc.y = fmaf(a, a, acc.y);
    acc.z = fmaf(b, b, acc.z);
    return acc;
  }
  DIST_HD float finalize(Acc acc) const {
    const float denom = sqrtf(acc.y * acc.z);
    // A zero vector has similarity 0 with everything (distance 1), as in sklearn.
    return denom > 0.0f ? 1.0f - acc.x / denom : 1.0f;
  }
};

// Traits that let kernels skip work. A user metric gets the safe default
// (false) unless it opts in with a specialisation.
//
// IsSymmetric: d(x, y) == d(y, x) bit for bit, so each unordered pair needs
// computing only once. True for the built-ins: each sums or maxes a function
// of |x_k - y_k|, or (cosine) commutative products.
template <typename Metric> struct IsSymmetric {
  static constexpr bool value = false;
};
template <> struct IsSymmetric<SqEuclidean> { static constexpr bool value = true; };
template <> struct IsSymmetric<Euclidean> { static constexpr bool value = true; };
template <> struct IsSymmetric<Manhattan> { static constexpr bool value = true; };
template <> struct IsSymmetric<Chebyshev> { static constexpr bool value = true; };
template <> struct IsSymmetric<Minkowski> { static constexpr bool value = true; };
template <> struct IsSymmetric<Cosine> { static constexpr bool value = true; };

// HasBoxBound: the distance depends on the per-feature gaps |x_k - y_k| only,
// and never decreases when one gap grows. Then the smallest gaps between two
// bounding boxes give a lower bound for every pair of points in them:
//     finalize(accumulate(... accumulate(init(), gap_k, 0) ...))
// (spatial.cuh). Cosine depends on the vectors' directions, not their gaps.
template <typename Metric> struct HasBoxBound {
  static constexpr bool value = false;
};
template <> struct HasBoxBound<SqEuclidean> { static constexpr bool value = true; };
template <> struct HasBoxBound<Euclidean> { static constexpr bool value = true; };
template <> struct HasBoxBound<Manhattan> { static constexpr bool value = true; };
template <> struct HasBoxBound<Chebyshev> { static constexpr bool value = true; };
template <> struct HasBoxBound<Minkowski> { static constexpr bool value = true; };

} // namespace metrics
