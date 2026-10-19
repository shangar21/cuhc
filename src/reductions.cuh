#pragma once

#include <climits>
#include <cstring>
#include <cuda_runtime.h>

// (distance, index) argmin building blocks shared by the clustering kernels.

namespace hc {

struct MinLoc {
  float distance;
  int index;
};

__device__ __forceinline__ MinLoc noMinLoc() { return MinLoc{INFINITY, INT_MAX}; }

// Smaller distance wins; ties go to the lower index, so results are deterministic.
__device__ __forceinline__ MinLoc minLoc(MinLoc x, MinLoc y) {
  return (y.distance < x.distance || (y.distance == x.distance && y.index < x.index)) ? y : x;
}

// Minimum over aligned groups of GroupSize consecutive lanes (GroupSize = 32 is
// the whole warp). Afterwards every lane holds its group's minimum. Each step
// pairs lane i with lane i ^ offset, which never leaves the group.
// All 32 lanes of the warp must call it.
template <int GroupSize = 32> __device__ __forceinline__ MinLoc groupMinLoc(MinLoc value) {
  static_assert(GroupSize >= 1 && GroupSize <= 32 && (GroupSize & (GroupSize - 1)) == 0,
                "group size must be a power of two <= 32");
#pragma unroll
  for (int offset = GroupSize / 2; offset > 0; offset /= 2) {
    MinLoc other;
    other.distance = __shfl_xor_sync(0xffffffffu, value.distance, offset);
    other.index = __shfl_xor_sync(0xffffffffu, value.index, offset);
    value = minLoc(value, other);
  }
  return value;
}

// Minimum over the block; the result is valid in thread 0. All threads must
// call it. Safe to call repeatedly in a loop.
template <int BlockSize> __device__ MinLoc blockMinLoc(MinLoc value) {
  static_assert(BlockSize % 32 == 0 && BlockSize <= 1024, "block size");
  __shared__ MinLoc warpMinimums[BlockSize / 32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;

  value = groupMinLoc(value);
  if (lane == 0)
    warpMinimums[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < BlockSize / 32 ? warpMinimums[lane] : noMinLoc();
    value = groupMinLoc(value);
  }
  __syncthreads(); // warpMinimums may be overwritten by the next call
  return value;
}

// (distance, index) packed into 64 bits so that comparing the packed values as
// unsigned integers compares distance first, then index. That lets
// atomicMin(unsigned long long) keep a global argmin. Non-negative floats
// order the same as their bit patterns, so distances are clamped at 0
// (rounding can produce e.g. -1e-8 for cosine).
__host__ __device__ __forceinline__ unsigned long long packMinLoc(float distance, int index) {
  const float clamped = distance > 0.0f ? distance : 0.0f;
  unsigned int bits;
  memcpy(&bits, &clamped, sizeof(bits));
  return ((unsigned long long)bits << 32) | (unsigned int)index;
}

constexpr unsigned long long kEmptyPackedMinLoc = ~0ull;

__host__ __device__ __forceinline__ float unpackDistance(unsigned long long packed) {
  const unsigned int bits = (unsigned int)(packed >> 32);
  float distance;
  memcpy(&distance, &bits, sizeof(distance));
  return distance;
}

__host__ __device__ __forceinline__ int unpackIndex(unsigned long long packed) {
  return (int)(packed & 0xffffffffull);
}

} // namespace hc
