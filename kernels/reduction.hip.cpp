#include <hip/hip_runtime.h>

// Block-wide sum reduction, one instantiation per block size. Each
// thread first sums a strided run of 8 elements from global memory
// (register-resident, no LDS cost) before the tree reduction in shared
// memory -- the standard grid-stride-then-reduce pattern. LDS here is
// deliberately small (one float per thread) across all three sizes, so
// these three are a contrast set against the GEMM/stencil/histogram
// kernels: register/bandwidth-bound, not LDS-bound, at any of these
// block sizes.
template <int BLOCK>
__global__ void reduce_sum(const float* X, float* partials, size_t n) {
  __shared__ float sdata[BLOCK];
  size_t i = (size_t)blockIdx.x * BLOCK * 8 + threadIdx.x;
  float local = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    size_t idx = i + (size_t)k * BLOCK;
    if (idx < n) local += X[idx];
  }
  sdata[threadIdx.x] = local;
  __syncthreads();

  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = sdata[0];
}

template __global__ void reduce_sum<128>(const float*, float*, size_t);
template __global__ void reduce_sum<256>(const float*, float*, size_t);
template __global__ void reduce_sum<512>(const float*, float*, size_t);
