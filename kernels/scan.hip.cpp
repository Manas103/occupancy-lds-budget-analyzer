#include <hip/hip_runtime.h>

// Single-block-sized Hillis-Steele inclusive scan (one block per segment
// of BLOCK elements). Uses a double-buffered LDS array (2 * BLOCK
// floats) to avoid in-place read/write hazards across sweep steps -- the
// one scan variant in this suite, included so the kernel mix isn't just
// GEMM/stencil/histogram/transpose/reduction.
template <int BLOCK>
__global__ void scan_block(const float* X, float* Y, size_t n) {
  __shared__ float buf[2][BLOCK];
  size_t gidx = (size_t)blockIdx.x * BLOCK + threadIdx.x;
  int cur = 0;
  buf[0][threadIdx.x] = (gidx < n) ? X[gidx] : 0.f;
  __syncthreads();

  for (int offset = 1; offset < BLOCK; offset <<= 1) {
    int nxt = cur ^ 1;
    float v = buf[cur][threadIdx.x];
    if (threadIdx.x >= offset) v += buf[cur][threadIdx.x - offset];
    buf[nxt][threadIdx.x] = v;
    __syncthreads();
    cur = nxt;
  }

  if (gidx < n) Y[gidx] = buf[cur][threadIdx.x];
}

template __global__ void scan_block<1024>(const float*, float*, size_t);
