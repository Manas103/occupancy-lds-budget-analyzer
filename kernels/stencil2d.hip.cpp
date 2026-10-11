#include <hip/hip_runtime.h>

// 2D 5-point-stencil-style blur (each output = mean of its radius-r
// neighborhood), one kernel instantiation per (output-tile, radius) pair.
// Block is fixed at 16x16 (256 threads, 4 wavefronts); each thread
// computes an RT x RT = (OT/16) x (OT/16) block of outputs and the whole
// workgroup cooperatively stages an (OT+2r) x (OT+2r) halo tile into LDS
// before computing (classic shared-memory halo exchange). Growing OT or r
// grows the halo tile's LDS footprint quadratically -- the same
// occupancy-vs-reuse tradeoff as the GEMM tiles, for a different kernel
// shape (memory-bound, register-light, LDS-heavy instead of
// compute-bound).
template <int OT, int R>
__global__ void stencil_blur(const float* X, float* Y, int H, int W) {
  constexpr int RT = OT / 16;
  constexpr int HT = OT + 2 * R;
  __shared__ float tile[HT][HT];

  int block_row0 = blockIdx.y * OT;
  int block_col0 = blockIdx.x * OT;
  int tid = threadIdx.y * 16 + threadIdx.x;

  for (int idx = tid; idx < HT * HT; idx += 256) {
    int r = idx / HT, c = idx % HT;
    int gr = block_row0 + r - R, gc = block_col0 + c - R;
    gr = gr < 0 ? 0 : (gr >= H ? H - 1 : gr);
    gc = gc < 0 ? 0 : (gc >= W ? W - 1 : gc);
    tile[r][c] = X[(size_t)gr * W + gc];
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < RT; ++i) {
    int row = block_row0 + threadIdx.y * RT + i;
    if (row >= H) continue;
#pragma unroll
    for (int j = 0; j < RT; ++j) {
      int col = block_col0 + threadIdx.x * RT + j;
      if (col >= W) continue;
      int tr = threadIdx.y * RT + i + R, tc = threadIdx.x * RT + j + R;
      float sum = 0.f;
      int count = 0;
#pragma unroll
      for (int dy = -R; dy <= R; ++dy)
#pragma unroll
        for (int dx = -R; dx <= R; ++dx) {
          sum += tile[tr + dy][tc + dx];
          ++count;
        }
      Y[(size_t)row * W + col] = sum / count;
    }
  }
}

template __global__ void stencil_blur<32, 1>(const float*, float*, int, int);
template __global__ void stencil_blur<32, 4>(const float*, float*, int, int);
template __global__ void stencil_blur<64, 2>(const float*, float*, int, int);
template __global__ void stencil_blur<96, 4>(const float*, float*, int, int);

// Retiled fix for stencil_blur<96,4> (LDS = 104^2*4 = 43264 bytes,
// over the 32768-byte budget; see docs/occupancy_report.json): same
// radius, a smaller output tile (48 instead of 96) so the halo tile
// drops to 56^2*4 = 12544 bytes.
template __global__ void stencil_blur<48, 4>(const float*, float*, int, int);
