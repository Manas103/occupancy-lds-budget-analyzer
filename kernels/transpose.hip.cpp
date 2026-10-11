#include <hip/hip_runtime.h>

// Shared-memory tiled matrix transpose with a +1-column pad (the
// standard trick to avoid LDS bank conflicts when a tile is read back
// column-major after being written row-major). One instantiation per
// tile size; TILE=64 uses a 32x32 thread block (each thread handles a
// 2x2 sub-tile) since 64x64 threads would exceed the hardware's
// per-workgroup thread limit.
template <int TILE>
__global__ void transpose_tiled(const float* X, float* Y, int H, int W) {
  __shared__ float tile[TILE][TILE + 1];
  constexpr int BLK = TILE <= 32 ? TILE : 32;
  constexpr int RT = TILE / BLK;

  int block_row0 = blockIdx.y * TILE;
  int block_col0 = blockIdx.x * TILE;

#pragma unroll
  for (int i = 0; i < RT; ++i) {
#pragma unroll
    for (int j = 0; j < RT; ++j) {
      int row = block_row0 + threadIdx.y + i * BLK;
      int col = block_col0 + threadIdx.x + j * BLK;
      tile[threadIdx.y + i * BLK][threadIdx.x + j * BLK] =
          (row < H && col < W) ? X[(size_t)row * W + col] : 0.f;
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < RT; ++i) {
#pragma unroll
    for (int j = 0; j < RT; ++j) {
      int out_row = block_col0 + threadIdx.y + i * BLK;  // transposed
      int out_col = block_row0 + threadIdx.x + j * BLK;
      if (out_row < W && out_col < H) {
        Y[(size_t)out_row * H + out_col] = tile[threadIdx.x + j * BLK][threadIdx.y + i * BLK];
      }
    }
  }
}

template __global__ void transpose_tiled<32>(const float*, float*, int, int);
template __global__ void transpose_tiled<64>(const float*, float*, int, int);
