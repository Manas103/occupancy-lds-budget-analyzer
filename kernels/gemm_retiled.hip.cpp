#include <hip/hip_runtime.h>

// Retiled fix for gemm_tile_regtile<64> and <80> (kernels/gemm_tiles.hip.cpp),
// both over the 32768-byte LDS budget (see docs/occupancy_report.json).
// The output tile (TILE_MN, which sets how much global-memory reuse each
// workgroup gets) and the K-dimension staging depth (TILE_K, which sets
// how much of the K loop is buffered in LDS per iteration) don't have to
// be the same number -- decoupling them keeps the same output-tile reuse
// while shrinking the LDS footprint to 2 * TILE_MN * TILE_K * 4 bytes
// instead of 2 * TILE_MN^2 * 4. This is the standard fix real GEMM
// kernels use once a square tile's LDS cost outgrows its occupancy
// budget: stage a thinner slab of K per iteration, loop more times.
template <int TILE_MN, int TILE_K>
__global__ void gemm_tile_regtile_kd(const float* X, const float* W, float* Y, int T, int K, int N) {
  constexpr int RT = TILE_MN / 16;
  __shared__ float Xs[TILE_MN][TILE_K];
  __shared__ float Ws[TILE_K][TILE_MN];

  int block_row0 = blockIdx.y * TILE_MN;
  int block_col0 = blockIdx.x * TILE_MN;
  int tid = threadIdx.y * 16 + threadIdx.x;

  float acc[RT][RT];
#pragma unroll
  for (int i = 0; i < RT; ++i)
#pragma unroll
    for (int j = 0; j < RT; ++j) acc[i][j] = 0.f;

  int n_tiles = (K + TILE_K - 1) / TILE_K;
  for (int t = 0; t < n_tiles; ++t) {
    int k0 = t * TILE_K;
    for (int idx = tid; idx < TILE_MN * TILE_K; idx += 256) {
      int r = idx / TILE_K, c = idx % TILE_K;
      int grow = block_row0 + r, kx = k0 + c;
      Xs[r][c] = (grow < T && kx < K) ? X[(size_t)grow * K + kx] : 0.f;
    }
    for (int idx = tid; idx < TILE_K * TILE_MN; idx += 256) {
      int r = idx / TILE_MN, c = idx % TILE_MN;
      int ky = k0 + r, gcol = block_col0 + c;
      Ws[r][c] = (gcol < N && ky < K) ? W[(size_t)ky * N + gcol] : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
#pragma unroll
      for (int i = 0; i < RT; ++i) {
        float xv = Xs[threadIdx.y * RT + i][kk];
#pragma unroll
        for (int j = 0; j < RT; ++j) acc[i][j] += xv * Ws[kk][threadIdx.x * RT + j];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < RT; ++i) {
    int row = block_row0 + threadIdx.y * RT + i;
    if (row >= T) continue;
#pragma unroll
    for (int j = 0; j < RT; ++j) {
      int col = block_col0 + threadIdx.x * RT + j;
      if (col < N) Y[(size_t)row * N + col] = acc[i][j];
    }
  }
}

// gemm_tile_regtile<64> (LDS 32768) retiled: same 64x64 output tile,
// K staged 16 at a time -> LDS = 2*64*16*4 = 8192 bytes.
template __global__ void gemm_tile_regtile_kd<64, 16>(const float*, const float*, float*, int, int, int);
// gemm_tile_regtile<80> (LDS 51200) retiled: same 80x80 output tile,
// K staged 16 at a time -> LDS = 2*80*16*4 = 10240 bytes.
template __global__ void gemm_tile_regtile_kd<80, 16>(const float*, const float*, float*, int, int, int);
