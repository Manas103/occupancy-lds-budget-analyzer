#include <hip/hip_runtime.h>

// Shared-memory-tiled GEMM, one kernel instantiation per TILE size. This
// is the tile-size sweep a real kernel author runs when tuning occupancy:
// a bigger tile amortizes more global-memory traffic per output element,
// but LDS is a fixed, per-CU budget shared by every resident workgroup,
// so a bigger tile can buy bandwidth at the direct cost of occupancy.
//
// Small tiles (16, 24, 32): block dim == TILE x TILE, one thread per
// output element, the simplest tiled-GEMM form.
template <int TILE>
__global__ void gemm_tile(const float* X, const float* W, float* Y, int T, int K, int N) {
  __shared__ float Xs[TILE][TILE];
  __shared__ float Ws[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.f;
  int n_tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < n_tiles; ++t) {
    int k = t * TILE;
    Xs[threadIdx.y][threadIdx.x] = (row < T && k + threadIdx.x < K) ? X[(size_t)row * K + k + threadIdx.x] : 0.f;
    Ws[threadIdx.y][threadIdx.x] = (col < N && k + threadIdx.y < K) ? W[(size_t)(k + threadIdx.y) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) acc += Xs[threadIdx.y][kk] * Ws[kk][threadIdx.x];
    __syncthreads();
  }
  if (row < T && col < N) Y[(size_t)row * N + col] = acc;
}

template __global__ void gemm_tile<16>(const float*, const float*, float*, int, int, int);
template __global__ void gemm_tile<24>(const float*, const float*, float*, int, int, int);
template __global__ void gemm_tile<32>(const float*, const float*, float*, int, int, int);

// Register-tiled GEMM, one instantiation per (larger) TILE size. Block
// dim is fixed at 16x16 (256 threads, 4 wavefronts); each thread computes
// an RT x RT = (TILE/16) x (TILE/16) block of outputs instead of one,
// and cooperatively loads the TILE x TILE staging tiles via a strided
// loop (256 threads loading TILE*TILE elements). This is the standard way
// a real GEMM kernel grows its LDS tile past "one thread per output"
// without exceeding the hardware's per-workgroup thread-count limit.
template <int TILE>
__global__ void gemm_tile_regtile(const float* X, const float* W, float* Y, int T, int K, int N) {
  constexpr int RT = TILE / 16;
  __shared__ float Xs[TILE][TILE];
  __shared__ float Ws[TILE][TILE];

  int block_row0 = blockIdx.y * TILE;
  int block_col0 = blockIdx.x * TILE;
  int tid = threadIdx.y * 16 + threadIdx.x;  // 0..255

  float acc[RT][RT];
#pragma unroll
  for (int i = 0; i < RT; ++i)
#pragma unroll
    for (int j = 0; j < RT; ++j) acc[i][j] = 0.f;

  int n_tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < n_tiles; ++t) {
    int k0 = t * TILE;
    for (int idx = tid; idx < TILE * TILE; idx += 256) {
      int r = idx / TILE, c = idx % TILE;
      int grow = block_row0 + r, kx = k0 + c;
      Xs[r][c] = (grow < T && kx < K) ? X[(size_t)grow * K + kx] : 0.f;
      int ky = k0 + r, gcol = block_col0 + c;
      Ws[r][c] = (gcol < N && ky < K) ? W[(size_t)ky * N + gcol] : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
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

template __global__ void gemm_tile_regtile<48>(const float*, const float*, float*, int, int, int);
template __global__ void gemm_tile_regtile<64>(const float*, const float*, float*, int, int, int);
template __global__ void gemm_tile_regtile<80>(const float*, const float*, float*, int, int, int);
