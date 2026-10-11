#include <hip/hip_runtime.h>

// Privatized per-block histogram: each workgroup accumulates into its own
// LDS-resident bin array via atomics (no cross-block contention), then
// folds its private bins into the global histogram once. NBINS directly
// sets the LDS footprint (NBINS*4 bytes) -- this is the natural way a
// histogram kernel's LDS cost grows with the data's value range, as
// distinct a mechanism from the GEMM/stencil tile-size tradeoff as this
// suite has.
template <int NBINS>
__global__ void histogram_privatized(const int* data, size_t n, unsigned int* global_bins) {
  __shared__ unsigned int bins[NBINS];
  for (int i = threadIdx.x; i < NBINS; i += blockDim.x) bins[i] = 0;
  __syncthreads();

  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = idx; i < n; i += stride) {
    int bin = data[i] % NBINS;
    atomicAdd(&bins[bin], 1u);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < NBINS; i += blockDim.x) {
    if (bins[i] != 0) atomicAdd(&global_bins[i], bins[i]);
  }
}

template __global__ void histogram_privatized<1024>(const int*, size_t, unsigned int*);
template __global__ void histogram_privatized<4096>(const int*, size_t, unsigned int*);
template __global__ void histogram_privatized<16384>(const int*, size_t, unsigned int*);

// Retiled fix for histogram_privatized<16384> (LDS = 16384*4 = 65536
// bytes, the full per-workgroup hard limit, occupancy 1; see
// docs/occupancy_report.json): same value range, split across more
// blocks with fewer bins privatized per block.
template __global__ void histogram_privatized<2048>(const int*, size_t, unsigned int*);
