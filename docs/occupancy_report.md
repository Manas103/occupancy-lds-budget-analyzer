# Occupancy report (static, gfx942)

**7 of 19 original kernels are LDS-bound** (predicted occupancy limited by LDS, not VGPRs): gemm_tile_regtile<48>, gemm_tile_regtile<64>, gemm_tile_regtile<80>, histogram_privatized<4096>, histogram_privatized<16384>, stencil_blur<64, 2>, stencil_blur<96, 4>

| Kernel | set | VGPRs | LDS (B) | occ (VGPR) | occ (LDS) | predicted | LDS-bound | compiler Occupancy |
|---|---|---|---|---|---|---|---|---|
| `gemm_tile<16>` | original | 36 | 2048 | 8 | 8 | 8 | False | 8 |
| `gemm_tile<24>` | original | 40 | 4608 | 8 | 8 | 8 | False | 8 |
| `gemm_tile<32>` | original | 40 | 8192 | 8 | 8 | 8 | False | 8 |
| `gemm_tile_regtile<48>` | original | 96 | 18432 | 5 | 3 | 3 | True | 5 |
| `gemm_tile_regtile<64>` | original | 96 | 32768 | 5 | 2 | 2 | True | 5 |
| `gemm_tile_regtile<80>` | original | 128 | 51200 | 4 | 1 | 1 | True | 4 |
| `histogram_privatized<1024>` | original | 8 | 4096 | 8 | 8 | 8 | False | 8 |
| `histogram_privatized<16384>` | original | 8 | 65536 | 8 | 1 | 1 | True | 4 |
| `histogram_privatized<4096>` | original | 8 | 16384 | 8 | 4 | 4 | True | 8 |
| `reduce_sum<128>` | original | 8 | 512 | 8 | 8 | 8 | False | 8 |
| `reduce_sum<256>` | original | 10 | 1024 | 8 | 8 | 8 | False | 8 |
| `reduce_sum<512>` | original | 10 | 2048 | 8 | 8 | 8 | False | 8 |
| `scan_block<1024>` | original | 6 | 8192 | 8 | 8 | 8 | False | 8 |
| `stencil_blur<32, 1>` | original | 16 | 4624 | 8 | 8 | 8 | False | 8 |
| `stencil_blur<32, 4>` | original | 26 | 6400 | 8 | 8 | 8 | False | 8 |
| `stencil_blur<64, 2>` | original | 40 | 18496 | 8 | 3 | 3 | True | 8 |
| `stencil_blur<96, 4>` | original | 40 | 43264 | 8 | 1 | 1 | True | 4 |
| `transpose_tiled<32>` | original | 8 | 4224 | 8 | 8 | 8 | False | 8 |
| `transpose_tiled<64>` | original | 11 | 16640 | 8 | 8 | 8 | False | 8 |
| `gemm_tile_regtile_kd<64, 16>` | retiled_fix | 80 | 8192 | 6 | 8 | 6 | False | 6 |
| `gemm_tile_regtile_kd<80, 16>` | retiled_fix | 128 | 10240 | 4 | 6 | 4 | False | 4 |
| `histogram_privatized<2048>` | retiled_fix | 8 | 8192 | 8 | 8 | 8 | False | 8 |
| `stencil_blur<48, 4>` | retiled_fix | 23 | 12544 | 8 | 5 | 5 | True | 8 |

## Retiled fixes

| Fix | fixes | before -> after predicted occupancy |
|---|---|---|
| `gemm_tile_regtile_kd<64, 16>` | `gemm_tile_regtile<64>` | 2 -> 6 |
| `gemm_tile_regtile_kd<80, 16>` | `gemm_tile_regtile<80>` | 1 -> 4 |
| `histogram_privatized<2048>` | `histogram_privatized<16384>` | 1 -> 8 |
| `stencil_blur<48, 4>` | `stencil_blur<96, 4>` | 1 -> 5 |
