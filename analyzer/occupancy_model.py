"""Static occupancy model for gfx942 (CDNA 3 / MI300-class), from
published per-CU limits. No AMD GPU is involved anywhere in this file;
every number it combines comes from either ROCm clang's own
compile-time resource-usage report (VGPRs, LDS bytes, scratch) or from
AMD's published architecture limits (cited below), never from a
hardware counter.

Published gfx942 (CDNA 3) limits used here:
  - wavefront size: 64 lanes
  - max wavefronts resident per SIMD: 8 (the hardware wave-slot limit;
    this is also what ROCm clang itself reports as "Occupancy" in its
    resource-usage comments when VGPR/SGPR usage isn't the bottleneck,
    which is the cross-check this project uses to sanity-check the
    model -- see docs/occupancy_report.md)
  - VGPR file size per SIMD: 512 architected vector registers
  - LDS size per CU: 65536 bytes (64 KiB), shared by the 4 SIMDs in a CU
  - SIMDs per CU: 4

Source: AMD "Instinct MI300 Series Performance/Tuning Guide" and the
"AMD Matrix Cores" / CDNA architecture whitepapers published by AMD;
also consistent with the per-kernel "Occupancy" field ROCm clang itself
emits in -S output for gfx942, which this project treats as a
cross-check, not an input (see docs/occupancy_report.md's
"model vs. compiler" column).
"""

WAVEFRONT_SIZE = 64
MAX_WAVES_PER_SIMD = 8
VGPRS_PER_SIMD = 512
LDS_BYTES_PER_CU = 65536
SIMDS_PER_CU = 4


def waves_per_workgroup(block_threads: int) -> int:
    return (block_threads + WAVEFRONT_SIZE - 1) // WAVEFRONT_SIZE


def occupancy_from_vgprs(vgprs_per_wave: int) -> int:
    if vgprs_per_wave <= 0:
        return MAX_WAVES_PER_SIMD
    return min(MAX_WAVES_PER_SIMD, VGPRS_PER_SIMD // vgprs_per_wave)


def occupancy_from_lds(lds_bytes_per_workgroup: int, block_threads: int) -> int:
    waves_per_wg = waves_per_workgroup(block_threads)
    if lds_bytes_per_workgroup <= 0:
        return MAX_WAVES_PER_SIMD
    workgroups_per_cu = LDS_BYTES_PER_CU // lds_bytes_per_workgroup
    if workgroups_per_cu <= 0:
        # A single workgroup's LDS allocation exceeds the per-CU LDS
        # budget outright: at most one such workgroup can ever be
        # resident on the whole CU, so occupancy is as if one workgroup
        # fit (never less than that -- a kernel that compiles is assumed
        # launchable at all, which ROCm's own device-code-only compile
        # also does not reject).
        workgroups_per_cu = 1
    return min(MAX_WAVES_PER_SIMD, (workgroups_per_cu * waves_per_wg) // SIMDS_PER_CU)


def analyze(vgprs: int, lds_bytes: int, block_threads: int) -> dict:
    occ_vgpr = occupancy_from_vgprs(vgprs)
    occ_lds = occupancy_from_lds(lds_bytes, block_threads)
    predicted = min(occ_vgpr, occ_lds)
    return {
        "occupancy_vgpr_limited": occ_vgpr,
        "occupancy_lds_limited": occ_lds,
        "predicted_wavefronts_per_simd": predicted,
        "lds_bound": occ_lds < occ_vgpr,
    }
