"""A deliberately slow, obviously-correct reference for occupancy_from_lds:
simulate packing workgroups onto one CU one at a time until the LDS
budget runs out, rather than the formula's single integer division.
Diffed exactly against the formula over a sweep of LDS sizes and
workgroup shapes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "analyzer"))
from occupancy_model import (  # noqa: E402
    LDS_BYTES_PER_CU,
    MAX_WAVES_PER_SIMD,
    SIMDS_PER_CU,
    occupancy_from_lds,
    waves_per_workgroup,
)


def reference_occupancy_from_lds(lds_bytes_per_workgroup: int, block_threads: int) -> int:
    if lds_bytes_per_workgroup <= 0:
        return MAX_WAVES_PER_SIMD
    waves_per_wg = waves_per_workgroup(block_threads)
    used = 0
    resident_workgroups = 0
    while used + lds_bytes_per_workgroup <= LDS_BYTES_PER_CU:
        used += lds_bytes_per_workgroup
        resident_workgroups += 1
    if resident_workgroups == 0:
        resident_workgroups = 1  # one still fits "at a time", see occupancy_model.py
    total_waves = resident_workgroups * waves_per_wg
    return min(MAX_WAVES_PER_SIMD, total_waves // SIMDS_PER_CU)


def test_reference_oracle_matches_formula_over_a_sweep():
    lds_sizes = [0, 512, 1024, 2048, 4096, 8192, 16384, 18432, 32768, 43264, 51200, 65536, 80000]
    block_sizes = [64, 128, 256, 512, 1024]
    for lds in lds_sizes:
        for block in block_sizes:
            formula = occupancy_from_lds(lds, block)
            oracle = reference_occupancy_from_lds(lds, block)
            assert formula == oracle, f"mismatch at lds={lds}, block={block}: formula={formula} oracle={oracle}"
