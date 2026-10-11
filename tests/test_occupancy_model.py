import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "analyzer"))
from occupancy_model import analyze, occupancy_from_lds, occupancy_from_vgprs, waves_per_workgroup  # noqa: E402

ROOT = Path(__file__).parent.parent


def test_waves_per_workgroup_exact_multiple():
    assert waves_per_workgroup(256) == 4


def test_waves_per_workgroup_rounds_up():
    assert waves_per_workgroup(100) == 2  # ceil(100/64)


def test_occupancy_from_vgprs_caps_at_max():
    # 8 VGPRs/wave: 512/8 = 64, capped at the architectural max of 8.
    assert occupancy_from_vgprs(8) == 8


def test_occupancy_from_vgprs_binding():
    # 128 VGPRs/wave: 512/128 = 4, below the cap.
    assert occupancy_from_vgprs(128) == 4


def test_occupancy_from_lds_zero_is_unbounded():
    assert occupancy_from_lds(0, 256) == 8


def test_occupancy_from_lds_hand_verifiable_case():
    # 4 waves/workgroup, 16384 B/workgroup: 65536/16384 = 4 workgroups/CU,
    # 4 workgroups * 4 waves / 4 SIMDs = 4 waves/SIMD.
    assert occupancy_from_lds(16384, 256) == 4


def test_occupancy_from_lds_oversized_workgroup_floors_to_one():
    # A single workgroup's LDS allocation (80000 B) exceeds the whole
    # CU's 65536 B budget: at most one such workgroup can ever be
    # resident, not zero.
    assert occupancy_from_lds(80000, 256) >= 1


def test_analyze_lds_bound_flag():
    # VGPRs allow 8 waves/SIMD; LDS allows only 2 -> LDS-bound.
    result = analyze(vgprs=16, lds_bytes=32768, block_threads=256)
    assert result["occupancy_vgpr_limited"] == 8
    assert result["occupancy_lds_limited"] == 2
    assert result["predicted_wavefronts_per_simd"] == 2
    assert result["lds_bound"] is True


def test_analyze_vgpr_bound_flag():
    # VGPRs allow only 2 waves/SIMD; LDS allows 8 -> not LDS-bound.
    result = analyze(vgprs=256, lds_bytes=0, block_threads=256)
    assert result["occupancy_vgpr_limited"] == 2
    assert result["lds_bound"] is False


def test_occupancy_report_matches_manifest_and_resume_shape():
    """Integration check against the committed report (regenerate with
    `python3 analyzer/run_analysis.py` if the kernels change)."""
    report_path = ROOT / "docs" / "occupancy_report.json"
    if not report_path.exists():
        import pytest
        pytest.skip("docs/occupancy_report.json not generated on this machine (no ROCm clang)")
    report = json.loads(report_path.read_text())
    assert report["summary"]["total_original_kernels"] == 19
    assert report["summary"]["retiled_fix_count"] == 4
    # Every retiled fix must show predicted occupancy improving.
    by_name = {k["name"]: k for k in report["kernels"]}
    for k in report["kernels"]:
        if k["set"] != "retiled_fix":
            continue
        before = by_name[k["fixes"]]["predicted_wavefronts_per_simd"]
        after = k["predicted_wavefronts_per_simd"]
        assert after > before, f"{k['name']} did not improve over {k['fixes']}"
