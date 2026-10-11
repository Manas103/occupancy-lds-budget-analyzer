"""Compiles every kernel in kernels/*.hip.cpp for gfx942 with ROCm clang,
parses VGPR/LDS/scratch usage out of the compiler's own resource-usage
report, computes predicted wavefronts resident per SIMD against
gfx942's published per-CU limits, and writes docs/occupancy_report.json
and docs/occupancy_report.md.

Static, from compiler reports: no AMD GPU involved. The only thing this
script knows that the compiler report doesn't is each kernel's launch
workgroup size (analyzer/kernel_manifest.json), since that's a launch
choice, not a device-code-only compile fact.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from occupancy_model import analyze  # noqa: E402
from parse_resource_usage import parse_kernel_file  # noqa: E402

ROOT = Path(__file__).parent.parent
KERNELS_DIR = ROOT / "kernels"
INCLUDE_DIR = ROOT / "include"
DOCS_DIR = ROOT / "docs"
MANIFEST_PATH = Path(__file__).parent / "kernel_manifest.json"

NAME_RE = re.compile(r"\bvoid\s+([\w:]+<[^()]*>)\(")


def extract_short_name(demangled: str) -> str:
    m = NAME_RE.search(demangled)
    if not m:
        raise ValueError(f"could not extract kernel name from: {demangled}")
    return m.group(1)


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)["kernels"]

    asm_out = Path("/tmp/occ_asm")
    asm_out.mkdir(parents=True, exist_ok=True)

    results = []
    seen = set()
    for hip_file in sorted(KERNELS_DIR.glob("*.hip.cpp")):
        for k in parse_kernel_file(hip_file, asm_out, INCLUDE_DIR):
            short_name = extract_short_name(k["demangled_name"])
            if short_name not in manifest:
                raise KeyError(f"{short_name!r} (from {hip_file.name}) not in kernel_manifest.json")
            if short_name in seen:
                raise ValueError(f"duplicate kernel name {short_name!r}")
            seen.add(short_name)

            meta = manifest[short_name]
            occ = analyze(k["vgprs"], k["lds_bytes"], meta["block_threads"])
            results.append({
                "name": short_name,
                "source_file": hip_file.name,
                "set": meta["set"],
                "fixes": meta.get("fixes"),
                "block_threads": meta["block_threads"],
                "vgprs": k["vgprs"],
                "lds_bytes": k["lds_bytes"],
                "scratch_bytes": k.get("scratch_bytes", 0),
                "compiler_reported_occupancy": k.get("compiler_occupancy"),
                **occ,
            })

    missing = set(manifest) - seen
    if missing:
        raise RuntimeError(f"manifest entries never matched to a compiled kernel: {missing}")

    original = [r for r in results if r["set"] == "original"]
    retiled = [r for r in results if r["set"] == "retiled_fix"]
    lds_bound = [r for r in original if r["lds_bound"]]

    summary = {
        "total_original_kernels": len(original),
        "lds_bound_count": len(lds_bound),
        "lds_bound_kernels": [r["name"] for r in lds_bound],
        "retiled_fix_count": len(retiled),
    }

    DOCS_DIR.mkdir(exist_ok=True)
    with open(DOCS_DIR / "occupancy_report.json", "w") as f:
        json.dump({"summary": summary, "kernels": results}, f, indent=2)

    write_markdown_report(results, summary)
    print(json.dumps(summary, indent=2))


def write_markdown_report(results, summary):
    lines = ["# Occupancy report (static, gfx942)", ""]
    lines.append(f"**{summary['lds_bound_count']} of {summary['total_original_kernels']} original kernels are LDS-bound** "
                  f"(predicted occupancy limited by LDS, not VGPRs): {', '.join(summary['lds_bound_kernels'])}")
    lines.append("")
    lines.append("| Kernel | set | VGPRs | LDS (B) | occ (VGPR) | occ (LDS) | predicted | LDS-bound | compiler Occupancy |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (x["set"], x["name"])):
        lines.append(
            f"| `{r['name']}` | {r['set']} | {r['vgprs']} | {r['lds_bytes']} | "
            f"{r['occupancy_vgpr_limited']} | {r['occupancy_lds_limited']} | "
            f"{r['predicted_wavefronts_per_simd']} | {r['lds_bound']} | "
            f"{r['compiler_reported_occupancy']} |"
        )
    lines.append("")
    lines.append("## Retiled fixes")
    lines.append("")
    lines.append("| Fix | fixes | before -> after predicted occupancy |")
    lines.append("|---|---|---|")
    by_name = {r["name"]: r for r in results}
    for r in results:
        if r["set"] != "retiled_fix":
            continue
        before = by_name[r["fixes"]]["predicted_wavefronts_per_simd"]
        after = r["predicted_wavefronts_per_simd"]
        lines.append(f"| `{r['name']}` | `{r['fixes']}` | {before} -> {after} |")
    (DOCS_DIR / "occupancy_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
