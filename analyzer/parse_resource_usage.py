"""Compiles each HIP kernel file for gfx942 with ROCm clang and parses
the compiler's own per-kernel resource-usage report out of the generated
assembly. No AMD GPU involved: this only ever runs `hipcc
--offload-arch=gfx942 --cuda-device-only -S`, which emits device
assembly without linking or needing a runtime.
"""
import re
import subprocess
import sys
from pathlib import Path

HIPCC = "/opt/rocm/bin/hipcc"
CPPFILT = "c++filt"

KERNEL_LABEL_RE = re.compile(r"^(_Z\w+):")
FIELD_RES = {
    "vgprs": re.compile(r";\s*NumVgprs:\s*(\d+)"),
    "lds_bytes": re.compile(r";\s*LDSByteSize:\s*(\d+)\s*bytes/workgroup"),
    "scratch_bytes": re.compile(r";\s*ScratchSize:\s*(\d+)"),
    "compiler_occupancy": re.compile(r";\s*Occupancy:\s*(\d+)"),
}


def compile_to_asm(hip_file: Path, out_dir: Path, include_dir: Path) -> Path:
    out_path = out_dir / (hip_file.name + ".s")
    cmd = [
        HIPCC, "--offload-arch=gfx942", "--cuda-device-only",
        "-I", str(include_dir), "-S", str(hip_file), "-o", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"hipcc failed on {hip_file}:\n{result.stdout}\n{result.stderr}")
    return out_path


def demangle(symbol: str) -> str:
    result = subprocess.run([CPPFILT, symbol], capture_output=True, text=True)
    return result.stdout.strip() or symbol


def parse_asm(asm_path: Path) -> list[dict]:
    """Returns one dict per kernel found in the assembly file."""
    lines = asm_path.read_text().splitlines()
    kernels = []
    current = None
    for line in lines:
        m = KERNEL_LABEL_RE.match(line)
        if m:
            if current is not None:
                kernels.append(current)
            current = {"mangled_name": m.group(1), "demangled_name": demangle(m.group(1))}
            continue
        if current is None:
            continue
        for field, rx in FIELD_RES.items():
            fm = rx.search(line)
            if fm and field not in current:
                current[field] = int(fm.group(1))
    if current is not None:
        kernels.append(current)
    # Keep only entries that actually got a resource-usage block (skips
    # any non-kernel labels the matcher accidentally caught).
    return [k for k in kernels if "vgprs" in k and "lds_bytes" in k]


def parse_kernel_file(hip_file: Path, out_dir: Path, include_dir: Path) -> list[dict]:
    asm_path = compile_to_asm(hip_file, out_dir, include_dir)
    return parse_asm(asm_path)


if __name__ == "__main__":
    kernels_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("kernels")
    include_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("include")
    out_dir = Path("/tmp/occ_asm")
    out_dir.mkdir(parents=True, exist_ok=True)
    for hip_file in sorted(kernels_dir.glob("*.hip.cpp")):
        for k in parse_kernel_file(hip_file, out_dir, include_dir):
            print(k)
