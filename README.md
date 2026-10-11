# Occupancy and LDS Budget Analyzer for HIP Kernels

Compiles 19 HIP kernels for gfx942 (CDNA 3, MI300-class) with ROCm clang,
parses VGPR, scratch and LDS-per-workgroup usage straight out of the
compiler's own resource-usage report, and computes predicted wavefronts
resident per SIMD against gfx942's published per-CU limits. Every number
below is either something ROCm clang itself printed for a real
`--offload-arch=gfx942` compile on this machine, or a published
architecture limit cited by name -- nothing here comes from a hardware
counter, because no AMD GPU is involved anywhere in this project.

**Honest framing, up front:**
- **Static, from compiler reports, no AMD GPU involved.** Every VGPR, LDS
  and scratch number is what ROCm clang's `-S` output says for a
  `--cuda-device-only` compile; occupancy is a hand-derived model against
  published gfx942 limits (`analyzer/occupancy_model.py`), not a
  profiler measurement.
- The one input this analysis takes from outside the compiler's own
  output is each kernel's launch workgroup size
  (`analyzer/kernel_manifest.json`) -- a launch-time choice, not
  something a device-code-only compile emits.
- ROCm clang's own `-S` output includes an `Occupancy` field per kernel.
  This project's model does **not** reuse that number as its answer (the
  point is to derive occupancy independently from VGPRs/LDS against
  published limits); it's recorded alongside the model's own prediction
  as a cross-check, and the two disagree on several kernels -- see
  Findings, not hidden.

## Why this exists

A resume line claiming GPU occupancy knowledge is worthless if the
candidate has never had an AMD GPU to measure it on. This project asks a
narrower, fully honest question instead: given only what ROCm's compiler
reports at compile time (register and shared-memory usage per kernel)
and AMD's own published per-CU hardware limits, how far can occupancy be
predicted without ever touching hardware? The answer turns out to be
"close enough to flag real problems" -- see Measured results.

## Architecture

```
kernels/gemm_tiles.hip.cpp     6 GEMM kernels: tile sizes 16,24,32 (one thread
                                per output) and 48,64,80 (register-tiled,
                                256-thread block)
kernels/gemm_retiled.hip.cpp   2 of the above, retiled: same output tile,
                                a smaller K-staging depth (see Findings)
kernels/stencil2d.hip.cpp      5 2D blur-stencil kernels: (tile, radius) pairs
                                32x1, 32x4, 64x2, 96x4, and 96x4's retiled fix (48x4)
kernels/reduction.hip.cpp      3 block-wide sum-reduction kernels (block 128/256/512)
kernels/histogram.hip.cpp      4 privatized-histogram kernels: 1024/4096/16384 bins,
                                plus 16384's retiled fix (2048 bins)
kernels/transpose.hip.cpp      2 padded shared-memory transpose kernels (tile 32/64)
kernels/scan.hip.cpp           1 Hillis-Steele block scan kernel (block 1024)

analyzer/occupancy_model.py       the gfx942 limits and the occupancy formula
analyzer/parse_resource_usage.py  compiles a kernel file for gfx942, parses its
                                   assembly's resource-usage comments
analyzer/kernel_manifest.json     each kernel's launch workgroup size
analyzer/run_analysis.py          orchestrates all of the above, writes docs/occupancy_report.{json,md}

tests/test_occupancy_model.py     unit tests + an integration check against the
                                   committed report
tests/test_reference_oracle.py    a deliberately slow, obviously-correct
                                   brute-force reference for the LDS-occupancy
                                   formula, diffed exactly over a sweep
```

### The occupancy model

gfx942 (CDNA 3) published limits used throughout
(`analyzer/occupancy_model.py`, cited to AMD's MI300 tuning guide and
CDNA architecture materials):

- wavefront size 64, max 8 wavefronts resident per SIMD
- 512 architected VGPRs per SIMD
- 65536 bytes (64 KiB) of LDS per CU, shared by that CU's 4 SIMDs

```
occ_vgpr = min(8, 512 // vgprs_per_wave)
occ_lds  = min(8, (floor(65536 / lds_bytes_per_workgroup) * waves_per_workgroup) // 4)
predicted_occupancy = min(occ_vgpr, occ_lds)
lds_bound = occ_lds < occ_vgpr
```

This is a simplification on purpose, disclosed rather than hidden: it
ignores SGPR usage, LDS allocation-granularity rounding, and scratch
(register-spill) overhead, all of which the real hardware's scheduler
also takes into account. See Findings for where that simplification
visibly disagrees with ROCm clang's own occupancy estimate.

### Register-tiled kernels: why the output tile and the thread block differ

`gemm_tile_regtile<TILE>`, `stencil_blur<OT, R>` and the retiled fixes all
launch a fixed 256-thread (16x16) block regardless of their output-tile
size, with each thread computing an `(TILE/16)^2` block of outputs. This
is the standard way a real kernel grows its LDS tile (and its
global-memory reuse per load) without exceeding HIP's 1024-thread
per-workgroup limit -- `TILE=96` would need 9216 threads at
one-thread-per-output, which isn't launchable at all.

## Validation

- **19 kernels compile clean for gfx942** via ROCm clang, device-code-only
  (no link, no AMD GPU needed): `docs/analysis_run_output.txt`.
- **A reference oracle for the LDS-occupancy formula**
  (`tests/test_reference_oracle.py`): a deliberately slow loop that packs
  workgroups onto a simulated CU one at a time until LDS runs out, diffed
  exactly against the single-division formula over 13 LDS sizes x 5 block
  sizes (65 cases). Needed because `occ_lds`'s one-line division has to
  agree with "how many workgroups actually fit" at every boundary
  (including the edge case where one workgroup's LDS already exceeds the
  whole CU's budget), and that is exactly the kind of off-by-one a single
  formula can get wrong silently.
- **11 pytest checks** (hand-verifiable cases, the VGPR/LDS-bound branch
  split, and an integration check against the committed report):
  `docs/test_output.txt`.

## Findings

**The model's "LDS-bound" count landed on exactly what was predicted
going in, and that is worth being suspicious of, not just reporting.**
Before building any kernel, the plan was "grow a few GEMM, stencil and
histogram variants' LDS tile past 32 KB and see roughly a third of 19
kernels come out LDS-bound." The measured result was 7 of 19 -- a clean
match. The honest read is not "the prediction was right," it's "the
kernel suite was designed by someone who already knew where the LDS
cliff was," which is a weaker claim. The oracle test and the
compiler-occupancy cross-check below exist specifically because a model
that happens to match a hand-picked kernel suite proves less than a model
that also survives being checked against something it wasn't tuned
against.

**ROCm clang's own reported `Occupancy` disagrees with this model on 3 of
the 7 LDS-bound kernels, and the disagreement is informative, not just
noise.** `docs/occupancy_report.md`'s last column is ROCm clang's own
per-kernel occupancy estimate, recorded but never used as this model's
input. For most kernels the two agree exactly. Three don't:

| Kernel | this model | ROCm clang's own estimate |
|---|---|---|
| `histogram_privatized<4096>` | 4 | 8 |
| `histogram_privatized<16384>` | 1 | 4 |
| `stencil_blur<96, 4>` | 1 | 4 |

All three are histogram/stencil kernels with very low VGPR counts (8-40)
relative to their LDS footprint, which is exactly where this model's
disclosed simplifications (SGPR usage, LDS allocation-granularity
rounding) matter most. The direction of every disagreement is the same
(this model is more pessimistic), consistent with the model rounding the
*whole* 64 KiB CU LDS budget down by simple integer division while ROCm's
own estimate likely accounts for allocation granularity that lets more
workgroups' LDS pack in than a clean division predicts. This is reported
as an open discrepancy, not resolved: closing it would need reading
ROCm's actual occupancy-estimation source, which is out of scope here.
It is also the single best argument in this project for measuring real
occupancy with `rocprof` on real hardware rather than trusting a static
model past the point where it agrees with the vendor's own compiler.

**One retiled kernel spills to scratch memory, a cost this model doesn't
count at all.** `gemm_tile_regtile<80>` (and its retiled fix) report
nonzero `ScratchSize` (112 and 16 bytes respectively,
`docs/occupancy_report.json`): the 5x5 register-tiled accumulator array
plus staging variables exceeds what fits in VGPRs alone, so the compiler
spills some to scratch (device memory, far slower than a register).
Scratch usage is itself a real-hardware occupancy cost (it adds
latency-hiding pressure the compiler schedules around) that this static
model doesn't model at all -- disclosed here, not absorbed into the
occupancy number.

## Measured results

Compiler: ROCm clang 18.0.0 (part of ROCm 6.2.2), `ubuntu 22.04`,
`--offload-arch=gfx942 --cuda-device-only -S`, no AMD GPU anywhere in
this pipeline. Full per-kernel table: `docs/occupancy_report.md`; raw
numbers: `docs/occupancy_report.json`.

**Claim: "LDS bound 7 of 19 kernels."** Measured **7 of 19**:
`gemm_tile_regtile<48>`, `gemm_tile_regtile<64>`, `gemm_tile_regtile<80>`,
`histogram_privatized<4096>`, `histogram_privatized<16384>`,
`stencil_blur<64, 2>`, `stencil_blur<96, 4>`. **Met.**

**Claim: "re-tiling 4 under 32KB raised predicted occupancy from 2 to 6
wavefronts per SIMD."** Four kernels with LDS at or over the 32768-byte
budget were retiled (same output-tile reuse, smaller LDS footprint; see
Architecture and `kernels/gemm_retiled.hip.cpp`). Measured, before ->
after predicted occupancy:

| kernel retiled | before | after |
|---|---|---|
| `gemm_tile_regtile<64>` -> `gemm_tile_regtile_kd<64, 16>` | 2 | **6** |
| `gemm_tile_regtile<80>` -> `gemm_tile_regtile_kd<80, 16>` | 1 | 4 |
| `stencil_blur<96, 4>` -> `stencil_blur<48, 4>` | 1 | 5 |
| `histogram_privatized<16384>` -> `histogram_privatized<2048>` | 1 | 8 |

**Met exactly for the kernel the claim describes** (the 64-wide GEMM
tile, 2 -> 6); the other three retiled kernels also improved, by
different amounts (4, 5, and 8 respectively) depending on how much LDS
each specific fix actually removed. Reported individually rather than
averaged into one number, since a single "2 to 6" would misrepresent the
other three.

**Claim: "static, from compiler reports, no AMD GPU involved."** **Met**:
every number above is ROCm clang's own `-S` output or a published gfx942
limit; `nvidia-smi`/CUDA do not appear anywhere in this project, and
there is no AMD GPU in this pipeline to run one against.

## Building and running

```bash
# WSL2 Ubuntu 22.04, ROCm 6.2.2 (rocm-llvm, rocm-device-libs, hip-dev)
python3 analyzer/run_analysis.py
# writes docs/occupancy_report.json and docs/occupancy_report.md

python3 -m pytest tests -q
```

`analyzer/run_analysis.py` calls `hipcc --offload-arch=gfx942
--cuda-device-only -S` once per kernel file (no CUDA, no AMD GPU, no
link step) and `c++filt` to demangle kernel symbol names; both ship with
a ROCm 6.2.2 + binutils install. The pytest suite (`tests/`) has no ROCm
dependency at all and runs anywhere Python 3.9+ is available; the one
integration test that reads `docs/occupancy_report.json` skips cleanly
if that file hasn't been generated on the current machine.

## Layout

See Architecture above; `docs/occupancy_report.md` and
`docs/occupancy_report.json` are the generated report this README's
Measured results section quotes, `docs/analysis_run_output.txt` and
`docs/test_output.txt` are the raw run logs behind every number here.

## Known limitations

- This model ignores SGPR usage, LDS allocation-granularity rounding,
  and scratch (register-spill) overhead -- see Findings for where that
  visibly disagrees with ROCm clang's own occupancy estimate and where a
  kernel spills to scratch at all.
- No kernel in this suite is ever executed, on any GPU, AMD or otherwise.
  There are no throughput, latency, or correctness-against-a-reference
  numbers for any kernel here; "correct" in this project means "compiles
  for gfx942," nothing more. A sibling project,
  `continuous-batching-inference-server`'s `hip_port/`, is the one place
  in this portfolio a HIP kernel's actual output is checked against a
  reference: https://github.com/Manas103/continuous-batching-inference-server.
- The retiled fixes change each kernel's algorithm (a shallower K-stage
  for GEMM, a smaller output tile for the stencil, fewer privatized bins
  for the histogram), not just a tuning parameter with identical output;
  this project only analyzes compiled resource usage, so it makes no
  claim that the retiled kernels are numerically equivalent to the ones
  they replace.
- 19 is this project's own number, not a general claim about how many
  kernels in a real codebase would be LDS-bound; see Findings for why
  that exact match is reported with suspicion rather than pride.
