# PlanComputeBlock IR comparison

This directory contains host-only MLIR snapshots generated on macOS. No NPU,
CANN runtime, kernel launch, or benchmark was used.

## Compared revisions

- `main` experiment revision: `d2dfc8b77e0fb05cf463bcdaf168ce90cd47cb2b`
- `main-dev` experiment revision: `9ffbfdb9ee7d81de0de05e4275e868c47c3bdd8e`
- AscendNPU-IR gitlink in both revisions:
  `4b9f1a56092d66a991b857ca4ca2b40f2cf06e53`

Both revisions were compiled against the same LLVM 22 macOS arm64 package.
The four candidate files differ only in experiment-option wrappers, progress
logging, and timing fallback outside the selected Triton JIT functions.

## Fixed compiler controls

- target: `Ascend950PR_9579`
- DynamicCV: enabled
- Vector/intra cache count: `4`
- Cross-core/inter cache count: `1`
- GM/load cache count: `1`
- static workspace multibuffer: `0`
- ordinary local multibuffer: enabled, count `2`
- VF merge level: `0`

Operator shapes, signatures, constants, and source hashes are recorded in each
operator's `metadata.json`.

## Snapshot boundary

The local build emits IR before every pass. The IR labeled as
`after-plan-compute-block.mlir` is the input of `OpClassifierPass`, which is
the pass immediately following `PlanComputeBlockPass`; it is therefore the
output of `PlanComputeBlockPass` without another transformation in between.

Each tracked operator directory contains:

- `after-plan-compute-block.mlir`: compared stage snapshot
- `metadata.json`: input and compiler-option provenance

Running `dump_plan_compute_block.py` also generates `optimized.ttir.mlir`,
`final.ttadapter.mlir`, and the complete `mlir-pass-dump.log`. Those larger,
reproducible diagnostics are intentionally not tracked.

## Result

All four raw stage files differ because the experiment wrappers add lines and
therefore shift MLIR source-location metadata. After normalizing only source
paths and `loc(file:line:column)` metadata, all four pairs are byte-for-byte
identical. Operation counts and line counts also match.

See `comparison.csv` and `comparison.json`. Empty files under `diffs/` mean
there is no normalized semantic diff.

## Build-only compatibility

The pinned `main-dev` revision contains one post-PlanComputeBlock Fixpipe
builder call for the newer AscendNPU-IR API while its gitlink points to the
older API above. The temporary macOS worktree removed that extra builder
argument solely to link the host library. This source is in
`SplitDataflow/InterCoreTransferAndSync.cpp`, after the captured boundary, and
neither PlanComputeBlock implementation nor its input was patched.
