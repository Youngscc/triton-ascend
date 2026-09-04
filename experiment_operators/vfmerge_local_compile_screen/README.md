# VFMerge local compile screen

This is a host-only compile screen. It does not use an NPU, CANN runtime, or
kernel launch.

## Scope and controls

- Compiler: `bishengir-compile 1.2.0`, AscendNPU-IR commit
  `3f962641ff90` (2026-08-31), LLVM 19.1.7.
- Targets: `Ascend910B1` and `Ascend950PR_9579`.
- Inputs: all 163 locally available TTAdapter inputs, compiled at VFMerge
  levels 0, 1, and 2: 489 compiles per target.
- Current candidate sources: fused attention, Flash Attention NPU V8, HSTU
  attention, and unified attention were freshly lowered from their current
  Python sources and compiled separately at all three levels.
- Fixed controls: workspace multibuffer 2, ordinary local multibuffer 1,
  ordinary auto-multibuffer enabled with MIX strategy `no-limit`, and
  PlanMemory seed 0.

An operator is classified as affected only when normalized IR immediately
before and after the actual `hfusion-merge-vf` pass differs. UB output is
recorded independently and is never used as the selection criterion.

The comparison parser was checked against
`hfusion-merge-vf-dependency-update.mlir`: it detects the directed fixture's
four-vector-function merge. This guards against an always-equal comparison.

## Result

For both targets, zero corpus inputs and zero current candidates were directly
rewritten by VFMerge at either level 1 or level 2.

On A3, both pass invocations were observed for 160/163 inputs; the three
remaining extension inputs failed before VFMerge. On A5, both invocations were
observed for 154/163. Every observed pass input contained zero
`hivm.vector_function` functions. `MergeVecScopePass` only collects functions
recognized by `hivm::isVF`, so these invocations are semantic no-ops.

This also explains why binary-hash or UB-only screening is unsafe. On A5, 26
inputs had different reported UB across levels even though the VFMerge
before/after IR was identical. Those rows are not classified as affected.

### Current candidates: A3 final local UB

| Operator | Level 0 | Level 1 | Level 2 | VFMerge affected |
| --- | ---: | ---: | ---: | --- |
| Fused attention | 578560 bit / 70.625 KiB | 578560 / 70.625 | 578560 / 70.625 | No |
| Flash Attention NPU V8 | 703744 / 85.90625 | 703744 / 85.90625 | 703744 / 85.90625 | No |
| HSTU attention | 562176 / 68.625 | 562176 / 68.625 | 562176 / 68.625 | No |
| Unified attention | 218112 / 26.625 | 218112 / 26.625 | 218112 / 26.625 | No |

### Current candidates: A5 host-only observation

| Operator | Level 0 | Level 1 | Level 2 | VFMerge affected |
| --- | ---: | ---: | ---: | --- |
| Fused attention | 699392 bit / 85.375 KiB | 699392 / 85.375 | 831488 / 101.5 | No |
| Flash Attention NPU V8 | 485376 / 59.25 | 485376 / 59.25 | 486400 / 59.375 | No |
| HSTU attention | 659456 / 80.5 | 659456 / 80.5 | 659456 / 80.5 | No |
| Unified attention | 139008 / 16.96875 | 139008 / 16.96875 | unavailable | No; level 2 failed later |

The A5 UB differences above are retained as observations, not evidence that
VFMerge changed the IR.

## `extracted_stages` coverage

The static scan found 45 Python files containing `tl.dot` or `tl.dot_scaled`,
representing 40 unique source hashes. Thirty-two files map to an existing
TTAdapter or one of the four freshly lowered current candidates and therefore
have compile results in `extracted_stages_candidates.csv`. Thirteen source-only
tests or experimental variants do not provide a deterministic signature,
launcher, and constants contract, so they are listed as
`source_only_needs_case_contract` rather than being silently treated as
compiled.

## Artifacts

- `results/operators.csv`: one A3 row per TTAdapter input.
- `results/results.csv`: all 489 A3 level observations.
- `results_a5/operators.csv`: one A5 row per TTAdapter input.
- `results_a5/results.csv`: all 489 A5 level observations.
- `current_candidate_results/operators.csv`: freshly lowered A3 candidate
  results.
- `current_candidate_results_a5/operators.csv`: freshly lowered A5 candidate
  results.
- `current_candidate_provenance.csv`: current Python source and generated
  TTAdapter hashes.
- `extracted_stages_candidates.csv`: static source coverage and mapped compile
  evidence.
- `run_screen.py`: reproducible host-only driver.

The compiler reaches the external `hivmc` boundary after the requested local
passes. A missing `hivmc` therefore does not invalidate a row whose VFMerge IR
and PlanMemory UB were already captured.
