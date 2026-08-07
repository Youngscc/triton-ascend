# Project memory: local Codex + remote Ascend experiments

This repository is developed locally and experiments run on `huawei-server-A5`.
Read this file before changing the remote-experiment workflow.

Documentation must describe only the current final procedure. Do not retain
change-history narration such as what an older command did, what was replaced,
or how the current procedure differs from a previous version. Git history is
the source for that information.

## Repositories and paths

- Local checkout: `/Users/YokeLove/huawei/triton-ascend`
- Fork remote: `origin` (`Youngscc/triton-ascend`)
- Source remote: `upstream` (`triton-lang/triton-ascend`)
- Experiment branch: local `codex/experiment`, tracking `origin/experiment`
- Server SSH alias: `huawei-server-A5`
- Server project path: `/home/yuanye/code/triton-ascend`
- Experiment container: `sgl-sky`

The `sgl-sky` container bind-mounts the host's `/home`, so the server project
path is also the project path inside the container. Do not use `docker cp` for
normal source synchronization.

## Standard experiment loop

Run these commands from the repository root:

```bash
./tools/remote_experiment/sync.sh
./tools/remote_experiment/rebuild-compiler.sh  # after AscendNPU-IR C++ changes
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  python -u path/to/experiment.py --arg value
./tools/remote_experiment/logs.sh latest
```

`run.sh` starts a detached command inside `sgl-sky` and prints a run ID, log
path, and host PID. `logs.sh` follows the newest log with `tail -F`; pressing
Ctrl-C only stops log following. The container is intentionally invoked with
non-login `bash -c`; this image's `bash -lc` initialization can block.

The scripts and their full options are documented in
`tools/remote_experiment/README.md`.

`run.sh` defaults to `REMOTE_MODE=baseline` and leaves the shared container's
preinstalled Python, Triton, and BishengIR selection untouched. Explicitly
passing `REMOTE_MODE=dev` loads the repository's Python tree, the isolated venv
at `/home/yuanye/.venvs/triton-ascend-dev`, and the custom `bishengir-compile` under
`.codex-remote/ascendnpu-ir-build-explicit/bin`. Use
`REMOTE_MODE=baseline ./tools/remote_experiment/run.sh ...` to load the
container's fully preinstalled Triton and BishengIR instead. Baseline mode is
the control for separating environment failures from source regressions.

`REMOTE_MODE=dev-compatible` is a control mode: it loads the current repository
Triton Python/core and Ascend backend, but selects CANN's preinstalled
BishengIR 1.1 plus `hivmc` 0.2. It is useful for distinguishing a custom
compiler packaging problem from a frontend or operator problem. It cannot
validate compiler options that exist only in the custom BishengIR 1.2.

The current Python tree uses a `libtriton.so` built in the remote-only
`.codex-remote/triton-compatible-src` tree. AscendNPU-IR and its vendored LLVM
must match the top-level repository gitlinks (`d4405acb` and `c195c1d8` at the
time of this note); `sync.sh` exact-mirrors both dependency trees.

## Verified environment state (2026-08-05)

- `REMOTE_MODE=baseline` passes
  `third_party/ascend/tutorials/01-vector-add.py` on NPU 0 with maximum
  Torch/Triton difference `0.0`.
- Dev-mode imports resolve to the current repository's Python and Ascend
  backend, its compatible remote-built `libtriton.so`, and the custom
  BishengIR 1.2.0 compiler.
- The custom BishengIR 1.2 compiler is built from the repository-pinned
  AscendNPU-IR and LLVM. Its executable alone is not a complete toolchain:
  adjacent `lib/meta_op.{aic,aiv,mix.aic,mix.aiv}.c220.bc` and `host.bc` are
  required before the external CANN `hivmc` can produce a binary. The rebuild
  script generates version-matched bitcode from the pinned Template sources
  with CANN's `ccec`; only if a generated file is absent does it create a
  private fallback link to CANN's legacy-named bitcode. It never changes CANN.
- A missing bitcode bundle manifests misleadingly as `Failed to compile
  BiShengLIR to binary`. Running `hivmc` directly on an earlier
  `kernel.npuir.mlir` can additionally report an unknown intermediate op; that
  is not the final IR handed to `hivmc` and must not be used as the root cause.
- The current Triton core now honors an Ascend backend's declared binary
  extensions (including `.mlirbc`) and forwards `mix_mode` when the backend
  metadata provides it. These compatibility fixes are required to combine the
  current Python/core with the CANN-matched compiler.
- `dev-compatible` has completed Python-to-benchmark smoke tests for fused
  attention, unified attention, and HSTU forward attention. Initial 2-warmup,
  5-active means were approximately 2.839409 ms, 57.708093 ms, and 0.044769 ms
  respectively; these are pipeline checks, not formal experiment results.
- Custom BishengIR 1.2 mode now completes unified-attention correctness and
  profiling. The candidate must present Cube operands as regular two-dimensional
  block pointers: one program handles 16 tokens for one query head, and Q, the
  selected paged K/V block, and output use `tl.make_block_ptr`. The earlier
  layout combined four query heads and four tokens into a raw-pointer gather;
  its first `tl.dot(Q, K)` compiled but the generated NPU kernel never returned.
  The repaired original case passed its PyTorch reference and a 5-warmup,
  30-active smoke benchmark reported 0.294887 ms. This is a pipeline check,
  not a formal experiment result.

## Primary experiment objective

The planned experiment accepts a Python-defined Triton operator, lowers it to
TTIR, compiles every valid combination of three backend controls, measures UB
usage and NPU execution time, and preserves the complete per-configuration
measurement table. It does **not** select, rank, or report a best configuration.
The independent variables are:

| Public experiment name | Requested values | Intended compiler control |
| --- | --- | --- |
| `depth` | `1, 2, 3, 4` | set both `CVPipeliningOptions.setDepthInUnrollMode` and the CVPipeline physical-buffer count to the same value |
| `multibuffer_num` | `1, 2, 3, 4` | independently replace the ordinary local `MarkMultiBuffer` default of 2 through `--set-local-multibuffer` |
| `vf_merge_level` | `0, 1, 2` | existing `NPUOptions.vf_merge_level` and `--enable-vf-merge-level` |

This full Cartesian search space has 48 requested combinations. There is no
ordering constraint between ordinary `multibuffer_num` and CV `depth`; include
the `multibuffer_num > depth` cases. Do not silently
coerce or drop a combination. Record it as `unsupported`, `compile_failed`,
`ub_overflow`, `incorrect`, or `measured` with a diagnostic.

Every accepted operator case must therefore produce 48 rows. A failed or
unsupported configuration is still an observation and must remain in the
dataset. Successful rows require both latency statistics and non-missing UB
usage; never keep only the fastest configuration.

### Current capability and gaps

- A Python `@triton.jit` kernel already lowers through AST/semantic analysis to
  TTIR and then through TTAdapter/Linalg and BishengIR/HIVM to an NPU binary.
- "Python operator input" means a Triton kernel plus a launcher/input factory
  and correctness oracle. The repository does not translate an arbitrary
  high-level PyTorch function into Triton automatically. The experiment runner
  must not claim that capability.
- Ascend autotune already compiles configurations and benchmarks callable
  kernels. `TRITON_BENCH_METHOD=npu` selects the NPU-profiler path;
  `do_bench_npu` performs warmup, synchronization, repeated launches, and
  kernel-name-filtered timing.
- The retained CV implementation still supports separate
  `cv_pipeline_depth` and `cv_num_buffers` values for compatibility and
  debugging, but the current experiment deliberately sets both to the single
  public `depth` value. Thus CV scheduling depth and CV workspace-buffer count
  retain the original `depth == numBuf` relationship in all new measurements.
- `NPUOptions.multibuffer_num` is a separate ordinary-local multibuffer
  control. The backend forwards it as `--set-local-multibuffer`; the HFusion
  pipelines use it when estimating ordinary multibuffer pressure and selecting
  tiles, and the HIVM pipelines pass the same value to
  `MarkMultiBufferOptions.localMultiBufferNum` when materializing ordinary
  Load/Store/ND2NZ/Fixpipe local buffers. It does not replace the special
  preload-local count of 4, the Workspace count, or either CVPipeline field.
  The command-line/pass default remains 2 outside explicit experiments, while
  the internal `mark()` helper has no implicit default.
- The earlier CV split/ring implementation remains in the source. It records
  `cv_pipeline_depth = D` and `multibuffer_unroll_factor = N`; unequal CV values
  are no longer part of the requested experiment space because fused-attention
  showed correctness failures and hangs for those historical configurations.
  Depth `1` retains the no-pipeline behavior.
- VF merge is consumed in both the regbase and current memory-based HIVM
  pipelines: level 0 disables the merge pass, level 1 runs it before one-shot
  bufferization, and level 2 after bufferization.
- UB capture is now wired for the current memory-based pipeline as well as
  RegBase. The ordinary `PlanMemory` pass can print the maximum successful UB
  address span, the compile driver forwards
  `--enable-print-memory-allocated-size`, and the Python backend stores the
  maximum value when mixed AIC/AIV functions emit multiple lines. A value of
  zero still means "measurement missing", not "zero UB".
- Dev-mode Python-to-binary compilation is unblocked with the pinned source,
  exact-mirrored LLVM, official A5/NPUIR CMake switches, and generated template
  bitcode. Fused attention, unified attention, and HSTU forward have all passed
  custom-compiler correctness and NPU benchmarks. Unified attention required
  the regular block-pointer layout described above; no compiler/pass logic was
  changed for that repair. Keep vector-add plus candidate correctness tests in
  the validation gate before trusting a sweep.
- Historically, the focused `cv-pipelining.mlir` test passed for equal settings, legacy auto
  mode, and split/ring cases including `(depth,numBuf)=(3,2)` and `(4,1)`.
  Unified-attention end-to-end smoke tests passed correctness and NPU timing for
  independent one-axis changes, including `numBuf < depth`; commands showed all
  three resolved flags and cache keys differed while TTIR stayed identical.
  All sampled binaries were nevertheless identical, so this operator/sample is
  currently classified as dynamically insensitive rather than proof that every
  optimization changes generated code.
- `experiment_operators/run_sweep.py` is the standalone step-4 controller. Its
  current `cv-depth-equals-buffers+independent-local-multibuffer-v2` schema
  enumerates `depth(1..4) x multibuffer_num(1..4) x merge(0..2)`, sets
  `cv_pipeline_depth == cv_num_buffers == depth`, and records all 48 triples.
  It writes incremental JSONL/CSV rows and per-row logs, records compiler time,
  cache/artifact hashes, correctness, timing, and UB metadata, and never
  chooses a winner. A row is `measured` only when correctness, timing, and a
  nonzero UB observation are all present.
- The repository-root `run_all_sweeps.sh` is the container-side entry point for
  one complete operator run: pass exactly one Python wrapper path. It activates
  the isolated development venv, selects the repository-built BishengIR,
  creates a fresh Triton cache, and runs all 48 configurations for that
  operator. It then scans every complete result under `.codex-remote/results`,
  selects the latest run independently for each operator, and refreshes the
  aggregate tables and HTML. Use `DRY_RUN=1` to validate the command without
  launching the NPU; `SWEEP_LIMIT` is smoke-only and cannot displace a complete
  run. The full operator, sync, pull, and reporting procedure is in
  `experiment_operators/EXECUTION_GUIDE.md`.
- User-facing measurement CSVs expose one `depth` column for the equal CV
  schedule-depth/buffer-count pair, followed immediately by
  `multibuffer_num`, `vf_merge_level`, latency, and UB. The duplicated resolved
  CV fields remain only in JSONL/cache metadata for auditing. The latest-run
  summarizer additionally emits `effects.{csv,json,md,svg}` using controlled
  matched comparisons: depth at ordinary buffer 1, ordinary buffer at depth 4,
  and VF merge across identical `(depth, multibuffer_num)` pairs. Do not replace
  these controlled views with marginal averages that hide factor interactions.
- `experiment_operators/generate_latest_report.sh` selects each operator's
  latest complete result and generates the self-contained offline report at
  `.codex-remote/results/latest-summary/experiment-report.html` from
  `experiment_report_template.html`. The UI selects an operator and x-axis
  variable, fixes the other two axes, and shows latency and UB together in
  absolute or relative mode with coverage, provenance, binary counts, hover
  details, and a configuration table. `run_all_sweeps.sh` refreshes it after
  every complete single-operator sweep, combining the new result with the
  latest complete runs of all other operators already present.
- The new ordinary-local count is verified independently of the other
  multibuffer mechanisms. The `mark-multi-buffer-count.mlir` test passes for
  ordinary values 1, 3, and 4 while the preload-local buffer remains 4, and the
  pre-existing MarkMultiBuffer test still passes. HFusion AutoSchedule uses the
  same explicit value rather than a hard-coded 2: its regression case produces
  tile-buffer sizes 39296, 21824, and 11552 bytes for counts 1, 2, and 4.
  An end-to-end unified
  attention smoke case with `(depth,multibuffer_num,merge)=(3,2,1)` showed the
  resolved compiler flags `(CV depth, CV buffers, ordinary local)=(3,3,2)`,
  passed correctness, and completed NPU timing. The one-row controller smoke
  artifact at
  `.codex-remote/results/20260806T144126+0800-unified_attention` uses schema
  `cv-depth-equals-buffers+local-multibuffer-v1` and recorded a measured row
  with non-missing UB. It is a plumbing check, not a formal 30-row experiment.
  After the HFusion estimator was wired to the same ordinary-local option, a
  fresh `(3,3,1)` Python-to-NPU smoke run again passed correctness and reported
  0.302838 ms over 2 warmups and 5 active launches.
- The complete fused/unified/HSTU runs from 2026-08-06 use the intermediate
  `cv-depth-equals-buffers+local-multibuffer-v1` schema. All three retained 30
  rows and all rows were `measured`, but they cover only
  `multibuffer_num <= depth`. They are historical subset data after the v2
  48-row Cartesian schema was introduced and provide no observation for the
  18 newly requested rows where `multibuffer_num > depth`.
- Earlier result directories used the `legacy-cv-split-v0` schema and therefore
  do not measure the new ordinary-local `multibuffer_num` axis. For historical
  context, a 30-row
  unified-attention controller smoke run is mirrored locally at
  `.codex-remote/results/20260805T022522Z-unified_attention`: all 30 rows passed
  correctness and produced timing, 30 distinct cache keys shared one TTIR hash
  and one binary hash, and every row remained `unsupported` because UB was
  missing. This run validated controller completeness but preceded the final
  graph-sync event-ring correction. After that correction, the compiler and
  `bishengir-opt` rebuilt, the CVPipeline and graph-sync regression tests passed,
  `(depth,numBuf,merge)=(4,2,1)` passed unified-attention correctness and an NPU
  timing smoke test, and dev-mode Vector Add still passed with maximum error 0.
- The legacy fresh-cache formal rerun is mirrored at
  `.codex-remote/results/20260805T040449Z-fused_attention` (5 warmups, 30
  active launches, 60-second timeout). It contains all 30 unique triples, one
  frozen TTIR hash, 23 binary hashes, and non-missing compile-time UB for every
  row. Twelve equal `(depth,numBuf)` rows are `measured`; latency spans
  2.860964--4.748766 ms and UB spans 578560--685056 bits. Twelve unequal rows
  fail correctness with a device exception and six unequal rows time out; they
  retain UB but intentionally have no latency. The valid-pair UB observations
  are `(1,1)=685056`, `(2,2)=578560`, `(3,3)=579584`, and
  `(4,4)=580608` bits for every VF merge level. Treat the 4.748766 ms
  `(3,3,0)` and 3.142435 ms `(4,4,0)` single-run values as observations, not
  stable conclusions, until a repeatability audit is run. Treat unequal
  settings as under investigation, not supported, until their
  synchronization/data-lifetime failure is resolved without changing operator
  semantics.
- `run_sweep.py` now gives each candidate an explicit timeout (default 120
  seconds) and records a timeout as an `unsupported` row with return code 124,
  so a non-returning NPU kernel cannot prevent the remaining configurations
  from being observed.
- `experiment_operators/summarize_latest.py` compares both historical UTC
  (`...Z`) and current fixed UTC+8 (`...+0800`) run IDs, selects the newest
  complete artifact directory for each operator, and writes a supported-only
  CSV/Markdown table plus per-operator coverage, failure, latency, UB, and
  artifact-diversity statistics. It labels schema versions so old CV-split
  `numBuf` rows cannot be mistaken for the new ordinary-local
  `multibuffer_num`. It never ranks or selects a configuration.

### Operator-corpus screening

The source archive `/home/yuanye/code/triton-ascend/extracted_stages` currently
mirrors locally as `extracted_stages/`. It contains 477 Python files; all parse,
459 contain Triton JIT code, and only a much smaller subset contains the mixed
Cube/Vector loop structure relevant to these three controls.

Never edit files in `extracted_stages`. Put screened copies and minimal wrapper
fixes under `experiment_operators/candidates/`, and keep the decision log in
`experiment_operators/screening.csv`. The initial baseline-correctness-passing
candidates are fused attention, unified attention, and HSTU forward attention.
They remain candidates until dynamic compiler sensitivity is proven.

Screen each operator in two phases:

1. **Static/functional:** require valid Python, Triton JIT, a looped mixed
   Cube/Vector call graph, deterministic inputs, a correctness oracle, no
   hard-coded override of the three axes, and a passing baseline launch.
2. **Dynamic/compiler:** trace all three values to their intended passes and
   compare resolved pass state, relevant IR, binary hash, UB, and timing. Mark
   an axis `sensitive`, `insensitive`, or `unsupported`, with evidence. Exclude
   a kernel from the main corpus when all three axes are insensitive; retain
   pure Vector or pure Cube kernels only as explicitly labeled controls.

Minimal fixes may add a deterministic main wrapper, input factory, reference
check, reset hook, or parameter injection. They must not change the operator's
algorithm merely to make it appear sensitive. Record source path, copied-file
hash, patch summary, runtime result, and screening reason.

## Implementation plan for the three-axis autotune experiment

### 1. Define a reproducible operator-case contract

Add an experiment runner that loads a Python module containing:

- the `@triton.jit` kernel and grid/launcher;
- an input factory parameterized by shape, dtype, and seed;
- a reference implementation and tolerance-based correctness check;
- optional reset/restore hooks for kernels that mutate inputs.

Generate inputs once per case with a recorded seed and reuse equivalent cloned
inputs for all candidates. Keep shape, dtype, layout, device, and data fixed so
only the three compiler variables change.

### 2. Freeze the frontend boundary at TTIR

Compile the Python kernel through the normal Triton frontend once, dump and
hash its TTIR, and use that TTIR as the common source for the candidate sweep.
This proves that every candidate starts from identical frontend IR and avoids
mixing Python/TTIR variation with backend tuning. Retain a normal Python launch
path for argument creation and final correctness/performance validation.

Use `ir_override`/`triton.compile(<file>.ttir, options=...)` where practical.
If runtime launch metadata makes direct TTIR replay inconvenient, allow normal
JIT compilation but assert and record the TTIR hash for every candidate.

### 3. Plumb CV depth and an independent ordinary multibuffer count

Keep the existing CV split plumbing, but have every experiment candidate set
`cv_pipeline_depth = cv_num_buffers = depth`. Add validated
`NPUOptions.multibuffer_num`, propagate it through
`third_party/ascend/backend/compiler.py` as `--set-local-multibuffer`, then into
both `HFusionPipelineOptions` and `HIVMPipelineOptions`. HFusion must use the
value for its buffer-pressure/tile estimate, and HIVM must forward the same
value to `MarkMultiBufferOptions.localMultiBufferNum`.

The resulting controls must obey these boundaries:

- `depth` controls both CV schedule/unroll depth and CV physical-buffer count;
- `multibuffer_num` replaces only the ordinary local default 2 used for
  HFusion estimation and automatically marked Load/Store/ND2NZ/Fixpipe
  buffers;
- the preload-local explicit 4 and Workspace multibuffer count remain
  unchanged by `multibuffer_num`;
- neither field changes TTIR or the meaning of `vf_merge_level`.

Add focused IR tests for ordinary local values 1, 3, and 4 while proving that
the same pass invocation retains preload-local `hivm.multi_buffer = 4`, and
for HFusion counts 1, 2, and 4 while proving the tiling estimate changes.

### 4. Extend autotune configuration and result capture

Generate `triton.Config` candidates for the 48 requested triples and pass the
three values as backend compile options, not `tl.constexpr` kernel arguments.
Extend the Ascend autotuner (or initially a thin controller around its compile
and benchmark primitives) to retain, for each candidate:

- compile status, diagnostic, compile time, cache key, and binary path/hash;
- resolved depth, equal CV buffer count, ordinary local multibuffer count, and
  VF merge level;
- `required_ub_bits`, plus converted bytes/KiB;
- correctness status and maximum error;
- warmup count, repeat count, raw/summary latency, and benchmark method.

Prefer a standalone controller for the first end-to-end prototype so the
existing general-purpose autotuner remains stable. Once the result schema and
semantics are validated, integrate the hook into `AutoTilingTuner`.

### 5. Make UB measurement explicit

Enable the compiler's UB reporting for every candidate (the existing path uses
`ENABLE_PRINT_UB_BITS=true`) and parse the planned/required UB value into
metadata. Read it from `CompiledKernel.metadata`, or deliberately add it to
packed metadata; do not scrape an unrelated runtime log after the fact.

Distinguish:

- predicted UB used for early pruning;
- actual successful plan-memory UB recorded as the experiment observation;
- device UB capacity used only as the feasibility bound.

Reject or flag a candidate when UB is missing, exceeds capacity, or compilation
reports overflow. Preserve compiler stderr and relevant IR for diagnosis.

### 6. Benchmark every configuration

For each successfully compiled configuration:

1. run the correctness check before timing;
2. warm up and synchronize the NPU;
3. measure repeated launches with `do_bench_npu` or the configured benchmarker;
4. store raw samples when available and at least median/p20/p80 latency;
5. avoid compilation, input generation, copies, and reference execution inside
   the timed region.

Do not rank configurations, calculate a winner, or discard slower results.
Latency and UB are two independent observations. Record every raw/summary
latency and the corresponding UB value for every successful triple. Re-run a
small deterministic sample of configurations only as a stability audit, and
associate each measurement with its exact binary hash so a rebuild cannot be
silently substituted.

### 7. Persist auditable artifacts

Write one experiment directory under `.codex-remote/results/<run-id>/` with:

- `manifest.json`: parent/submodule commits and dirty state, TTIR hash, tool and
  device versions, input specification/seed, environment, and benchmark policy;
- `measurements.jsonl` and `measurements.csv`: one row per requested triple,
  including unsuccessful or unsupported triples;
- `summary.json`: coverage counts and sensitivity classifications only, with no
  best/winner field;
- `ir/` and `logs/`: TTIR plus opt-in failed/intermediate IR and diagnostics.

The stable candidate key must include operator case, TTIR hash, target, all
three parameter values, and compiler version so Triton cache entries cannot
collapse distinct experiments.

### 8. Delivery sequence and gates

1. **Environment gate:** baseline mode still passes Vector Add and remains the
   default; all development tests explicitly use `REMOTE_MODE=dev`.
2. **Compiler gate:** fix the current custom BishengIR-to-`hivmc` failure.
3. **Plumbing gate:** one hand-written TTIR case proves each option reaches the
   intended pass and distinct triples create distinct cache keys/artifacts.
4. **Compile-only gate:** enumerate all 48 triples and classify every result,
   with UB captured or an explicit reason it is unavailable.
5. **Measurement gate:** run correctness and stable NPU timing for every
   successfully compiled configuration and verify the output has exactly one
   row for each requested operator/case/triple.
6. **Corpus gate:** dynamically classify sensitivity for each copied candidate,
   exclude wholly insensitive kernels, and retain the reason for every rejected
   or deferred source.
7. **Generalization gate:** complete the 48-row table for multiple accepted
   mixed Cube/Vector operators; use vector-only and Cube-only cases only as
   labeled negative controls.

Do not mark this objective complete merely because ordinary Triton tiling
autotune works. Completion requires independent control of the public `depth`,
ordinary-local `multibuffer_num`, and `vf_merge_level` variables,
per-candidate UB and latency records, correctness filtering, and a reproducible
complete table for every accepted operator. No winning configuration is part
of the requested deliverable.

## Synchronization safety

Experiment result IDs use fixed UTC+8 time with an explicit `+0800` suffix,
for example `20260805T120449+0800-fused_attention`. A fixed offset is used so
the container does not require an IANA tzdata installation. Historical result
directories ending in `Z` are UTC and must not be renamed because their stored
artifact paths use the original directory name.

The default rsync is additive and excludes `.git`, `.codex-remote`, virtual
environments, build/cache directories, and generated experiment output. Keep
this default for normal iterations. `RSYNC_DELETE=1` enables
`--delete-delay` and can remove files under the server target that are absent
locally; use it only when an exact mirror is intended.

The first sync can traverse many files because this project contains nested
third-party source trees. Later syncs are incremental. Avoid syncing large
generated snapshots or output directories into Git or the experiment mirror.
The vendored LLVM below AscendNPU-IR is intentionally excluded: the preserved
branch requires its patch series to remain applied on the server build tree.

## Submodule caution

`third_party/ascend/AscendNPU-IR` is a Git submodule with further nested
third-party submodules. A dirty submodule status does not necessarily mean
the top-level Triton source changed. Inspect the nested repository before
staging a submodule pointer, and never reset or commit a large nested deletion
without confirming it is intentional.
