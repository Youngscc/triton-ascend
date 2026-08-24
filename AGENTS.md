# Project memory: server-side experiments

The server checkout is the primary working tree. Source normally updates with
Git directly on the server. Builds, compiler invocations, experiments, logs,
and reporting run on the server; build and foreground experiment commands run
inside the existing experiment container. Workstation rsync is only a fallback
when the server cannot access GitHub. Read this file before changing the
remote-experiment workflow.

Documentation must describe only the current final procedure. Do not retain
change-history narration such as what an older command did, what was replaced,
or how the current procedure differs from a previous version. Git history is
the source for that information.

Keep the unified A3 and A5/Ascend 950 experiment procedure in
`experiment_operators/EXECUTION_GUIDE.md`.

## Repositories and paths

- Local checkout: `/Users/YokeLove/huawei/triton-ascend`
- Server checkout: `/home/y00969467/triton-ascend`
- Fork remote: `origin` (`Youngscc/triton-ascend`)
- Source remote: `upstream` (`triton-lang/triton-ascend`)
- Experiment branch: local `codex/experiment-main-dev`, based on
  `upstream/main-dev`
- Triton-Ascend source baseline: `41509cb78d0fda91b521b3ad8a896e915f208829`
- AscendNPU-IR upstream baseline: `aea934a66646e837c54fea11e87db54d42eb3221`
- AscendNPU-IR experiment gitlink: `f67dc61f0143f625984c54a0f96bddd9829ba933`
- Active server SSH alias: `huaweiyun`
- Active server project path: `/home/y00969467/triton-ascend`
- Active experiment container: `triton-ascend-exp`

The optional ignored `config.local.sh` can override the project path and
existing container name; without it, `config.sh` uses the current checkout and
container name `triton-ascend-exp`.
Workstation paths and the SSH alias are configured only when the offline
`sync.sh` fallback or optional result retrieval is needed. The server project
path must be visible at the same absolute path inside the container.

## Standard experiment loop

Update the checkout on the server, enter the configured existing container,
and run every build command inside that container:

```bash
git fetch origin
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive

# Enter the container from the server host.
source tools/remote_experiment/config.sh
docker exec -u root -it "$REMOTE_CONTAINER" bash
cd "$REMOTE_PROJECT"
source tools/remote_experiment/config.sh

# Run build commands inside the container.
./tools/remote_experiment/setup-dev-environment.sh  # first build / Triton rebuild
./tools/remote_experiment/rebuild-compiler.sh       # first build / AscendNPU-IR rebuild

# Foreground experiments can also run directly inside the container.
./run_all_sweeps.sh path/to/operator.py

# Detached execution and log following run on the server host.
exit
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  python -u path/to/experiment.py --arg value
./tools/remote_experiment/logs.sh latest
```

The setup script creates or repairs `.codex-remote/venv` and uses the server
clone's own `.git`. The mirrored `.codex-remote/top-git` remains only for the
offline rsync fallback. Do not copy a venv between hosts, containers, or
project paths.

`tools/remote_experiment/clean-build-cache.sh` is the explicit clean-rebuild
entry point. It removes only allowlisted project-local build products, Triton
runtime cache, the project-local `.codex-remote/ccache`, Python caches, and
`tmp/`. It preserves results, logs, the venv, Git metadata, and downloaded LLVM
packages; normal sync and experiment commands never invoke it automatically.

`run.sh` calls the server's local Docker daemon, starts a detached command inside
the configured container, and prints a run ID, log
path, and host PID. `logs.sh` follows the newest log with `tail -F`; pressing
Ctrl-C only stops log following. The container is intentionally invoked with
non-login `bash -c`; this image's `bash -lc` initialization can block.

The scripts and their full options are documented in
`tools/remote_experiment/README.md`.

All experiment commands explicitly pass `REMOTE_MODE=dev`. This loads the
repository's Python tree, the isolated venv at `.codex-remote/venv`, and the
custom `bishengir-compile` under
`.codex-remote/ascendnpu-ir-build-explicit/bin`. The active A3 image's
preinstalled Triton package is incomplete, so baseline mode is not an
environment gate for this checkout.

`REMOTE_MODE=dev-compatible` is a control mode: it loads the current repository
Triton Python/core and Ascend backend, but selects CANN's preinstalled
BishengIR 1.1 plus `hivmc` 0.2. It is useful for distinguishing a custom
compiler packaging problem from a frontend or operator problem. It cannot
validate compiler options that exist only in the custom BishengIR 1.2.

The current Python tree must use a `libtriton.so` built from this checkout in
the server-only development environment. Its first build downloads the
repository-selected prebuilt LLVM because the top-level Triton LLVM and the
AscendNPU-IR vendored LLVM serve different builds. AscendNPU-IR and its vendored
LLVM must match the top-level repository gitlinks in the server checkout.

## Verified environment state (2026-08-10)

- The dedicated `triton-ascend-exp` container uses image
  `swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.0.0-a3-20260723`.
  It is non-privileged, mounts only the active project plus required driver
  resources, and exposes all 16 `/dev/davinci*` devices. The pre-existing
  `tritonsim` container is not modified by this workflow.
- Dev-mode imports resolve to the current repository's Python and Ascend
  backend, its compatible remote-built `libtriton.so`, and the custom
  BishengIR 1.2.0 compiler. The venv interpreter may be a symlink to the
  container's `/usr/local` Python; validate the environment with `sys.prefix`,
  not the resolved `sys.executable` target.
- The project venv contains CMake 3.31.10 because the image's CMake 3.22.1 is
  below AscendNPU-IR's minimum 3.28 requirement. The host/container system
  CMake is unchanged.
- The dedicated compiler reports `bishengir-compile 1.2.0` with LLVM 19.1.7;
  CANN supplies `hivmc 0.2.0`. All four C220 `meta_op` files plus `host.bc`
  were generated successfully. A development-mode Vector Add launch on an
  idle A3 card completed with maximum error 0, and a fresh-cache TTIR compile
  produced an NPU binary through the custom compiler. Recheck card availability
  immediately before performance measurements.
- Experiment runs pin `TRITON_NPU_COMPILER_PATH` to the project build and
  reject any resolved `bishengir-compile` or `bishengir-opt` outside
  `.codex-remote/ascendnpu-ir-build-explicit/bin`. The two tools must be built
  from the same repository-pinned AscendNPU-IR source. CANN's `hivmc` and the
  downstream `bisheng` device backend still come from the installed toolkit.
- A5 detection must not depend exclusively on the optional Python `acl`
  module. The backend checks `TRITON_ASCEND_ARCH`, then ACL when available,
  then `torch.npu.get_device_name`; otherwise an A5 can be misclassified as
  A3, enabling FFTS and producing a launch that does not write output.
- The custom BishengIR 1.2 compiler is built from the repository-pinned
  AscendNPU-IR and LLVM. Its executable alone is not a complete toolchain:
  A3 requires adjacent `lib/meta_op.{aic,aiv,mix.aic,mix.aiv}.c220.bc`, A5
  requires the corresponding `.c310.bc` files, and both require `host.bc`
  before the external CANN `hivmc` can produce a binary. The rebuild script
  detects the SoC, generates version-matched bitcode from the pinned Template
  sources with CANN's `ccec`, and only falls back to same-architecture CANN
  bitcode. Experiment startup rejects a missing or wrong-architecture package.
  It never changes CANN.
- A missing bitcode bundle manifests misleadingly as `Failed to compile
  BiShengLIR to binary`. Running `hivmc` directly on an earlier
  `kernel.npuir.mlir` can additionally report an unknown intermediate op; that
  is not the final IR handed to `hivmc` and must not be used as the root cause.
- The current Triton core now honors an Ascend backend's declared binary
  extensions (including `.mlirbc`) and forwards `mix_mode` when the backend
  metadata provides it. These compatibility fixes are required to combine the
  current Python/core with the CANN-matched compiler.
- The top-level Triton and `triton-mlir-opt` use the repository-selected MLIR
  22, while the standalone `bishengir-compile` uses AscendNPU-IR's pinned MLIR
  19.1.7. Bytecode version 4 is the shared format. CANN's older `bishengir-opt`
  cannot decode DynamicCV's `HIVM_AddressSpaceAttr<ssbuf>`, so the rebuild
  produces the repository-matched MLIR 19 `bishengir-opt` as the compatibility
  reader. `rebuild-compiler.sh` verifies an MLIR 22-to-19 SSBUF round trip with
  the project-built `bishengir-opt` and prints
  `HIVM_SSBUF_BYTECODE_ROUNDTRIP_OK`. It does not run a post-build SSBUF input
  through the full compiler or use downstream `hivmc-a5` as an SSBUF parsing
  gate. Development activation must clear
  `BISHENGIR_LEGACY_A5_REGBASE`; otherwise the wrapper delegates to CANN's old
  A5 compiler, which lacks SSBUF. Do not globally disable bytecode: direct
  TTAdapter text changes the input path and regresses Vector Add.
  The compiler build stores AscendNPU-IR and vendored LLVM source identities in
  `.source-revisions`; Git checkouts use commits and offline rsync checkouts use
  content fingerprints because nested `.git` metadata is intentionally absent.
  A missing or changed stamp cleans the build targets before rebuilding so
  stale TableGen output cannot retain an old HIVM schema.
  The rebuild prints `HIVM_TABLEGEN_SSBUF_OK` only after the generated enum
  implementation itself contains `ssbuf`, before testing either executable.
  The RegBase compile driver keeps the native SSBUF representation through all
  HIVM optimization and memory-planning passes, then lowers rank-0 SSBUF
  pointer casts and zero-index loads/stores to volatile LLVM accesses through
  `!llvm.ptr<11>` immediately before invoking external `hivmc-a5`. Unsupported
  SSBUF users fail explicitly. The focused `hivmc-a5.mlir` test requires the
  external tool, rejects residual SSBUF syntax, and requires a nonempty object;
  CANN 9.0's `hivmc-a5` produced a 2584-byte object in the A3 container. This
  validates the SSBUF boundary representation, not the complete A5 toolchain:
  the A3 container's older `hivmc-a5` rejects current typed bitcode attributes
  when handed a full custom-compiler A5 module.
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
| A3: `depth`; A5: DynamicCV state plus `buf_slot_num_of_veccore` | A3 `1, 2, 3, 4`; A5 `off`, then `1, 2, 3, 4` | A3 uses native static CV workspace/depth with DynamicCV disabled; A5 first runs a disabled baseline with static depth 1, then enables DynamicCV and varies its Vector Core buffer slots while fixing cross-core/GM slots to 1 |
| ordinary MultiBuffer state plus `multibuffer_num` | `off`, then `1, 2, 3, 4` | `off` sets `NPUOptions.multibuffer=False`; numeric values enable the pass, replace the ordinary local `MarkMultiBuffer` default through `--set-local-multibuffer`, and set the MIX strategy to `no-limit` |
| `vf_merge_level` | `0, 1, 2` | existing `NPUOptions.vf_merge_level` and `--enable-vf-merge-level` |

The default search space has 40 A3 or 50 A5 combinations because VF merge
level 2 is temporarily excluded; adding level 2 gives 60 A3 or 75 A5
combinations. A5 always executes its ten DynamicCV-disabled combinations
before the 40 enabled combinations. There is no
ordering constraint between ordinary `multibuffer_num` and the first axis. Do not silently
coerce or drop a combination. Record it as `unsupported`, `compile_failed`,
`ub_overflow`, `incorrect`, or `measured` with a diagnostic.

Every default accepted operator case must therefore produce 40 A3 or 50 A5
rows. A failed or unsupported configuration is still an observation and must remain in the
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
- On A3, CV scheduling depth and workspace-buffer count use BishengIR's native
  `set_workspace_multibuffer` value and DynamicCV is disabled. On A5, `off`
  disables DynamicCV with static workspace depth 1; numeric first-axis values
  enable DynamicCV, set static workspace multibuffer to zero, and use the value
  as `buf_slot_num_of_veccore`, with `buf_slot_num_of_crosscore` plus `buf_slot_num_of_gm` fixed to 1.
  HIVM UnitFlag synchronization is not an experiment axis and must be omitted
  from operator options. The compiler enables it by default on A5 RegBase and
  retains the generic false default on A3; explicitly disabling it on A5 can
  break Cube/Vector synchronization and correctness.
  DynamicCV fallback resolves the metadata switch to false and is rejected as
  unsupported rather than mixed into measurements.
- DynamicCV return code 2 is `ERRCODE_IGNORED`, meaning the pass is not
  applicable to the lowered IR (for example no `linalg.matmul`, a blacklist
  hit, or an existing `scope.scope`). The main-dev pipeline supports
  `scf.while`; HSTU and unified attention must therefore be screened again on
  this branch instead of inheriting their earlier unsupported classification.
- `NPUOptions.multibuffer_num` is a separate ordinary-local multibuffer
  control. The backend forwards it as `--set-local-multibuffer`; the HFusion
  pipelines use it when estimating ordinary multibuffer pressure and selecting
  tiles, and the HIVM pipelines pass the same value to
  `MarkMultiBufferOptions.localMultiBufferNum` when materializing ordinary
  Load/Store/ND2NZ/Fixpipe local buffers. It does not replace the independently
  inferred preload-local count or the native CV workspace/depth value.
  The experiment's `off` value sets `multibuffer=False`, omits
  `multibuffer_num`, and does not override the MIX strategy. An explicit count
  sets `multibuffer=True` and resolves `limit_auto_multi_buffer_buffer` to
  `no-limit`, enabling ordinary multibuffering for the Vector-side UB buffers
  of MIX functions as well as Cube-side L1/L0C buffers. Thus `off` is a real
  pass-disabled control and is not equivalent to numeric value 1. The internal
  `mark()` helper has no implicit default.
- Historical schemas contain experiments with separate CV depth and physical
  buffer values. That split/ring implementation was removed after unequal
  configurations showed correctness failures and hangs. New experiments use
  only the native equal-value behavior. Depth `1` retains the no-pipeline
  behavior.
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
- `experiment_operators/run_sweep.py` is the standalone step-4 controller. A3
  uses schema `native-cv-depth+no-dynamic-cv+local-multibuffer-off-v8`;
  A5 uses `dynamic-cv-slots+local-multibuffer-off+native-unit-flag-v6`.
  All three axis value lists and the benchmark/timeout policy live only in
  `experiment_operators/experiment_config.py`. Disabled values run first. The
  ordinary MIX strategy is `no-limit` only for numeric MultiBuffer values. The
  controller rejects cache metadata that does not resolve the requested
  values, writes every row incrementally, and never chooses a winner. A row is
  `measured` only when correctness, timing, and a nonzero UB observation are
  all present.
- The repository-root `run_all_sweeps.sh` is the container-side entry point for
  one complete operator run: pass exactly one Python wrapper path. It activates
  the isolated development venv, selects the repository-built BishengIR,
  creates a fresh Triton cache, runs the configuration in
  `experiment_config.py`, and refreshes the HTML. There are no simple,
  detailed, dry-run, limited, or progress modes. The `--case` form accepts one
  existing first-axis/multibuffer/VF triple and updates that row in the latest
  complete result; `off` is accepted for A5 DynamicCV and ordinary MultiBuffer.
  A final timeout row is rerun directly; replacing any other
  status requires interactive confirmation.
- Foreground TTY runs render one fixed two-line progress dashboard: aggregate
  counts above and the current parameter triple below. Non-TTY detached logs
  append `CASE_START` and `CASE_RESULT` records instead of repeated progress
  bars, so `tail -F` remains readable without terminal-control artifacts.
- Each run has one readable `results.csv`, one complete machine record
  `measurements.jsonl`, `manifest.json`, and one log per case. Artifact hashes
  appear only in `measurements.jsonl`. Requested parameters, resolved backend
  options, the exact BishengIR command, compiler diagnostics, correctness
  output, and benchmark output remain in the corresponding case log.
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
  ordinary values 1, 3, and 4 while the preload-local buffer retains its
  independently inferred value, and the pre-existing MarkMultiBuffer test still
  passes. HFusion AutoSchedule uses the
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
- `run_sweep.py` gives each candidate the timeout configured in
  `experiment_config.py` and records a timeout as an `unsupported` row with return code 124,
  so a non-returning NPU kernel cannot prevent the remaining configurations
  from being observed.

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

### 3. Select the architecture-specific CV axis and retain independent ordinary multibuffering

On A3, pass `depth` through `NPUOptions.set_workspace_multibuffer` and disable
DynamicCV. On A5, include a DynamicCV-disabled `off` control, then enable
DynamicCV for numeric `NPUOptions.buf_slot_num_of_veccore` values, fix
`buf_slot_num_of_crosscore=1` and `buf_slot_num_of_gm=1`, preserve the
compiler-native UnitFlag synchronization policy, and leave static workspace
multibuffer at zero for enabled rows. Retain the validated
`NPUOptions.multibuffer_num`, propagate it through
`third_party/ascend/backend/compiler.py` as `--set-local-multibuffer`, then into
both `HFusionPipelineOptions` and `HIVMPipelineOptions`. HFusion must use the
value for its buffer-pressure/tile estimate, and HIVM must forward the same
value to `MarkMultiBufferOptions.localMultiBufferNum`.

The resulting controls must obey these boundaries:

- on A3, `depth` controls both CV schedule/unroll depth and CV physical-buffer count;
- on A5, `off` disables DynamicCV; numeric `buf_slot_num_of_veccore` values enable it and cross-core/GM slot counts stay fixed at 1;
- ordinary MultiBuffer `off` sets `NPUOptions.multibuffer=False`, omits the explicit local count, and does not force the MIX strategy;
- `multibuffer_num` replaces only the ordinary local default 2 used for
  HFusion estimation and automatically marked Load/Store/ND2NZ/Fixpipe
  buffers, and an explicit value fixes
  `limit_auto_multi_buffer_buffer=no-limit` so MIX Vector-side UB buffers are
  included;
- numeric `1` remains pass-enabled and must not be treated as equivalent to
  the explicit `off` control;
- the independently inferred preload-local count and native CV
  workspace/depth value remain unchanged by `multibuffer_num`;
- neither field changes TTIR or the meaning of `vf_merge_level`.

Add focused IR tests for ordinary local values 1, 3, and 4 while proving that
the same pass invocation retains the independently inferred preload-local
`hivm.multi_buffer` value, and
for HFusion counts 1, 2, and 4 while proving the tiling estimate changes.

### 4. Extend autotune configuration and result capture

Generate candidates for the 40 A3 or 50 A5 default architecture-specific configurations and pass the
three values as backend compile options, not `tl.constexpr` kernel arguments.
Extend the Ascend autotuner (or initially a thin controller around its compile
and benchmark primitives) to retain, for each candidate:

- compile status, diagnostic, compile time, cache key, and binary path/hash;
- resolved architecture-specific first axis, fixed DynamicCV counts where
  applicable, ordinary local multibuffer count, and VF merge level;
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

- `manifest.json`: parent/submodule commits, exact axis values, operator hash,
  environment, and benchmark policy;
- `measurements.jsonl`: the single complete machine table, with one row per
  requested triple including unsuccessful or unsupported triples;
- `results.csv`: the readable status, latency, UB, retry, and log-path table;
- `logs/`: one complete compiler, correctness, and benchmark log per case.

The stable candidate key must include operator case, TTIR hash, target, all
three parameter values, and compiler version so Triton cache entries cannot
collapse distinct experiments.

### 8. Delivery sequence and gates

1. **Environment gate:** baseline mode still passes Vector Add and remains the
   default; all development tests explicitly use `REMOTE_MODE=dev`.
2. **Compiler gate:** fix the current custom BishengIR-to-`hivmc` failure.
3. **Plumbing gate:** one hand-written TTIR case proves each option reaches the
   intended pass and distinct triples create distinct cache keys/artifacts.
4. **Compile-only gate:** enumerate all 40 A3 or 50 A5 default configurations and classify every result,
   with UB captured or an explicit reason it is unavailable.
5. **Measurement gate:** run correctness and stable NPU timing for every
   successfully compiled configuration and verify the output has exactly one
   row for each requested operator/case/triple.
6. **Corpus gate:** dynamically classify sensitivity for each copied candidate,
   exclude wholly insensitive kernels, and retain the reason for every rejected
   or deferred source.
7. **Generalization gate:** complete the 40-row A3 or 50-row A5 default table for multiple accepted
   mixed Cube/Vector operators; use vector-only and Cube-only cases only as
   labeled negative controls.

Do not mark this objective complete merely because ordinary Triton tiling
autotune works. Completion requires independent control of the architecture-specific
CV axis, ordinary-local `multibuffer_num`, and `vf_merge_level` variables,
per-candidate UB and latency records, correctness filtering, and a reproducible
complete table for every accepted operator. No winning configuration is part
of the requested deliverable.

## Server artifact safety

Experiment result IDs use fixed UTC+8 time with an explicit `+0800` suffix,
for example `20260805T120449+0800-fused_attention`. A fixed offset is used so
the container does not require an IANA tzdata installation. Historical result
directories ending in `Z` are UTC and must not be renamed because their stored
artifact paths use the original directory name.

Keep venvs, compiler builds, caches, logs, raw measurements, and generated
reports under the server checkout's `.codex-remote/`; it is ignored by Git.
Keep transient files produced by `tempfile`, `mktemp`, and compiler subprocesses
under the ignored project-root `tmp/` by exporting `TMPDIR`, `TMP`, and `TEMP`.
Do not commit or move generated build and experiment artifacts into tracked
source directories. The project path must remain identical on the server host
and inside the container.

Offline source synchronization is additive and excludes `.git`,
`.codex-remote`, `tmp`, venvs, build/cache directories, and generated experiment
output. `RSYNC_DELETE=1` enables deletion only inside the allowlisted
`experiment_operators/` and `tools/remote_experiment/` source directories;
candidate kernels, archived originals, `config.local.sh`, and conventional
generated-output directories inside them remain protected. Use
`RSYNC_DRY_RUN=1` with it first to inspect the itemized changes. The project
root, `python/`, `third_party/`, and AscendNPU-IR are never deletion targets.
The top-level Git metadata, excluding `.git/modules`, is mirrored separately to
the replaceable `.codex-remote/top-git` directory so `setup.py` can apply the
Triton patches when the destination is not a real clone. Offline worktrees
explicitly set `REMOTE_SOURCE_MODE=rsync`; normal server clones use `auto` and
their own `.git`. Result retrieval is additive unless deletion is explicitly
enabled.

## Submodule caution

`third_party/ascend/AscendNPU-IR` is a Git submodule with further nested
third-party submodules. A dirty submodule status does not necessarily mean
the top-level Triton source changed. Inspect the nested repository before
staging a submodule pointer, and never reset or commit a large nested deletion
without confirming it is intentional.
