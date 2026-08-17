# Three-axis experiment operator candidates

This directory contains **copies** selected from `extracted_stages`. The source
archive is never edited. Minimal launch fixes are made only to the copies.

These files are candidates, not yet a claim that all three compiler controls
affect every kernel. Static screening and baseline correctness have passed;
dynamic parameter-sensitivity screening still requires the development
compiler pipeline.

## Initial candidates

| ID | Copied from | Why it is a candidate | Copy-only change | Current status |
| --- | --- | --- | --- | --- |
| `fused_attention` | `extracted_stages/ascend_tutorials/06-fused-attention.py` | looped attention with `tl.dot` and vector softmax/reduction work | deterministic benchmark wrapper | custom compiler correctness and benchmark passed |
| `unified_attention` | `extracted_stages/pytest_ut/test_triton_unified_attention.py` | looped paged attention with Cube and Vector work | deterministic wrapper; regular block-pointer Cube layout | custom compiler correctness and benchmark passed |
| `hstu_attention` | `extracted_stages/pytest_ut/test_12_hstu_attention.py` | looped forward attention with Cube and Vector work | deterministic forward wrapper | custom compiler correctness and benchmark passed |

The sparse-prefill snapshot `s0_original_baseline_5800us.py` remains deferred:
it reached NPU input allocation but the device failed to start with error
`507033`. It also hard-codes multibuffer-related launch options, which must be
removed or parameterized in a copy before it can be an independent-axis case.

## Screening rules

An operator is promoted from candidate to accepted only when all of these hold:

1. Python parses and defines at least one Triton JIT kernel.
2. The tested call graph contains a looped mixed Cube/Vector region (normally
   `tl.dot` plus vector/reduction work), so CVPipeline and VF merge are relevant.
3. A deterministic launcher, input factory, and correctness oracle exist or
   can be supplied with a small wrapper without rewriting the kernel algorithm.
4. Baseline execution passes correctness in `yy-npu`.
5. None of the three experiment values is hard-coded in a way that overrides
   the requested configuration.
6. In development mode, each axis is proven to reach the intended compiler
   pass. At least one requested value on an axis must change its resolved pass
   state, relevant IR/binary, UB usage, or measured behavior. An axis with no
   observable compiler effect is recorded as insensitive; a kernel insensitive
   to all three axes is excluded from the measurement corpus.

Pure Vector elementwise kernels and pure Cube matmul kernels may be retained as
controls, but they do not qualify as the main mixed-CV corpus.

## Required output semantics

The current default sweep enumerates 32 architecture-specific triples:

- A3 uses `depth = 1..4`, disables DynamicCV, and passes depth through
  BishengIR's native `set_workspace_multibuffer` control
- A5 enables DynamicCV and uses `intra_cache_num = 1..4`, with
  `inter_cache_num = 1`, `load_cache_num = 1`, and
  `set_workspace_multibuffer = 0`
- `multibuffer_num = 1..4`, independently of the architecture-specific first axis; this replaces the ordinary local
  `MarkMultiBuffer` default of 2
- `vf_merge_level = 0, 1`

`vf_merge_level=2` is temporarily excluded because the A5 RegBase pipeline can
produce an SSA dominance error after bufferization. Set
`SWEEP_INCLUDE_VF_MERGE_LEVEL_2=1` only to restore and diagnose all 48 triples.

A3 candidates explicitly set `enable_dynamic_cv_pipeline=false`. A5 candidates
explicitly set it to true and vary the DynamicCV intra-cache count. If the
frontend records a DynamicCV fallback by resolving the option to false, the
controller rejects that artifact as unsupported.

`multibuffer_num` does not control CVPipeline workspace buffers or the
independently inferred preload-local value. CV depth and physical workspace
buffer count retain BishengIR's native single-value behavior. There is no
ordering constraint between the independent ordinary
`multibuffer_num` axis and the architecture-specific first axis.

An explicit `multibuffer_num` also resolves
`limit_auto_multi_buffer_buffer=no-limit`. The upstream default is
`only-cube`: it permits ordinary multibuffering for the Cube-side L1/L0C
buffers of a MIX function but excludes the Vector-side UB Load/Store buffers.
With `only-cube`, changing the ordinary count can therefore leave PlanMemory's
UB result unchanged. `no-limit` allows the same requested count to reach both
sides. All formal values 1 through 4 use `no-limit`, so comparisons within the
sweep vary the count under one fixed strategy. Omitting `multibuffer_num`
retains the upstream `only-cube` default; comparing an omitted value with an
explicit value would change both count and strategy and is not a controlled
count-only comparison.

Write one row for every operator/case/triple, including failures and unsupported
relationships. Successful rows must contain both repeated NPU latency statistics
and non-missing `required_ub_bits`. The experiment reports the complete table;
it does not select or publish a "best" configuration.

## Baseline smoke commands

After synchronization, run each copied file without altering the shared default
environment:

```bash
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  env EXPERIMENT_WARMUP=1 EXPERIMENT_ACTIVE=1 \
  python -u experiment_operators/candidates/fused_attention.py
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  env EXPERIMENT_WARMUP=1 EXPERIMENT_ACTIVE=1 \
  python -u experiment_operators/candidates/unified_attention.py
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  env EXPERIMENT_WARMUP=1 EXPERIMENT_ACTIVE=1 \
  python -u experiment_operators/candidates/hstu_attention.py
```

`REMOTE_MODE=dev` is the verified custom BishengIR 1.2 path. Keep
`REMOTE_MODE=dev-compatible` as the CANN-matched compiler control.
`EXPERIMENT_WARMUP` and `EXPERIMENT_ACTIVE` must be passed after `run.sh` (for
example through `env` above) so they enter the container; their defaults are 5
and 30.

## Three-axis sweep controller

`run_sweep.py` defaults to either A3
`depth(1..4) x multibuffer_num(1..4) x merge(0..1)` or A5
`intra_cache_num(1..4) x multibuffer_num(1..4) x merge(0..1)` (32 rows), launches the selected
correctness/benchmark wrapper once per configuration, and retains failures
instead of selecting a best result. Run it only in development mode:

```bash
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  python -u experiment_operators/run_sweep.py \
  --operator unified_attention --warmup 5 --active 30 --timeout 120 \
  --simple-output
```

Results are written under `.codex-remote/results/<run-id>-<operator>/` inside
the server checkout. The default repository entry point writes `results.csv`,
with one readable row per requested triple, and `logs/<case>.log`, with the
complete stdout/stderr of each configuration. Its result column is `成功`,
`失败`, or `不支持`; it does not expose hashes, cache keys, binary paths, or
compiler commands. `--limit N` exists only for controller smoke tests and must
not be used for a formal 32-row run.

Each new row records the architecture-specific first axis, `multibuffer_num`,
and `vf_merge_level`. A3 metadata must resolve static depth with DynamicCV
disabled. A5 metadata must resolve the requested `intra_cache_num`, fixed
`inter/load=1`, zero static workspace multibuffer, and DynamicCV enabled. Both
require `limit_auto_multi_buffer_buffer=no-limit`; otherwise the controller
rejects the artifact for that requested row.
Historical result directories use the `legacy-cv-split-v0` schema and must not
be interpreted as measurements of the new ordinary-local `multibuffer_num`.
The intermediate `cv-depth-equals-buffers+local-multibuffer-v1` schema contains
only the older 30-row `multibuffer_num <= depth` subset. New A3 runs use
`native-cv-depth+no-dynamic-cv+independent-local-multibuffer-v4`; new A5 runs
use `dynamic-cv-intra-cache+independent-local-multibuffer-v1`. The preceding
`native-cv-depth+independent-local-multibuffer-v3` and
`cv-depth-equals-buffers+independent-local-multibuffer-v2` results remain valid
pre-rollback comparison data.

Run IDs use fixed UTC+8 time and include the numeric UTC offset. This avoids a
runtime dependency on the container's optional IANA timezone database. For
example, `20260805T120449+0800-fused_attention` means 2026-08-05 12:04:49 in
UTC+8. Historical directories ending in `Z` retain their original UTC names.

The controller enables compiler UB reporting and reads `required_ub_bits` from
the Triton cache metadata. A zero or absent value is recorded as missing and
the row is not labeled `measured`, even if correctness and timing completed.
The custom memory-based PlanMemory pipeline reports the maximum successfully
allocated UB address span; when mixed AIC/AIV functions emit multiple values,
the backend records their maximum. Use a fresh cache after rebuilding the
compiler so metadata generated before UB reporting is not reused.
Each candidate also has a subprocess timeout. A kernel that does not return is
retained as an `unsupported` row with `timed_out=true`; it cannot block the
remaining configurations indefinitely.

Set `SWEEP_DETAILED_OUTPUT=1` only when debugging the compiler. That optional
mode writes the full parameter audit, per-row stdout/stderr, hashes, cache
metadata, and aggregate report used by the historical analysis tools.

### Run one complete operator sweep inside the container

When already attached to the `yy-npu` container, pass exactly one Python
operator wrapper to the repository-root entry point. It activates the
development venv, selects the repository-built BishengIR compiler, creates a
fresh Triton cache, and runs all 32 default configurations for that operator:

```bash
# The container-entry command in EXECUTION_GUIDE.md starts in the project root.
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

`REMOTE_PROJECT` is the server checkout path configured before following
`EXECUTION_GUIDE.md`; no username-specific project path is required.

Do not run multiple sweeps concurrently on the same NPU. The default wrapper
updates the same `results.csv` after every configuration, so completed rows
survive an interrupted run. Detailed logs and aggregate HTML are generated
only with `SWEEP_DETAILED_OUTPUT=1`.
Useful overrides are environment variables rather than source edits:

```bash
DRY_RUN=1 ./run_all_sweeps.sh \
  experiment_operators/candidates/fused_attention.py
SWEEP_WARMUP=2 SWEEP_ACTIVE=5 SWEEP_TIMEOUT=60 \
  ./run_all_sweeps.sh experiment_operators/candidates/hstu_attention.py
SWEEP_LIMIT=2 SWEEP_WARMUP=1 SWEEP_ACTIVE=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/unified_attention.py
```

`DRY_RUN=1` validates the operator path and prints the command without
launching an experiment. `SWEEP_LIMIT` is for smoke testing only; incomplete
runs are ignored by latest-run aggregation. Formal measurements should use the
defaults of 5 warmups, 30 active launches, and a 120-second timeout per
candidate. See [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) for the complete
source-sync, execution, result-pull, and reporting procedure.

## Summarize the latest sweeps

After pulling server results to the local checkout, summarize the latest run
for every operator:

```bash
python3 experiment_operators/summarize_latest.py
```

The script writes `.codex-remote/results/latest-summary/supported.csv` and
`supported.md`, containing every configuration that is `measured`, passes
correctness, and has both latency and UB. The public tables use `depth` for A3
and `intra_cache_num` for A5; resolved audit fields remain available in JSONL
and cache metadata. The architecture-specific first axis is followed by
`multibuffer_num`, `vf_merge_level`, latency, and UB.

The same command also writes `effects.csv`, `effects.json`, `effects.md`, and
`effects.svg`. These compare each variable on matched controlled slices rather
than using confounded marginal averages:

- the architecture-specific pipeline axis varies with `multibuffer_num=1` and matches VF merge levels;
- ordinary multibuffer count varies at pipeline-axis value 4 and matches VF merge levels;
- VF merge level varies across identical `(pipeline axis, multibuffer_num)` pairs.

The chart and tables report median latency/UB plus percentage change from the
lowest value of the selected variable. Negative latency change means faster;
positive UB change means more memory. `summary.json` retains per-operator
row/status counts, correctness/latency/UB coverage, timeout counts, distinct
cache/TTIR/binary counts, and range/mean/median statistics. No output ranks
configurations or chooses a winner. Incomplete and `--limit` smoke runs are
ignored, so they cannot displace the latest complete sweep. Current default
formal runs contain 32 rows per operator.

## Generate the interactive latest-results report

Generate a self-contained offline HTML report from the latest complete run for
each operator:

```bash
./experiment_operators/generate_latest_report.sh
```

The default output is
`.codex-remote/results/latest-summary/experiment-report.html`. The page embeds
the selected result data directly, so it can be copied or opened without a web
server or network connection. Select an operator and one x-axis variable, then
fix the other two variables to compare latency and UB side by side. Absolute
and relative-to-first-point modes, hover details, coverage, binary-count, run
provenance, and a configuration table are included. Override paths when needed:

```bash
EXPERIMENT_RESULTS_DIR=/path/to/results \
REPORT_OUTPUT=/path/to/report.html \
  ./experiment_operators/generate_latest_report.sh
```

`run_all_sweeps.sh` regenerates this report after each complete single-operator
sweep, combining that result with the latest complete result of every other
operator already in the results directory.
