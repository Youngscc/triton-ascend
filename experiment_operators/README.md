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

For every accepted operator case, enumerate all 48 requested triples:

- `depth = 1..4`; each candidate resolves
  `cv_pipeline_depth = cv_num_buffers = depth`
- `multibuffer_num = 1..4`, independently of `depth`; this replaces the ordinary local
  `MarkMultiBuffer` default of 2
- `vf_merge_level = 0, 1, 2`

`multibuffer_num` does not control CVPipeline workspace buffers or the special
preload-local value of 4. The retained CV depth/buffer split remains available
for compiler debugging, but the experiment controller never requests unequal
CV values. There is no ordering constraint between the independent ordinary
`multibuffer_num` axis and CV `depth`, so the sweep includes
`multibuffer_num > depth`.

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

`run_sweep.py` enumerates the full Cartesian space
`depth(1..4) x multibuffer_num(1..4) x merge(0..2)` (48 rows), launches the selected
correctness/benchmark wrapper once per configuration, and retains failures
instead of selecting a best result. Run it only in development mode:

```bash
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  python -u experiment_operators/run_sweep.py \
  --operator unified_attention --warmup 5 --active 30 --timeout 120
```

Results are written under `.codex-remote/results/<run-id>-<operator>/` inside
the server checkout. `measurements.jsonl` and `measurements.csv` contain one
row per requested triple; `manifest.json`, `summary.json`, and per-row logs are
stored alongside them. `--limit N` exists only for controller smoke tests and
must not be used for a formal 48-row run.

Each new row records the public axes `depth`, `multibuffer_num`, and
`vf_merge_level`, plus the resolved audit fields `cv_pipeline_depth` and
`cv_num_buffers`. The two resolved CV fields must both equal `depth`.
Historical result directories use the `legacy-cv-split-v0` schema and must not
be interpreted as measurements of the new ordinary-local `multibuffer_num`.
The intermediate `cv-depth-equals-buffers+local-multibuffer-v1` schema contains
only the older 30-row `multibuffer_num <= depth` subset. New 48-row runs use
`cv-depth-equals-buffers+independent-local-multibuffer-v2`.

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

### Run one complete operator sweep inside the container

When already attached to the `yy-npu` container, pass exactly one Python
operator wrapper to the repository-root entry point. It activates the
development venv, selects the repository-built BishengIR compiler, creates a
fresh Triton cache, and runs all 48 configurations for that operator:

```bash
# The container-entry command in EXECUTION_GUIDE.md starts in the project root.
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

`REMOTE_PROJECT` is the server checkout path configured before following
`EXECUTION_GUIDE.md`; no username-specific project path is required.

Do not run multiple sweeps concurrently on the same NPU. The wrapper writes a
session log under `.codex-remote/logs/`; per-configuration logs and tables
remain under `.codex-remote/results/`. After the selected operator finishes,
it scans every operator already present in that results directory, selects the
latest complete run for each, and regenerates `latest-summary/` and its HTML.
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
correctness, and has both latency and UB. The public tables show the single CV
`depth` axis instead of duplicating its resolved schedule-depth and buffer-count
fields; those audit fields remain available in JSONL and cache metadata. Table
columns start with `depth`, `multibuffer_num`, and `vf_merge_level`, immediately
followed by latency and UB, while hashes and provenance are placed later.

The same command also writes `effects.csv`, `effects.json`, `effects.md`, and
`effects.svg`. These compare each variable on matched controlled slices rather
than using confounded marginal averages:

- depth varies with `multibuffer_num=1` and matches VF merge levels;
- ordinary multibuffer count varies at `depth=4` and matches VF merge levels;
- VF merge level varies across identical `(depth, multibuffer_num)` pairs.

The chart and tables report median latency/UB plus percentage change from the
lowest value of the selected variable. Negative latency change means faster;
positive UB change means more memory. `summary.json` retains per-operator
row/status counts, correctness/latency/UB coverage, timeout counts, distinct
cache/TTIR/binary counts, and range/mean/median statistics. No output ranks
configurations or chooses a winner. Incomplete and `--limit` smoke runs are
ignored, so they cannot displace the latest complete sweep. New-schema formal
runs contain 48 rows per operator.

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
