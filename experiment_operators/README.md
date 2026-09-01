# Three-axis NPU experiment

This directory contains the operator wrappers and controller for measuring every requested compiler configuration. The experiment records UB usage, correctness, compilation outcome, and NPU latency; it does not rank configurations or choose a winner.

## Configuration

Edit `experiment_config.py` before a full run:

```python
A3_DEPTH_VALUES = (1, 2, 3, 4)
A5_INTRA_CACHE_NUM_VALUES = ("off", 1, 2, 3, 4)
MULTIBUFFER_NUM_VALUES = ("off", 1, 2, 3, 4)
VF_MERGE_LEVEL_VALUES = (0, 1)
```

On A3, the first axis is static CV `depth` and DynamicCV is always disabled. On A5, `"off"` disables DynamicCV and numeric values enable it with that `intra_cache_num`; the A5 inter/load cache counts remain fixed at 1. For the second axis, `"off"` passes `multibuffer=False` and omits the explicit local count, while `1` keeps the pass enabled with one buffer. `vf_merge_level=0` disables VF merge.

The default order runs disabled states first. It produces 40 A3 rows and 50 A5 rows.

The same file contains warmup, active measurement, timeout, and timeout-retry values. These settings are not duplicated as command-line switches.

## Operators

The current wrappers are:

- `candidates/fused_attention.py`
- `candidates/flash_attention_npu_v8.py`
- `candidates/unified_attention.py`
- `candidates/hstu_attention.py`

Each wrapper creates deterministic inputs, runs a reference correctness check, and prints one `BENCHMARK` record after a successful NPU measurement. Measurement first uses the CANN NPU profiler. If CANN does not produce `kernel_details.csv`, the wrapper preserves the failed profile and measures the same single-kernel callable with NPU events; the row records `benchmark_method=npu_event_fallback`. The files under `origin/` are retained only as source references; experiments run the wrappers under `candidates/`.

## Run

From the project root inside the configured container:

```bash
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

This is the complete run. There are no simple, detailed, smoke, or report-output modes. Every completed row is written incrementally to both the readable table and the full machine record.

To rerun one existing row in the latest complete result:

```bash
# A3: depth=3, multibuffer_num=2, vf_merge_level=1
./run_all_sweeps.sh --case \
  experiment_operators/candidates/fused_attention.py 3 2 1

# A5 DynamicCV-off baseline
./run_all_sweeps.sh --case \
  experiment_operators/candidates/fused_attention.py off 2 1

# A5: both DynamicCV and ordinary MultiBuffer disabled
./run_all_sweeps.sh --case \
  experiment_operators/candidates/fused_attention.py off off 0
```

A timed-out row is refilled directly. Replacing any other row requires interactive confirmation.

## Results

Each run directory contains only the core experiment artifacts:

- `results.csv`: readable status, reason, latency, UB, attempts, and log path.
- `measurements.jsonl`: complete per-row machine data and the only table containing artifact hashes.
- `manifest.json`: source versions and the exact experiment definition.
- `logs/<case>.log`: complete output for that parameter combination.

Case logs mark correctness, compilation, and benchmark stages. A timeout also
records the case process tree before termination, distinguishing a remaining
compiler child from a Python or NPU launch hang.

The top-level command also refreshes:

```text
.codex-remote/results/latest-summary/experiment-report.html
.codex-remote/results/latest-summary/combined-results.csv
```

The combined CSV uses the same latest complete run for each operator as the
HTML report. It preserves every source `results.csv` row and prefixes the
operator, run ID, and result-directory provenance.

Compare UB usage from two result tables with:

```bash
python experiment_operators/compare_ub_csv.py \
  path/to/first/results.csv path/to/second/results.csv
```

Only rows with the same compiler parameter values are compared. The command prints each UB difference and a summary; exit status `0` means the comparable rows have the same UB, `1` means at least one UB value differs, and `2` means the input is invalid or contains no comparable UB. Missing UB values and parameter combinations present in only one file are counted but are not treated as UB differences.
The old `intra_cache_num` and new `buf_slot_num_of_veccore` CSV columns are treated as the same DynamicCV axis.

The statuses are:

- `measured`: correctness, latency, and nonzero UB data are all present.
- `incorrect`: the compiled kernel failed its reference check.
- `compile_failed`: the compiler or wrapper exited before a valid benchmark.
- `unsupported`: the case timed out, DynamicCV was inapplicable, compiler metadata did not match, or UB data was missing.

See `EXECUTION_GUIDE.md` for container creation, build commands, and the complete operating procedure.

The A5 DynamicCV and ordinary MultiBuffer UB cost model for the
`PlanComputeBlock` boundary is specified in
[`UB_COST_MODEL_DESIGN.md`](UB_COST_MODEL_DESIGN.md).
The guarded reference implementation, PlanLite, certificate schema, regression
suite, and timing benchmark are under
[`cost_model_demo/`](cost_model_demo/README.md).
The complete shape and kernel-structure validation matrix is in
[`DYNAMIC_CV_MODEL_VALIDATION_REPORT.md`](DYNAMIC_CV_MODEL_VALIDATION_REPORT.md).
