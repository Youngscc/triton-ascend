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
4. Baseline execution passes correctness in `sgl-sky`.
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

For every accepted operator case, enumerate all 30 requested triples:

- `cv_pipeline_depth = 1..4`
- `cv_num_buffers = 1..cv_pipeline_depth`
- `vf_merge_level = 0, 1, 2`

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
