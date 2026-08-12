#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEV_VENV="${TRITON_ASCEND_DEV_VENV:-$PROJECT_ROOT/.codex-remote/venv}"
DEV_COMPILER_DIR="${TRITON_ASCEND_COMPILER_DIR:-$PROJECT_ROOT/.codex-remote/ascendnpu-ir-build-explicit/bin}"
WARMUP="${SWEEP_WARMUP:-5}"
ACTIVE="${SWEEP_ACTIVE:-30}"
CANDIDATE_TIMEOUT="${SWEEP_TIMEOUT:-120}"
RUN_TAG="${SWEEP_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${SWEEP_LIMIT:-}"

usage() {
  cat >&2 <<'EOF'
Usage: ./run_all_sweeps.sh /path/to/operator.py

Runs all 48 compiler configurations for exactly one Python operator wrapper,
then rebuilds the latest-result summary and HTML across every operator already
present in .codex-remote/results.

Useful overrides:
  DRY_RUN=1
  SWEEP_WARMUP=5
  SWEEP_ACTIVE=30
  SWEEP_TIMEOUT=120
  SWEEP_LIMIT=N       # smoke test only; incomplete runs are ignored by summary
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

OPERATOR_FILE="$1"
if [[ "$OPERATOR_FILE" != /* ]]; then
  OPERATOR_FILE="$PWD/$OPERATOR_FILE"
fi
if [[ ! -f "$OPERATOR_FILE" ]]; then
  printf 'operator file not found: %s\n' "$OPERATOR_FILE" >&2
  exit 1
fi
if [[ "$OPERATOR_FILE" != *.py ]]; then
  printf 'operator file must end in .py: %s\n' "$OPERATOR_FILE" >&2
  exit 1
fi

OPERATOR_BASENAME="${OPERATOR_FILE##*/}"
OPERATOR_BASENAME="${OPERATOR_BASENAME%.py}"
OPERATOR_TAG="$(printf '%s' "$OPERATOR_BASENAME" | tr -cs '[:alnum:]_.-' '_')"
LOG_DIR="$PROJECT_ROOT/.codex-remote/logs"
RESULTS_DIR="$PROJECT_ROOT/.codex-remote/results"
SESSION_LOG="$LOG_DIR/$RUN_TAG-$OPERATOR_TAG-sweep.log"

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR" "$RESULTS_DIR"
  exec > >(tee -a "$SESSION_LOG") 2>&1
else
  SESSION_LOG="(dry run; no log created)"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  # shellcheck source=tools/remote_experiment/load-cann-environment.sh
  source "$PROJECT_ROOT/tools/remote_experiment/load-cann-environment.sh"
  if [[ ! -f "$DEV_VENV/bin/activate" || ! -x "$DEV_VENV/bin/python" ]]; then
    printf 'development venv not found: %s\n' "$DEV_VENV" >&2
    printf '%s\n' \
      'Inside the experiment container, run: ./tools/remote_experiment/setup-dev-environment.sh' >&2
    exit 1
  fi
  if [[ ! -x "$DEV_COMPILER_DIR/bishengir-compile" ]]; then
    printf 'custom bishengir-compile not found: %s\n' \
      "$DEV_COMPILER_DIR/bishengir-compile" >&2
    exit 1
  fi

  experiment_soc="$("$DEV_VENV/bin/python" - <<'PY'
import torch
import torch_npu  # noqa: F401

print(torch.npu.get_device_name(torch.npu.current_device()))
PY
)"
  case "$experiment_soc" in
    *Ascend910_95*|*Ascend950*|*910_958*) experiment_bitcode_arch=c310 ;;
    *Ascend910B*|*Ascend910_93*) experiment_bitcode_arch=c220 ;;
    *)
      printf 'unsupported or unknown experiment SoC: %s\n' "$experiment_soc" >&2
      exit 1
      ;;
  esac
  compiler_lib="${DEV_COMPILER_DIR%/bin}/lib"
  for file in \
    "meta_op.aic.$experiment_bitcode_arch.bc" \
    "meta_op.aiv.$experiment_bitcode_arch.bc" \
    "meta_op.mix.aic.$experiment_bitcode_arch.bc" \
    "meta_op.mix.aiv.$experiment_bitcode_arch.bc" \
    host.bc; do
    if [[ ! -s "$compiler_lib/$file" ]]; then
      printf 'missing experiment bitcode for %s: %s\n' \
        "$experiment_soc" "$compiler_lib/$file" >&2
      exit 1
    fi
  done

  # shellcheck disable=SC1091
  source "$DEV_VENV/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$DEV_COMPILER_DIR:$PATH"
export TRITON_NPU_COMPILER_PATH="$DEV_COMPILER_DIR"
export TRITON_BENCH_METHOD="${TRITON_BENCH_METHOD:-npu}"
export TRITON_CACHE_DIR="${SWEEP_CACHE_DIR:-$PROJECT_ROOT/.codex-remote/triton-cache/formal-$RUN_TAG}"

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$TRITON_CACHE_DIR"

  expected_bishengir_compile="$(realpath "$DEV_COMPILER_DIR/bishengir-compile")"
  EXPECTED_BISHENGIR_COMPILE="$expected_bishengir_compile" \
    PROJECT_ROOT="$PROJECT_ROOT" "$DEV_VENV/bin/python" - <<'PY'
import os
from pathlib import Path

from triton.backends.ascend import utils

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
utils_file = Path(utils.__file__).resolve()
expected = Path(os.environ["EXPECTED_BISHENGIR_COMPILE"]).resolve()
selected = Path(utils._get_npucompiler_path()[0]).resolve()

if not utils_file.is_relative_to(project_root):
    raise RuntimeError(f"Ascend backend is not from this checkout: {utils_file}")
if selected != expected:
    raise RuntimeError(f"wrong bishengir-compile: {selected}; expected: {expected}")

print(f"ascend_backend={utils_file} (project checkout; required)")
print(f"bishengir_compile={selected} (project build; required)")
PY

  actual_bishengir_opt="$(command -v bishengir-opt || true)"
  actual_hivmc="$(command -v hivmc || true)"
  if [[ -z "$actual_bishengir_opt" || -z "$actual_hivmc" ]]; then
    printf 'missing CANN tools: bishengir-opt=%s hivmc=%s\n' \
      "${actual_bishengir_opt:-not found}" "${actual_hivmc:-not found}" >&2
    exit 1
  fi
fi

PYTHON_BIN="$(command -v python3 || command -v python)"
if [[ "$DRY_RUN" != "1" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

printf 'project=%s\n' "$PROJECT_ROOT"
printf 'operator_file=%s\n' "$OPERATOR_FILE"
printf 'python=%s\n' "$PYTHON_BIN"
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'bishengir_compile=%s\n' "$DEV_COMPILER_DIR/bishengir-compile"
else
  printf 'bitcode_package=soc:%s arch:%s (project build; required)\n' \
    "$experiment_soc" "$experiment_bitcode_arch"
  printf 'bishengir_opt=%s (CANN bytecode reader; expected)\n' \
    "$(realpath "$actual_bishengir_opt")"
  printf 'hivmc=%s (CANN binary backend; expected)\n' \
    "$(realpath "$actual_hivmc")"
fi
printf 'triton_cache=%s\n' "$TRITON_CACHE_DIR"
printf 'results_root=%s\n' "$RESULTS_DIR"
printf 'session_log=%s\n' "$SESSION_LOG"
printf 'benchmark_policy=warmup:%s active:%s timeout:%s\n' \
  "$WARMUP" "$ACTIVE" "$CANDIDATE_TIMEOUT"

command=(
  "$PYTHON_BIN" -u "$SCRIPT_DIR/run_sweep.py"
  --operator-file "$OPERATOR_FILE"
  --warmup "$WARMUP"
  --active "$ACTIVE"
  --timeout "$CANDIDATE_TIMEOUT"
)
if [[ -n "$LIMIT" ]]; then
  command+=(--limit "$LIMIT")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'dry_run command='
  printf '%q ' "${command[@]}"
  printf '\n'
  printf 'dry run complete; no experiment was launched\n'
  exit 0
fi

printf '\nstarting single-operator sweep\n'
"${command[@]}"
printf 'completed operator_file=%s\n' "$OPERATOR_FILE"

printf '\nselecting the latest complete result for every operator\n'
if "$PYTHON_BIN" -u "$SCRIPT_DIR/summarize_latest.py"; then
  PYTHON_BIN="$PYTHON_BIN" "$SCRIPT_DIR/generate_latest_report.sh"
elif [[ -n "$LIMIT" ]]; then
  printf 'No complete sweep is available yet; skipped latest-summary generation.\n'
else
  printf 'Latest-result aggregation failed after a complete sweep.\n' >&2
  exit 1
fi

printf '\nsweep and aggregate report complete\n'
printf 'HTML report: %s/latest-summary/experiment-report.html\n' "$RESULTS_DIR"
printf 'From the local checkout, pull all server results with:\n'
printf './tools/remote_experiment/pull-results.sh\n'
