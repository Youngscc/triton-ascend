#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEV_VENV="${TRITON_ASCEND_DEV_VENV:-$PROJECT_ROOT/.codex-remote/venv}"
DEV_COMPILER_DIR="${TRITON_ASCEND_COMPILER_DIR:-$PROJECT_ROOT/.codex-remote/ascendnpu-ir-build-explicit/bin}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./run_all_sweeps.sh operator.py
  ./run_all_sweeps.sh --case operator.py FIRST_AXIS MULTIBUFFER VF_MERGE

The first form runs the complete experiment defined in experiment_config.py.
The second reruns one row in the latest complete result. On A5, FIRST_AXIS may
be "off" for the DynamicCV-disabled baseline or a buf_slot_num_of_veccore value.
MULTIBUFFER may be "off" to disable ordinary Auto MultiBuffer, or a count.
EOF
}

if [[ $# -eq 1 ]]; then
  operator_file="$1"
elif [[ $# -eq 5 && "$1" == "--case" ]]; then
  operator_file="$2"
else
  usage
  exit 2
fi

if [[ "$operator_file" != /* ]]; then
  operator_file="$PWD/$operator_file"
fi
if [[ ! -f "$operator_file" || "$operator_file" != *.py ]]; then
  printf 'operator must be an existing Python file: %s\n' "$operator_file" >&2
  exit 1
fi

export TRITON_ASCEND_DEV_VENV="$DEV_VENV"
export TRITON_ASCEND_COMPILER_DIR="$DEV_COMPILER_DIR"
# shellcheck source=tools/remote_experiment/activate-dev-environment.sh
source "$PROJECT_ROOT/tools/remote_experiment/activate-dev-environment.sh"

"$DEV_VENV/bin/python" - <<'PY'
import numpy
import pandas
import torch
import torch_npu
PY

compiler_lib="${DEV_COMPILER_DIR%/bin}/lib"
for file in \
  "meta_op.aic.$TRITON_ASCEND_BITCODE_ARCH.bc" \
  "meta_op.aiv.$TRITON_ASCEND_BITCODE_ARCH.bc" \
  "meta_op.mix.aic.$TRITON_ASCEND_BITCODE_ARCH.bc" \
  "meta_op.mix.aiv.$TRITON_ASCEND_BITCODE_ARCH.bc" \
  host.bc; do
  if [[ ! -s "$compiler_lib/$file" ]]; then
    printf 'missing experiment bitcode: %s\n' "$compiler_lib/$file" >&2
    exit 1
  fi
done

expected_compile="$(realpath "$DEV_COMPILER_DIR/bishengir-compile")"
expected_opt="$(realpath "$DEV_COMPILER_DIR/bishengir-opt")"
EXPECTED_BISHENGIR_COMPILE="$expected_compile" \
EXPECTED_BISHENGIR_OPT="$expected_opt" \
PROJECT_ROOT="$PROJECT_ROOT" "$DEV_VENV/bin/python" - <<'PY'
import os
from pathlib import Path

from triton.backends.ascend import utils

root = Path(os.environ["PROJECT_ROOT"]).resolve()
backend = Path(utils.__file__).resolve()
selected_compile = Path(utils._get_npucompiler_path()[0]).resolve()
selected_opt = Path(utils._get_bishengir_opt_path()[0]).resolve()
expected_compile = Path(os.environ["EXPECTED_BISHENGIR_COMPILE"]).resolve()
expected_opt = Path(os.environ["EXPECTED_BISHENGIR_OPT"]).resolve()

if not backend.is_relative_to(root):
    raise RuntimeError(f"Ascend backend is not from this checkout: {backend}")
if selected_compile != expected_compile:
    raise RuntimeError(
        f"wrong bishengir-compile: {selected_compile}; expected: {expected_compile}"
    )
if selected_opt != expected_opt:
    raise RuntimeError(f"wrong bishengir-opt: {selected_opt}; expected: {expected_opt}")
PY

command -v hivmc >/dev/null
export TRITON_BENCH_METHOD=npu
run_tag="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.codex-remote/triton-cache/$run_tag"
mkdir -p "$TRITON_CACHE_DIR"

if [[ "$1" == "--case" ]]; then
  command=(
    "$DEV_VENV/bin/python" -u "$SCRIPT_DIR/run_sweep.py"
    --case "$operator_file" "$3" "$4" "$5"
  )
else
  command=("$DEV_VENV/bin/python" -u "$SCRIPT_DIR/run_sweep.py" "$operator_file")
fi

"${command[@]}"
PYTHON_BIN="$DEV_VENV/bin/python" "$SCRIPT_DIR/generate_latest_report.sh"

printf 'HTML report: %s\n' \
  "$PROJECT_ROOT/.codex-remote/results/latest-summary/experiment-report.html"
printf 'Combined CSV: %s\n' \
  "$PROJECT_ROOT/.codex-remote/results/latest-summary/combined-results.csv"
