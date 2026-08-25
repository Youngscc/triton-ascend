#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

EXPERIMENT_RESULTS_DIR="${EXPERIMENT_RESULTS_DIR:-$PROJECT_ROOT/.codex-remote/results}"
REPORT_OUTPUT="${REPORT_OUTPUT:-$EXPERIMENT_RESULTS_DIR/latest-summary/experiment-report.html}"
COMBINED_CSV_OUTPUT="${COMBINED_CSV_OUTPUT:-$EXPERIMENT_RESULTS_DIR/latest-summary/combined-results.csv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" "$SCRIPT_DIR/generate_experiment_report.py" \
  --results-dir "$EXPERIMENT_RESULTS_DIR" \
  --output "$REPORT_OUTPUT" \
  --combined-csv "$COMBINED_CSV_OUTPUT"

printf 'Open the report at: %s\n' "$REPORT_OUTPUT"
printf 'Combined CSV: %s\n' "$COMBINED_CSV_OUTPUT"
