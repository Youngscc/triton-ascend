#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_ROOT/experiment_operators/run_all_sweeps.sh" "$@"
