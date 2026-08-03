#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=tools/remote_experiment/config.sh
source "$SCRIPT_DIR/config.sh"

ssh "$REMOTE_HOST" "mkdir -p -- '$REMOTE_PROJECT'"

rsync_args=(
  -az
  --partial
  --human-readable
  --progress
  --stats
  --exclude=.git
  --exclude=.codex-remote/
  --exclude=.venv/
  --exclude=__pycache__/
  --exclude='*.py[cod]'
  --exclude=build/
  --exclude=output/
  --exclude=extracted_stages/
  --exclude=ub_overflow_kernel_candidates/
)

# Set RSYNC_DELETE=1 only when the server should mirror deletions from local.
if [[ "${RSYNC_DELETE:-0}" == "1" ]]; then
  rsync_args+=(--delete-delay)
fi

rsync "${rsync_args[@]}" "$PROJECT_ROOT/" "$REMOTE_HOST:$REMOTE_PROJECT/"
printf 'Synced %s -> %s:%s\n' "$PROJECT_ROOT" "$REMOTE_HOST" "$REMOTE_PROJECT"
