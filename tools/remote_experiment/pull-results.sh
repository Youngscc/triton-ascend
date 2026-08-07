#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/remote_experiment/config.sh
source "$SCRIPT_DIR/config.sh"
PROJECT_ROOT="$LOCAL_PROJECT"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  printf 'LOCAL_PROJECT does not exist: %s\n' "$PROJECT_ROOT" >&2
  exit 2
fi

LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-$PROJECT_ROOT/.codex-remote/results}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-$PROJECT_ROOT/.codex-remote/logs}"

mkdir -p "$LOCAL_RESULTS_DIR"

rsync_args=(
  -az
  --partial
  --human-readable
  --progress
  --stats
)

# This direction treats the server as the source of truth for generated data.
# Deletion is opt-in because a local checkout may contain historical results
# that have intentionally been removed from the server.
if [[ "${RSYNC_DELETE:-0}" == "1" ]]; then
  rsync_args+=(--delete-delay)
fi

rsync "${rsync_args[@]}" \
  "$REMOTE_HOST:$REMOTE_PROJECT/.codex-remote/results/" \
  "$LOCAL_RESULTS_DIR/"

if [[ "${PULL_SESSION_LOGS:-0}" == "1" ]]; then
  mkdir -p "$LOCAL_LOG_DIR"
  rsync "${rsync_args[@]}" \
    "$REMOTE_HOST:$REMOTE_PROJECT/.codex-remote/logs/" \
    "$LOCAL_LOG_DIR/"
fi

printf 'Pulled results %s:%s/.codex-remote/results -> %s\n' \
  "$REMOTE_HOST" "$REMOTE_PROJECT" "$LOCAL_RESULTS_DIR"
if [[ "${PULL_SESSION_LOGS:-0}" != "1" ]]; then
  printf 'Set PULL_SESSION_LOGS=1 to pull top-level session logs too.\n'
fi
