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
#   --exclude=extracted_stages/
  --exclude=ub_overflow_kernel_candidates/
)

# Set RSYNC_DELETE=1 only when the server should mirror deletions from local.
if [[ "${RSYNC_DELETE:-0}" == "1" ]]; then
  rsync_args+=(--delete-delay)
fi

rsync "${rsync_args[@]}" "$PROJECT_ROOT/" "$REMOTE_HOST:$REMOTE_PROJECT/"

# The top-level additive sync cannot remove files left by a previously checked
# out AscendNPU-IR revision. Mirror this dependency exactly to the gitlink that
# is checked out locally, while preserving Git metadata. This deliberately
# includes its LLVM source: mixing a new AscendNPU-IR checkout with an older
# server-side LLVM/MLIR tree produces an ABI/API-incompatible compiler build.
rsync -az --delete \
  --exclude=.git \
  --exclude=__pycache__/ \
  --exclude='*.py[cod]' \
  --exclude=build/ \
  --exclude=third-party/torch-mlir/externals/llvm-project/ \
  "$PROJECT_ROOT/third_party/ascend/AscendNPU-IR/" \
  "$REMOTE_HOST:$REMOTE_PROJECT/third_party/ascend/AscendNPU-IR/"

# setup.py normally materializes this package during an editable install. The
# remote workflow executes directly from the source tree, so mirror the actual
# backend source into that generated package. Deletion is scoped to this one
# generated directory and cannot affect builds, caches, results, or other code.
rsync -az --delete \
  --exclude=__pycache__/ \
  --exclude='*.py[cod]' \
  "$PROJECT_ROOT/third_party/ascend/backend/" \
  "$REMOTE_HOST:$REMOTE_PROJECT/python/triton/backends/ascend/"

printf 'Synced %s -> %s:%s\n' "$PROJECT_ROOT" "$REMOTE_HOST" "$REMOTE_PROJECT"
