#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/remote_experiment/config.sh
source "$SCRIPT_DIR/config.sh"
if [[ -z "${LOCAL_PROJECT:-}" || -z "${REMOTE_HOST:-}" ]]; then
  printf '%s\n' \
    'sync.sh requires LOCAL_PROJECT and REMOTE_HOST in config.local.sh.' >&2
  exit 2
fi
if [[ "$REMOTE_SOURCE_MODE" != "rsync" ]]; then
  printf '%s\n' \
    'sync.sh is the offline fallback; set REMOTE_SOURCE_MODE="rsync" in config.local.sh first.' >&2
  exit 2
fi
PROJECT_ROOT="$LOCAL_PROJECT"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  printf 'LOCAL_PROJECT does not exist: %s\n' "$PROJECT_ROOT" >&2
  exit 2
fi

# Never transfer host-specific caches or generated directories. Keep this
# common list on every source rsync below so a nested dependency cannot
# reintroduce a file excluded by the top-level transfer. Binary exclusions are
# scoped below because LLVM source contains tracked binary test fixtures.
generated_excludes=(
  '--exclude=__pycache__/'
  '--exclude=*.py[cod]'
  '--exclude=.pytest_cache/'
  '--exclude=.mypy_cache/'
  '--exclude=.ruff_cache/'
  '--exclude=.hypothesis/'
  '--exclude=.tox/'
  '--exclude=.nox/'
  '--exclude=.cache/'
  '--exclude=.clangd/'
  '--exclude=.vscode/'
  '--exclude=.vs/'
  '--exclude=.idea/'
  '--exclude=.cursor/'
  '--exclude=.coverage'
  '--exclude=.coverage.*'
  '--exclude=htmlcov/'
  '--exclude=*.egg-info/'
  '--exclude=.eggs/'
  '--exclude=.venv/'
  '--exclude=venv/'
  '--exclude=venv.bak/'
  '--exclude=build/'
  '--exclude=build-*/'
  '--exclude=cmake-build-*/'
  '--exclude=dist/'
  '--exclude=output/'
  '--exclude=CMakeFiles/'
  '--exclude=.ninja_deps'
  '--exclude=.ninja_log'
  '--exclude=compile_commands.json'
  '--exclude=.DS_Store'
  '--exclude=*.swp'
)

ssh "$REMOTE_HOST" "mkdir -p -- '$REMOTE_PROJECT'"

rsync_args=(
  -az
  --partial
  --human-readable
  --progress
  --stats
  --exclude=.git
  --exclude=.codex-remote/
  "${generated_excludes[@]}"
  --exclude=/llvm-project/
  --exclude=/llvm-project-*/
  --exclude=/.llvm-project/
  --exclude=/python/triton/_C/*.so
  --exclude=/python/triton/_C/*.dylib
  --exclude=/python/triton/_C/*.pyd
  --exclude=/python/triton/_C/*.pdb
  --exclude=/python/triton/_C/*.exe
  --exclude=/python/triton/_C/*.ilk
  --exclude=/python/triton/_C/triton-mlir-opt
  --exclude=/python/triton/_C/triton-opt
  --exclude=/python/triton/_C/FileCheck
  --exclude=/python/triton/FileCheck
  --exclude=/python/triton/backends/*/
  --exclude=/python/triton/language/extra/*/
  --exclude=/python/triton/tools/extra/
  --exclude=/python/triton/profiler/
  --exclude=/python/triton/instrumentation/
#   --exclude=extracted_stages/
  --exclude=ub_overflow_kernel_candidates/
)

# Set RSYNC_DELETE=1 only when the server should mirror deletions from local.
if [[ "${RSYNC_DELETE:-0}" == "1" ]]; then
  rsync_args+=(--delete-delay)
fi

rsync "${rsync_args[@]}" "$PROJECT_ROOT/" "$REMOTE_HOST:$REMOTE_PROJECT/"

# setup.py restores and reapplies the repository's Triton patches with Git.
# Mirror only the top-level metadata into the generated area; nested submodule
# repositories are large and are not needed for those top-level operations.
ssh "$REMOTE_HOST" "mkdir -p -- '$REMOTE_PROJECT/.codex-remote/top-git'"
rsync -az --delete \
  --exclude=modules/ \
  "$PROJECT_ROOT/.git/" \
  "$REMOTE_HOST:$REMOTE_PROJECT/.codex-remote/top-git/"

# The top-level additive sync cannot remove files left by a previously checked
# out AscendNPU-IR revision. Mirror this dependency exactly to the gitlink that
# is checked out locally, while preserving Git metadata. This deliberately
# includes its LLVM source: mixing a new AscendNPU-IR checkout with an older
# server-side LLVM/MLIR tree produces an ABI/API-incompatible compiler build.
rsync -az --delete \
  --exclude=.git \
  "${generated_excludes[@]}" \
  --exclude=third-party/torch-mlir/externals/llvm-project/ \
  "$PROJECT_ROOT/third_party/ascend/AscendNPU-IR/" \
  "$REMOTE_HOST:$REMOTE_PROJECT/third_party/ascend/AscendNPU-IR/"

# setup.py normally materializes this package during an editable install. The
# remote workflow executes directly from the source tree, so mirror the actual
# backend source into that generated package. Deletion is scoped to this one
# generated directory and cannot affect builds, caches, results, or other code.
rsync -az --delete \
  "${generated_excludes[@]}" \
  "$PROJECT_ROOT/third_party/ascend/backend/" \
  "$REMOTE_HOST:$REMOTE_PROJECT/python/triton/backends/ascend/"

printf 'Synced %s -> %s:%s\n' "$PROJECT_ROOT" "$REMOTE_HOST" "$REMOTE_PROJECT"
