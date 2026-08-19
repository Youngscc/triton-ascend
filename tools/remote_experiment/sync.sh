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
  '--exclude=.codex-remote/'
  '--exclude=tmp/'
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
  '--exclude=logs/'
  '--exclude=results/'
  '--exclude=profiles/'
  '--exclude=artifacts/'
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

remote_mkdir() {
  local quoted_path
  printf -v quoted_path '%q' "$1"
  printf 'mkdir -p -- %s\n' "$quoted_path" | ssh "$REMOTE_HOST" bash -s
}

dry_run_args=()
if [[ "${RSYNC_DRY_RUN:-0}" == "1" ]]; then
  dry_run_args+=(--dry-run --itemize-changes)
else
  remote_mkdir "$REMOTE_PROJECT"
fi

rsync_args=(
  -az
  --partial
  --human-readable
  --progress
  --stats
  "${dry_run_args[@]}"
  --exclude=.git
  --exclude=.codex-remote/
  "${generated_excludes[@]}"
  --exclude=/python/triton/_C/*.so
  --exclude=/python/triton/_C/*.so.*
  --exclude=/python/triton/_C/*.dylib
  --exclude=/python/triton/_C/*.dll
  --exclude=/python/triton/_C/*.pyd
  --exclude=/python/triton/_C/*.pdb
  --exclude=/python/triton/_C/*.exe
  --exclude=/python/triton/_C/*.ilk
  --exclude=/python/triton/_C/triton-mlir-opt
  --exclude=/python/triton/_C/FileCheck
  --exclude=/llvm-project/
  --exclude=/llvm-project-*/
  --exclude=/.llvm-project/
#   --exclude=extracted_stages/
  --exclude=ub_overflow_kernel_candidates/
)

rsync "${rsync_args[@]}" "$PROJECT_ROOT/" "$REMOTE_HOST:$REMOTE_PROJECT/"

# Deletion is deliberately limited to source-only directories. Candidate
# kernels, archived originals, local configuration, and generated artifacts
# remain protected even inside these scopes. Every other project directory is
# always additive.
if [[ "${RSYNC_DELETE:-0}" == "1" ]]; then
  printf '%s\n' \
    'Safe delete scope: experiment_operators/ (excluding candidates/ and origin/)' \
    'Safe delete scope: tools/remote_experiment/ (excluding config.local.sh)'

  safe_delete_args=(
    -az
    --delete-delay
    --human-readable
    --itemize-changes
    --stats
    "${dry_run_args[@]}"
    "${generated_excludes[@]}"
  )
  rsync "${safe_delete_args[@]}" \
    --exclude=/candidates/ \
    --exclude=/origin/ \
    "$PROJECT_ROOT/experiment_operators/" \
    "$REMOTE_HOST:$REMOTE_PROJECT/experiment_operators/"
  rsync "${safe_delete_args[@]}" \
    --exclude=/config.local.sh \
    "$PROJECT_ROOT/tools/remote_experiment/" \
    "$REMOTE_HOST:$REMOTE_PROJECT/tools/remote_experiment/"
fi

# setup.py restores and reapplies the repository's Triton patches with Git.
# Mirror only the top-level metadata into the generated area; nested submodule
# repositories are large and are not needed for those top-level operations.
if [[ "${RSYNC_DRY_RUN:-0}" != "1" ]]; then
  remote_mkdir "$REMOTE_PROJECT/.codex-remote/top-git"
fi
rsync -az --delete "${dry_run_args[@]}" \
  --exclude=modules/ \
  "$PROJECT_ROOT/.git/" \
  "$REMOTE_HOST:$REMOTE_PROJECT/.codex-remote/top-git/"

# AscendNPU-IR is outside the safe deletion allowlist. Transfer its source and
# matching LLVM additively; use Git on the environment machine when its gitlink
# changes so removed dependency files are handled by Git rather than rsync.
rsync -az "${dry_run_args[@]}" \
  --exclude=.git \
  "${generated_excludes[@]}" \
  --exclude=third-party/torch-mlir/externals/llvm-project/ \
  "$PROJECT_ROOT/third_party/ascend/AscendNPU-IR/" \
  "$REMOTE_HOST:$REMOTE_PROJECT/third_party/ascend/AscendNPU-IR/"

if [[ "${RSYNC_DRY_RUN:-0}" == "1" ]]; then
  printf 'Previewed %s -> %s:%s; no files changed\n' \
    "$PROJECT_ROOT" "$REMOTE_HOST" "$REMOTE_PROJECT"
else
  printf 'Synced %s -> %s:%s\n' "$PROJECT_ROOT" "$REMOTE_HOST" "$REMOTE_PROJECT"
fi
