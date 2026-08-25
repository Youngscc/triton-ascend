#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f /.dockerenv ]]; then
  printf '%s\n' 'Run clean-build-cache.sh inside the experiment container.' >&2
  exit 2
fi

if [[ $# -ne 0 ]]; then
  printf '%s\n' 'Usage: ./tools/remote_experiment/clean-build-cache.sh' >&2
  exit 2
fi

# shellcheck source=tools/remote_experiment/config.sh
source "$SCRIPT_DIR/config.sh"

project_root="$(cd -- "$REMOTE_PROJECT" && pwd -P)"
if [[ "$project_root" == / \
  || ! -f "$project_root/setup.py" \
  || ! -d "$project_root/experiment_operators" ]]; then
  printf 'refusing to clean an invalid project root: %s\n' "$project_root" >&2
  exit 1
fi

managed_path() {
  local label="$1"
  local path="$2"
  local normalized

  normalized="$(realpath -m -- "$path")"
  case "$normalized" in
    "$project_root"/*) ;;
    *)
      printf 'refusing to clean %s outside the project: %s\n' \
        "$label" "$normalized" >&2
      exit 1
      ;;
  esac
  printf '%s\n' "$normalized"
}

require_exact_path() {
  local label="$1"
  local configured
  local expected

  configured="$(managed_path "$label" "$2")"
  expected="$(managed_path "$label" "$3")"
  if [[ "$configured" != "$expected" ]]; then
    printf 'refusing nonstandard %s path: %s (expected %s)\n' \
      "$label" "$configured" "$expected" >&2
    exit 1
  fi
}

remove_tree() {
  local label="$1"
  local path
  path="$(managed_path "$label" "$2")"

  if [[ -e "$path" || -L "$path" ]]; then
    rm -rf -- "$path"
    printf 'removed: %s\n' "$label"
  fi
}

remove_generated_path() {
  local label="$1"
  local path="$2"

  managed_path "$label" "$path" >/dev/null
  if [[ -e "$path" || -L "$path" ]]; then
    rm -rf -- "$path"
    printf 'removed: %s\n' "$label"
  fi
}

# Validate configurable deletion roots before removing anything. Cleaning is
# deliberately limited to the repository's standard generated paths.
require_exact_path 'AscendNPU-IR build' "$REMOTE_COMPILER_BUILD" \
  "$project_root/.codex-remote/ascendnpu-ir-build-explicit"
require_exact_path 'Triton runtime cache' "$REMOTE_TRITON_CACHE" \
  "$project_root/.codex-remote/triton-cache"
require_exact_path 'ccache' "$REMOTE_CCACHE_DIR" \
  "$project_root/.codex-remote/ccache"
require_exact_path 'project temporary files' "$REMOTE_TMP_DIR" \
  "$project_root/tmp"

remove_tree 'Triton CMake build' "$project_root/python/build"
remove_tree 'AscendNPU-IR build' "$REMOTE_COMPILER_BUILD"
remove_tree 'Triton runtime cache' "$REMOTE_TRITON_CACHE"
remove_tree 'project-local ccache' "$REMOTE_CCACHE_DIR"
remove_tree 'temporary host toolchain links' \
  "$project_root/.codex-remote/host-toolchain-bin"
remove_tree 'project temporary files' "$REMOTE_TMP_DIR"
remove_tree 'root Python bytecode cache' "$project_root/__pycache__"
remove_tree 'pytest cache' "$project_root/.pytest_cache"
remove_tree 'mypy cache' "$project_root/.mypy_cache"
remove_tree 'ruff cache' "$project_root/.ruff_cache"

remove_generated_path 'compile_commands.json link' \
  "$project_root/compile_commands.json"
remove_generated_path 'Python package metadata' \
  "$project_root/python/triton.egg-info"
remove_generated_path 'root Python package metadata' \
  "$project_root/triton.egg-info"

shopt -s nullglob
for path in \
  "$project_root/python/triton/_C"/libtriton*.so \
  "$project_root/python/triton/_C/triton-mlir-opt" \
  "$project_root/python/triton/_C/triton-opt"; do
  remove_generated_path "Triton extension/tool $(basename -- "$path")" "$path"
done
shopt -u nullglob

for path in \
  "$project_root/python/triton/backends/ascend" \
  "$project_root/python/triton/backends/amd" \
  "$project_root/python/triton/backends/nvidia" \
  "$project_root/python/triton/language/extra/cann" \
  "$project_root/python/triton/language/extra/kernels" \
  "$project_root/python/triton/language/extra/cuda" \
  "$project_root/python/triton/language/extra/hip" \
  "$project_root/python/triton/tools/extra/cuda" \
  "$project_root/python/triton/tools/extra/hip" \
  "$project_root/python/triton/profiler" \
  "$project_root/python/triton_dist"; do
  if [[ -L "$path" ]]; then
    remove_generated_path "generated link ${path#"$project_root"/}" "$path"
  fi
done

python_cache_count=0
for source_root in \
  "$project_root/python" \
  "$project_root/experiment_operators" \
  "$project_root/third_party/ascend/backend" \
  "$project_root/third_party/ascend/language"; do
  [[ -d "$source_root" ]] || continue
  while IFS= read -r -d '' cache_dir; do
    rm -rf -- "$cache_dir"
    python_cache_count=$((python_cache_count + 1))
  done < <(find "$source_root" -type d \
    \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache \
      -o -name .ruff_cache \) -prune -print0)
done
if (( python_cache_count > 0 )); then
  printf 'removed: Python cache directories (%d)\n' "$python_cache_count"
fi

printf '%s\n' \
  'preserved: .codex-remote/results, .codex-remote/logs, venv, Git metadata, downloaded LLVM'
printf '%s\n' 'CLEAN_BUILD_CACHE_OK'
