#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=tools/remote_experiment/config.sh
source "$SCRIPT_DIR/config.sh"

usage() {
  cat <<'EOF'
Usage:
  ./tools/remote_experiment/clean-environment.sh [rebuild|runtime|results|all] [--execute]

Scopes:
  rebuild  Remove the project venv and generated Triton/BishengIR build files.
  runtime  Remove experiment logs and Triton caches.
  results  Remove all stored experiment results and generated reports.
  all      Remove rebuild, runtime, and result artifacts.

The default scope is rebuild. Without --execute, the command only previews
the paths it would remove.

Always preserved:
  source files, Git metadata, config.local.sh, .codex-remote/llvm, and
  .codex-remote/top-git.
EOF
}

scope=rebuild
execute=false
scope_seen=false
for arg in "$@"; do
  case "$arg" in
    rebuild|runtime|results|all)
      if [[ "$scope_seen" == true ]]; then
        printf '%s\n' 'specify only one cleanup scope' >&2
        usage >&2
        exit 2
      fi
      scope="$arg"
      scope_seen=true
      ;;
    --execute) execute=true ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

project="$(cd -- "$REMOTE_PROJECT" && pwd -P)"
if [[ "$project" == / || "$project" == /home || "$project" == /usr \
  || "$project" == /var || "$project" == /tmp ]]; then
  printf 'refusing unsafe REMOTE_PROJECT: %s\n' "$project" >&2
  exit 1
fi

declare -a targets=()
add_target() {
  local target="$1"
  [[ -n "$target" ]] || return 0
  target="$(realpath -m -- "$target")"
  if [[ "$target" != "$project"/* ]]; then
    printf 'refusing target outside REMOTE_PROJECT: %s\n' "$target" >&2
    exit 1
  fi
  targets+=("$target")
}

if [[ "$scope" == rebuild || "$scope" == all ]]; then
  add_target "$REMOTE_VENV"
  add_target "$REMOTE_COMPILER_BUILD"
  add_target "$project/.codex-remote/host-toolchain-bin"
  add_target "$project/python/build"
  add_target "$project/python/dist"

  while IFS= read -r generated; do
    add_target "$generated"
  done < <(
    find "$project/python" -maxdepth 1 -mindepth 1 \
      \( -name 'triton*.egg-info' -o -name '*.egg-info' \) -print 2>/dev/null \
      | sort
    find "$project/python/triton/_C" -maxdepth 1 -type f \
      \( -name '*.so' -o -name '*.dylib' -o -name '*.pyd' -o -name '*.pdb' \
         -o -name '*.exe' -o -name '*.ilk' -o -name 'triton-mlir-opt' \
         -o -name 'FileCheck' \) -print 2>/dev/null | sort
  )
fi

if [[ "$scope" == runtime || "$scope" == all ]]; then
  add_target "$REMOTE_LOG_DIR"
  add_target "$REMOTE_TRITON_CACHE"
fi

if [[ "$scope" == results || "$scope" == all ]]; then
  add_target "$project/.codex-remote/results"
fi

printf 'cleanup scope: %s\n' "$scope"
printf 'project: %s\n' "$project"
if [[ ${#targets[@]} -eq 0 ]]; then
  printf '%s\n' 'nothing to remove'
  exit 0
fi

for target in "${targets[@]}"; do
  if [[ -e "$target" || -L "$target" ]]; then
    size="$(du -sh -- "$target" 2>/dev/null | awk '{print $1}')"
    printf 'REMOVE  %s  %s\n' "${size:-unknown-size}" "$target"
  else
    printf 'ABSENT  %s\n' "$target"
  fi
done

if [[ "$execute" != true ]]; then
  printf '%s\n' 'preview only; add --execute to remove the listed paths'
  exit 0
fi

for target in "${targets[@]}"; do
  if [[ -e "$target" || -L "$target" ]]; then
    rm -rf -- "$target"
  fi
done

printf 'CLEANUP_OK scope=%s\n' "$scope"
