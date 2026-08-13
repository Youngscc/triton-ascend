#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=tools/remote_experiment/config.sh
source "$SCRIPT_DIR/config.sh"

usage() {
  cat <<'EOF'
Usage:
  ./tools/remote_experiment/clean-environment.sh [rebuild|runtime|latest-results|results|all] [--execute]

Scopes:
  rebuild  Remove the project venv and generated Triton/BishengIR build files.
  runtime  Remove experiment logs and Triton caches.
  latest-results
           Keep the latest complete sweep for each operator and latest-summary;
           remove older or incomplete result directories.
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
    rebuild|runtime|latest-results|results|all)
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
declare -a preserved=()
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

if [[ "$scope" == latest-results ]]; then
  results_root="$project/.codex-remote/results"
  if [[ ! -d "$results_root" ]]; then
    printf 'results directory does not exist: %s\n' "$results_root"
    exit 0
  fi

  selector_python="${REMOTE_VENV}/bin/python"
  if [[ ! -x "$selector_python" ]]; then
    selector_python="$(command -v python3 || true)"
  fi
  if [[ -z "$selector_python" || ! -x "$selector_python" ]]; then
    printf '%s\n' 'python3 is required to select the latest complete results' >&2
    exit 1
  fi

  declare -a latest_result_dirs=()
  while IFS= read -r selected; do
    [[ -n "$selected" ]] && latest_result_dirs+=("$(realpath -m -- "$selected")")
  done < <(
    "$selector_python" - "$project" "$results_root" <<'PY'
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve()
results_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(project / "experiment_operators"))

from summarize_latest import find_latest_runs

latest = find_latest_runs(results_root)
for operator in sorted(latest):
    print(latest[operator]["result_dir"])
PY
  )

  if [[ ${#latest_result_dirs[@]} -eq 0 ]]; then
    printf 'refusing cleanup: no complete full-sweep results found under %s\n' \
      "$results_root" >&2
    exit 1
  fi

  if [[ -d "$results_root/latest-summary" ]]; then
    preserved+=("$results_root/latest-summary")
  fi
  preserved+=("${latest_result_dirs[@]}")

  while IFS= read -r result_dir; do
    result_dir="$(realpath -m -- "$result_dir")"
    keep=false
    for selected in "${latest_result_dirs[@]}"; do
      if [[ "$result_dir" == "$selected" ]]; then
        keep=true
        break
      fi
    done
    [[ "$keep" == true ]] && continue

    result_name="$(basename -- "$result_dir")"
    if [[ -f "$result_dir/manifest.json" \
      || -f "$result_dir/measurements.jsonl" \
      || "$result_name" =~ ^[0-9]{8}T[0-9]{6}(Z|\+0800)- ]]; then
      add_target "$result_dir"
    else
      preserved+=("$result_dir")
    fi
  done < <(find "$results_root" -mindepth 1 -maxdepth 1 -type d \
    ! -name latest-summary -print | sort)
  unset result_dir result_name keep selected selector_python results_root
fi

if [[ "$scope" == results || "$scope" == all ]]; then
  add_target "$project/.codex-remote/results"
fi

printf 'cleanup scope: %s\n' "$scope"
printf 'project: %s\n' "$project"
for target in "${preserved[@]}"; do
  printf 'KEEP    %s\n' "$target"
done
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
