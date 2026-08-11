#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f /.dockerenv ]]; then
  printf '%s\n' 'Run setup-dev-environment.sh inside the experiment container.' >&2
  exit 2
fi

# shellcheck source=tools/remote_experiment/load-cann-environment.sh
source "$SCRIPT_DIR/load-cann-environment.sh"
# shellcheck source=tools/remote_experiment/config.sh
source "$SCRIPT_DIR/config.sh"

jobs="${JOBS:-32}"
host_toolchain="$REMOTE_PROJECT/.codex-remote/host-toolchain-bin"

cd "$REMOTE_PROJECT"
repo_git_dir=""
project_top="$(pwd -P)"
if repo_top="$(git -C "$REMOTE_PROJECT" rev-parse --show-toplevel 2>/dev/null)" \
  && repo_top="$(cd "$repo_top" && pwd -P)" \
  && [[ "$repo_top" == "$project_top" ]]; then
  checkout_git_dir="$(git -C "$REMOTE_PROJECT" rev-parse --absolute-git-dir)"
else
  checkout_git_dir=""
fi

if [[ "$REMOTE_SOURCE_MODE" == "rsync" ]]; then
  if [[ -f "$REMOTE_TOP_GIT_DIR/HEAD" ]]; then
    repo_git_dir="$REMOTE_TOP_GIT_DIR"
  fi
elif [[ -n "$checkout_git_dir" ]]; then
  repo_git_dir="$checkout_git_dir"
elif [[ "$REMOTE_SOURCE_MODE" == "auto" && -f "$REMOTE_TOP_GIT_DIR/HEAD" ]]; then
  # Compatibility path for the offline rsync fallback, which mirrors only the
  # top-level Git metadata because the destination is not a normal clone.
  repo_git_dir="$REMOTE_TOP_GIT_DIR"
fi

if [[ -z "$repo_git_dir" ]]; then
  printf 'missing Git metadata for server checkout: %s\n' "$REMOTE_PROJECT" >&2
  printf '%s\n' \
    'Clone the repository on the server, or use sync.sh only when GitHub is unavailable.' >&2
  exit 1
fi
unset checkout_git_dir project_top repo_top
test -x "$REMOTE_HOST_CC" || {
  printf 'missing host C compiler: %s\n' "${REMOTE_HOST_CC:-not found}" >&2
  exit 1
}
test -x "$REMOTE_HOST_CXX" || {
  printf 'missing host C++ compiler: %s\n' "${REMOTE_HOST_CXX:-not found}" >&2
  exit 1
}

if "$REMOTE_HOST_CC" --version 2>/dev/null | head -1 | grep -qi clang \
  && [[ -x "$REMOTE_HOST_LLD" && -x "$REMOTE_HOST_LD_LLD" ]]; then
  mkdir -p "$host_toolchain"
  ln -sfn "$REMOTE_HOST_CC" "$host_toolchain/clang"
  ln -sfn "$REMOTE_HOST_CXX" "$host_toolchain/clang++"
  ln -sfn "$REMOTE_HOST_LLD" "$host_toolchain/lld"
  ln -sfn "$REMOTE_HOST_LD_LLD" "$host_toolchain/ld.lld"
  export PATH="$host_toolchain:$PATH"
  export TRITON_BUILD_WITH_CLANG_LLD=true
  printf 'host build toolchain: clang/lld (%s, %s)\n' \
    "$REMOTE_HOST_CC" "$REMOTE_HOST_CXX"
else
  export CC="$REMOTE_HOST_CC"
  export CXX="$REMOTE_HOST_CXX"
  export TRITON_BUILD_WITH_CLANG_LLD=false
  printf 'host build toolchain: default linker (%s, %s)\n' \
    "$REMOTE_HOST_CC" "$REMOTE_HOST_CXX"
fi

if [[ ! -f "$REMOTE_VENV/bin/activate" || ! -x "$REMOTE_VENV/bin/python" ]]; then
  printf 'creating or repairing project venv: %s\n' "$REMOTE_VENV"
  python3 -m venv --system-site-packages "$REMOTE_VENV"
fi

# shellcheck disable=SC1091
source "$REMOTE_VENV/bin/activate"

if ! python - <<'PY'
import re
import subprocess

text = subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0]
match = re.search(r"(\d+)\.(\d+)", text)
raise SystemExit(0 if match and tuple(map(int, match.groups())) >= (3, 28) else 1)
PY
then
  # AscendNPU-IR requires CMake >= 3.28. Keep it private to the project venv.
  python -m pip install \
    --index-url https://repo.huaweicloud.com/repository/pypi/simple \
    --timeout 300 --retries 5 \
    'cmake>=3.28,<4'
fi

export MAX_JOBS="$jobs"
export TRITON_BUILD_WITH_CCACHE=true
export TRITON_BUILD_PROTON=OFF
export TRITON_BUILD_DISTRIBUTED=OFF
export TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF
export GIT_DIR="$repo_git_dir"
export GIT_WORK_TREE="$REMOTE_PROJECT"

python -m pip install --no-build-isolation --no-deps -e .
PYTHONPATH="$REMOTE_PROJECT/python" python - <<'PY'
import triton
from triton._C import libtriton

print(triton.__version__)
print(triton.__file__)
print(libtriton.__file__)
PY
