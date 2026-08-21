#!/usr/bin/env bash

# Project-dependent values may remain in a shell after the checkout is moved.
# Recompute them on every load; config.local.sh is the persistent override.
unset REMOTE_PROJECT REMOTE_LOG_DIR REMOTE_VENV REMOTE_COMPILER_BUILD \
  REMOTE_TRITON_CACHE REMOTE_TOP_GIT_DIR REMOTE_TMP_DIR

_REMOTE_CONFIG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_REMOTE_DEFAULT_PROJECT="$(cd -- "$_REMOTE_CONFIG_DIR/../.." && pwd)"
if [[ -f "$_REMOTE_CONFIG_DIR/config.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "$_REMOTE_CONFIG_DIR/config.local.sh"
fi
REMOTE_PROJECT="${REMOTE_PROJECT:-$_REMOTE_DEFAULT_PROJECT}"
REMOTE_CONTAINER="${REMOTE_CONTAINER:-triton-ascend-exp}"
unset _REMOTE_CONFIG_DIR _REMOTE_DEFAULT_PROJECT
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-$REMOTE_PROJECT/.codex-remote/logs}"
REMOTE_VENV="${REMOTE_VENV:-$REMOTE_PROJECT/.codex-remote/venv}"
REMOTE_COMPILER_BUILD="${REMOTE_COMPILER_BUILD:-$REMOTE_PROJECT/.codex-remote/ascendnpu-ir-build-explicit}"
REMOTE_TRITON_CACHE="${REMOTE_TRITON_CACHE:-$REMOTE_PROJECT/.codex-remote/triton-cache}"
REMOTE_TOP_GIT_DIR="${REMOTE_TOP_GIT_DIR:-$REMOTE_PROJECT/.codex-remote/top-git}"
REMOTE_TMP_DIR="${REMOTE_TMP_DIR:-$REMOTE_PROJECT/tmp}"
if [[ -n "${TRITON_ASCEND_DEV_VENV:-}" ]]; then
  REMOTE_VENV="$TRITON_ASCEND_DEV_VENV"
fi
if [[ -n "${TRITON_ASCEND_COMPILER_DIR:-}" ]]; then
  REMOTE_COMPILER_BUILD="${TRITON_ASCEND_COMPILER_DIR%/bin}"
fi
REMOTE_SOURCE_MODE="${REMOTE_SOURCE_MODE:-auto}"
if [[ "$REMOTE_SOURCE_MODE" != "auto" && "$REMOTE_SOURCE_MODE" != "git" \
  && "$REMOTE_SOURCE_MODE" != "rsync" ]]; then
  printf 'REMOTE_SOURCE_MODE must be auto, git, or rsync; got: %s\n' \
    "$REMOTE_SOURCE_MODE" >&2
  return 2 2>/dev/null || exit 2
fi
# Keep the shared container unchanged by default. The current source tree and
# custom compiler are enabled only when REMOTE_MODE=dev is explicitly passed.
REMOTE_MODE="${REMOTE_MODE:-baseline}"
_REMOTE_CANN_HOME="${ASCEND_HOME_PATH:-}"
if [[ -z "$_REMOTE_CANN_HOME" ]]; then
  if [[ -d /usr/local/Ascend/cann ]]; then
    _REMOTE_CANN_HOME=/usr/local/Ascend/cann
  else
    _REMOTE_CANN_HOME=/usr/local/Ascend/ascend-toolkit/latest
  fi
fi
REMOTE_SYSTEM_COMPILER_BIN="${REMOTE_SYSTEM_COMPILER_BIN:-$_REMOTE_CANN_HOME/bin}"
REMOTE_SYSTEM_COMPILER_LIB="${REMOTE_SYSTEM_COMPILER_LIB:-$_REMOTE_CANN_HOME/tools/bishengir/lib}"
REMOTE_CCEC="${REMOTE_CCEC:-$_REMOTE_CANN_HOME/tools/ccec_compiler/bin/ccec}"
REMOTE_LLVM_LINK="${REMOTE_LLVM_LINK:-$_REMOTE_CANN_HOME/tools/bisheng_compiler/bin/llvm-link}"
REMOTE_HOST_CC="${REMOTE_HOST_CC:-${REMOTE_HOST_CLANG:-}}"
if [[ -z "$REMOTE_HOST_CC" ]]; then
  if [[ -x /usr/bin/clang-15 ]]; then
    REMOTE_HOST_CC=/usr/bin/clang-15
  elif [[ -x /usr/bin/clang ]]; then
    REMOTE_HOST_CC=/usr/bin/clang
  else
    REMOTE_HOST_CC="$(command -v clang-15 || command -v clang || true)"
  fi
fi
REMOTE_HOST_CXX="${REMOTE_HOST_CXX:-${REMOTE_HOST_CLANGXX:-}}"
if [[ -z "$REMOTE_HOST_CXX" ]]; then
  if [[ -x /usr/bin/clang++-15 ]]; then
    REMOTE_HOST_CXX=/usr/bin/clang++-15
  elif [[ -x /usr/bin/clang++ ]]; then
    REMOTE_HOST_CXX=/usr/bin/clang++
  else
    REMOTE_HOST_CXX="$(command -v clang++-15 || command -v clang++ || true)"
  fi
fi
REMOTE_HOST_LLD="${REMOTE_HOST_LLD:-$(command -v lld-15 || command -v lld || true)}"
REMOTE_HOST_LD_LLD="${REMOTE_HOST_LD_LLD:-$(command -v ld.lld-15 || command -v ld.lld || true)}"
unset _REMOTE_CANN_HOME
