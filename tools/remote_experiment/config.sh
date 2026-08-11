#!/usr/bin/env bash

_REMOTE_CONFIG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_REMOTE_CONFIG_DIR/config.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "$_REMOTE_CONFIG_DIR/config.local.sh"
fi
unset _REMOTE_CONFIG_DIR

if [[ -z "${REMOTE_PROJECT:-}" || -z "${REMOTE_CONTAINER:-}" ]]; then
  printf '%s\n' \
    'Edit REMOTE_PROJECT and REMOTE_CONTAINER in tools/remote_experiment/config.local.sh first.' >&2
  return 2 2>/dev/null || exit 2
fi
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-$REMOTE_PROJECT/.codex-remote/logs}"
REMOTE_VENV="${REMOTE_VENV:-$REMOTE_PROJECT/.codex-remote/venv}"
REMOTE_COMPILER_BUILD="${REMOTE_COMPILER_BUILD:-$REMOTE_PROJECT/.codex-remote/ascendnpu-ir-build-explicit}"
REMOTE_TRITON_CACHE="${REMOTE_TRITON_CACHE:-$REMOTE_PROJECT/.codex-remote/triton-cache}"
REMOTE_TOP_GIT_DIR="${REMOTE_TOP_GIT_DIR:-$REMOTE_PROJECT/.codex-remote/top-git}"
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
REMOTE_HOST_CLANG="${REMOTE_HOST_CLANG:-/usr/bin/clang}"
REMOTE_HOST_CLANGXX="${REMOTE_HOST_CLANGXX:-/usr/bin/clang++}"
REMOTE_HOST_LLD="${REMOTE_HOST_LLD:-/usr/bin/lld}"
REMOTE_HOST_LD_LLD="${REMOTE_HOST_LD_LLD:-/usr/bin/ld.lld}"
unset _REMOTE_CANN_HOME
