#!/usr/bin/env bash

# One-time user configuration: edit only these two absolute project paths.
LOCAL_PROJECT="/CHANGE_ME/local/triton-ascend"
REMOTE_PROJECT="/CHANGE_ME/server/triton-ascend"

# These names are part of the fixed experiment environment.
REMOTE_HOST="huawei-server"
REMOTE_CONTAINER="sgl-sky"

if [[ "$LOCAL_PROJECT" == /CHANGE_ME/* || "$REMOTE_PROJECT" == /CHANGE_ME/* ]]; then
  printf '%s\n' \
    'Edit LOCAL_PROJECT and REMOTE_PROJECT in tools/remote_experiment/config.sh first.' >&2
  return 2 2>/dev/null || exit 2
fi
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-$REMOTE_PROJECT/.codex-remote/logs}"
REMOTE_VENV="${REMOTE_VENV:-/home/yuanye/.venvs/triton-ascend-dev}"
REMOTE_COMPILER_BUILD="${REMOTE_COMPILER_BUILD:-$REMOTE_PROJECT/.codex-remote/ascendnpu-ir-build-explicit}"
REMOTE_TRITON_CACHE="${REMOTE_TRITON_CACHE:-$REMOTE_PROJECT/.codex-remote/triton-cache}"
# Keep the shared container unchanged by default. The current source tree and
# custom compiler are enabled only when REMOTE_MODE=dev is explicitly passed.
REMOTE_MODE="${REMOTE_MODE:-baseline}"
REMOTE_SYSTEM_COMPILER_BIN="${REMOTE_SYSTEM_COMPILER_BIN:-/usr/local/Ascend/ascend-toolkit/latest/bin}"
REMOTE_SYSTEM_COMPILER_LIB="${REMOTE_SYSTEM_COMPILER_LIB:-/usr/local/Ascend/ascend-toolkit/latest/tools/bishengir/lib}"
REMOTE_BISHENG_COMPILER_BIN="${REMOTE_BISHENG_COMPILER_BIN:-/usr/local/Ascend/ascend-toolkit/latest/tools/bisheng_compiler/bin}"
