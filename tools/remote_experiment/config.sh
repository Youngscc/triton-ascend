#!/usr/bin/env bash

# Override these values in the environment when needed, for example:
#   REMOTE_CONTAINER=other-container ./tools/remote_experiment/run.sh ...
REMOTE_HOST="${REMOTE_HOST:-huawei-server}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/yuanye/code/triton-ascend}"
REMOTE_CONTAINER="${REMOTE_CONTAINER:-sgl-sky}"
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
