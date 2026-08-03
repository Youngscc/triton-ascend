#!/usr/bin/env bash

# Override these values in the environment when needed, for example:
#   REMOTE_CONTAINER=other-container ./tools/remote_experiment/run.sh ...
REMOTE_HOST="${REMOTE_HOST:-huawei-server}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/yuanye/code/triton-ascend}"
REMOTE_CONTAINER="${REMOTE_CONTAINER:-sgl-sky}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-$REMOTE_PROJECT/.codex-remote/logs}"

