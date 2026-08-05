#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/remote_experiment/config.sh
source "$SCRIPT_DIR/config.sh"

if (( $# == 0 )); then
  echo "usage: $0 <command> [args ...]" >&2
  echo "example: $0 python -u test.py" >&2
  exit 2
fi

run_id="$(date +%Y%m%d-%H%M%S)-$$"
log_file="$REMOTE_LOG_DIR/$run_id.log"

# Quote each local argument for bash on the remote host, then pass the whole
# command as one argument to `bash -c` inside the container. The image's
# login-shell initialization can block, so experiments intentionally use a
# non-login shell.
printf -v command_q '%q ' "$@"
printf -v project_q '%q' "$REMOTE_PROJECT"
printf -v python_q '%q' "$REMOTE_PROJECT/python"
printf -v compiler_bin_q '%q' "$REMOTE_COMPILER_BUILD/bin"
printf -v venv_bin_q '%q' "$REMOTE_VENV/bin"
printf -v cache_q '%q' "$REMOTE_TRITON_CACHE"
if [[ "$REMOTE_MODE" == "baseline" ]]; then
  printf -v system_compiler_q '%q' "$REMOTE_SYSTEM_COMPILER_BIN"
  container_command="cd $project_q && mkdir -p $cache_q && unset PYTHONPATH TRITON_NPU_COMPILER_PATH && export PATH=$system_compiler_q:\$PATH && export TRITON_CACHE_DIR=$cache_q/baseline && exec $command_q"
elif [[ "$REMOTE_MODE" == "dev-compatible" ]]; then
  printf -v system_compiler_q '%q' "$REMOTE_SYSTEM_COMPILER_BIN"
  container_command="cd $project_q && mkdir -p $cache_q && export PYTHONPATH=$python_q\${PYTHONPATH:+:\$PYTHONPATH} && export PATH=$system_compiler_q:$venv_bin_q:\$PATH && export TRITON_NPU_COMPILER_PATH=$system_compiler_q && export TRITON_CACHE_DIR=$cache_q/dev-compatible && exec $command_q"
elif [[ "$REMOTE_MODE" == "dev" ]]; then
  container_command="cd $project_q && mkdir -p $cache_q && export PYTHONPATH=$python_q\${PYTHONPATH:+:\$PYTHONPATH} && export PATH=$compiler_bin_q:$venv_bin_q:\$PATH && export TRITON_NPU_COMPILER_PATH=$compiler_bin_q && export TRITON_CACHE_DIR=$cache_q/dev-custom && exec $command_q"
else
  echo "REMOTE_MODE must be 'baseline', 'dev-compatible', or 'dev', got: $REMOTE_MODE" >&2
  exit 2
fi
printf -v container_command_q '%q' "$container_command"
printf -v log_file_q '%q' "$log_file"
log_dir="$(dirname -- "$log_file")"
printf -v log_dir_q '%q' "$log_dir"
printf -v container_q '%q' "$REMOTE_CONTAINER"

ssh "$REMOTE_HOST" "
  mkdir -p -- $log_dir_q
  nohup docker exec $container_q bash -c $container_command_q > $log_file_q 2>&1 < /dev/null &
  pid=\$!
  printf 'run_id=%s\\nlog=%s\\npid=%s\\n' '$run_id' '$log_file' \"\$pid\"
"
