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

command -v docker >/dev/null 2>&1 || {
  printf '%s\n' 'run.sh must be executed on the server host with Docker available.' >&2
  exit 1
}

if [[ "$REMOTE_MODE" == "dev" ]]; then
  if [[ ! -x "$REMOTE_VENV/bin/python" ]]; then
    printf 'development venv not found: %s\n' "$REMOTE_VENV" >&2
    printf '%s\n' \
      'Enter the experiment container and run ./tools/remote_experiment/setup-dev-environment.sh.' >&2
    exit 1
  fi
  if [[ ! -x "$REMOTE_COMPILER_BUILD/bin/bishengir-compile" ]]; then
    printf 'custom compiler not found: %s\n' \
      "$REMOTE_COMPILER_BUILD/bin/bishengir-compile" >&2
    printf '%s\n' \
      'Enter the experiment container and run ./tools/remote_experiment/rebuild-compiler.sh.' >&2
    exit 1
  fi
  if [[ ! -x "$REMOTE_COMPILER_BUILD/bin/bishengir-opt" ]]; then
    printf 'custom bytecode reader not found: %s\n' \
      "$REMOTE_COMPILER_BUILD/bin/bishengir-opt" >&2
    printf '%s\n' \
      'Enter the experiment container and run ./tools/remote_experiment/rebuild-compiler.sh.' >&2
    exit 1
  fi
fi

# Quote each argument for one `bash -c` inside the local server container. The
# image's login-shell initialization can block, so experiments intentionally use
# a non-login shell.
printf -v command_q '%q ' "$@"
printf -v project_q '%q' "$REMOTE_PROJECT"
printf -v python_q '%q' "$REMOTE_PROJECT/python"
printf -v venv_bin_q '%q' "$REMOTE_VENV/bin"
printf -v cache_q '%q' "$REMOTE_TRITON_CACHE"
printf -v cann_env_q '%q' \
  "$REMOTE_PROJECT/tools/remote_experiment/load-cann-environment.sh"
printf -v dev_env_q '%q' \
  "$REMOTE_PROJECT/tools/remote_experiment/activate-dev-environment.sh"
if [[ "$REMOTE_MODE" == "baseline" ]]; then
  printf -v system_compiler_q '%q' "$REMOTE_SYSTEM_COMPILER_BIN"
  container_command="cd $project_q && mkdir -p $cache_q && unset PYTHONPATH TRITON_NPU_COMPILER_PATH && export PATH=$system_compiler_q:\$PATH && export TRITON_CACHE_DIR=$cache_q/baseline && exec $command_q"
elif [[ "$REMOTE_MODE" == "dev-compatible" ]]; then
  printf -v system_compiler_q '%q' "$REMOTE_SYSTEM_COMPILER_BIN"
  container_command="cd $project_q && mkdir -p $cache_q && export PYTHONPATH=$python_q\${PYTHONPATH:+:\$PYTHONPATH} && export PATH=$system_compiler_q:$venv_bin_q:\$PATH && export TRITON_NPU_COMPILER_PATH=$system_compiler_q && export TRITON_CACHE_DIR=$cache_q/dev-compatible && exec $command_q"
elif [[ "$REMOTE_MODE" == "dev" ]]; then
  container_command="cd $project_q && source $dev_env_q && mkdir -p $cache_q && export TRITON_CACHE_DIR=$cache_q/dev-custom && exec $command_q"
else
  echo "REMOTE_MODE must be 'baseline', 'dev-compatible', or 'dev', got: $REMOTE_MODE" >&2
  exit 2
fi
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  printf -v visible_devices_q '%q' "$ASCEND_RT_VISIBLE_DEVICES"
  container_command="export ASCEND_RT_VISIBLE_DEVICES=$visible_devices_q && $container_command"
fi
container_command="source $cann_env_q && $container_command"
log_dir="$(dirname -- "$log_file")"
mkdir -p -- "$log_dir"
nohup docker exec "$REMOTE_CONTAINER" bash -c "$container_command" \
  >"$log_file" 2>&1 < /dev/null &
pid=$!
printf 'run_id=%s\nlog=%s\npid=%s\n' "$run_id" "$log_file" "$pid"
