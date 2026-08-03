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
container_command="cd $(printf '%q' "$REMOTE_PROJECT") && exec $command_q"
printf -v container_command_q '%q' "$container_command"
printf -v log_file_q '%q' "$log_file"
log_dir="$(dirname -- "$log_file")"
printf -v log_dir_q '%q' "$log_dir"
printf -v project_q '%q' "$REMOTE_PROJECT"
printf -v container_q '%q' "$REMOTE_CONTAINER"

ssh "$REMOTE_HOST" "
  mkdir -p -- $log_dir_q
  nohup docker exec $container_q bash -c $container_command_q > $log_file_q 2>&1 < /dev/null &
  pid=\$!
  printf 'run_id=%s\\nlog=%s\\npid=%s\\n' '$run_id' '$log_file' \"\$pid\"
"
