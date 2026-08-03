#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/remote_experiment/config.sh
source "$SCRIPT_DIR/config.sh"

run_id="${1:-latest}"
lines="${LINES:-80}"

if [[ "$run_id" == "latest" ]]; then
  log_file="$(ssh "$REMOTE_HOST" "find '$REMOTE_LOG_DIR' -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-")"
  if [[ -z "$log_file" ]]; then
    echo "no experiment logs found under $REMOTE_LOG_DIR" >&2
    exit 1
  fi
else
  log_file="$REMOTE_LOG_DIR/$run_id.log"
fi

echo "following $REMOTE_HOST:$log_file (Ctrl-C to stop following)"
ssh "$REMOTE_HOST" "exec tail -n '$lines' -F '$log_file'"

