#!/usr/bin/env bash

# Create and configure the fixed single-card A5 experiment container.
# Run this file as an executable on the A5 server host from any directory.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'Run this script as an executable; do not source it:' >&2
  printf '  %s\n' './tools/remote_experiment/setup-a5-container.sh' >&2
  return 2
fi

set -euo pipefail

container_name="yy-npu"
image_name="quay.io/ascend/cann:9.0.0-950-ubuntu22.04-py3.11"
physical_device_id="7"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"

say() {
  printf '[A5 setup] %s\n' "$*"
}

fail() {
  printf '[A5 setup] ERROR: %s\n' "$*" >&2
  exit 1
}

add_bind_if_present() {
  local source_path="$1"
  local target_path="$2"
  if [[ -e "$source_path" ]]; then
    docker_args+=(--volume "$source_path:$target_path")
  else
    say "optional host path is absent; skip mount: $source_path"
  fi
}

add_optional_a5_devices() {
  local device_dir device_path
  for device_dir in /dev/uburma /dev/ummu; do
    if [[ -c "$device_dir" ]]; then
      docker_args+=(--device "$device_dir:$device_dir:rwm")
    elif [[ -d "$device_dir" ]]; then
      while IFS= read -r -d '' device_path; do
        docker_args+=(--device "$device_path:$device_path:rwm")
      done < <(find "$device_dir" -type c -print0 2>/dev/null)
    fi
  done
}

command -v docker >/dev/null 2>&1 || fail 'docker is not installed or not in PATH'
command -v npu-smi >/dev/null 2>&1 || fail 'npu-smi is not installed or not in PATH'

say "project: $project_root"
say 'checking A5 host NPU state'
npu-smi info

if docker image inspect "$image_name" >/dev/null 2>&1; then
  say "using existing official CANN image: $image_name"
else
  say "pulling official CANN image: $image_name"
  docker pull "$image_name"
fi

if docker container inspect "$container_name" >/dev/null 2>&1; then
  container_status="$(docker container inspect --format '{{.State.Status}}' "$container_name")"
  say "container already exists: $container_name (status=$container_status)"
  if [[ "$container_status" != "running" ]]; then
    say "start it with: docker start $container_name"
  fi
  say "enter it with: docker exec -u root -it $container_name /bin/bash"
  say "to recreate it, review and run: docker rm -f $container_name"
  say 'no existing container was changed'
  exit 0
fi

docker_args=(
  run
  --user 0
  --detach
  --interactive
  --tty
  --name "$container_name"
  --net host
  --workdir "$project_root"
  --shm-size 512g
  --security-opt seccomp=unconfined
)

if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"ascend"'; then
  say 'using Ascend Docker Runtime device injection'
  docker_args+=(
    --runtime ascend
    --env "ASCEND_VISIBLE_DEVICES=$physical_device_id"
  )
else
  say 'Ascend Docker Runtime is unavailable; using explicit A5 device mounts'
  required_devices=(
    "/dev/davinci$physical_device_id"
    /dev/davinci_manager
    /dev/hisi_hdc
  )
  for required_device in "${required_devices[@]}"; do
    [[ -c "$required_device" ]] || fail "required A5 device is absent: $required_device"
    docker_args+=(--device "$required_device:$required_device:rwm")
  done
  add_optional_a5_devices
  docker_args+=(--env "ASCEND_RT_VISIBLE_DEVICES=$physical_device_id")
fi

add_bind_if_present /usr/local/dcmi /usr/local/dcmi
add_bind_if_present /usr/local/bin/npu-smi /usr/local/bin/npu-smi
add_bind_if_present /usr/local/sbin/npu-smi /usr/local/sbin/npu-smi
add_bind_if_present /usr/local/Ascend/driver /usr/local/Ascend/driver
add_bind_if_present /etc/ascend_install.info /etc/ascend_install.info
docker_args+=(--volume /home:/home)

if [[ "$project_root" != /home/* ]]; then
  docker_args+=(--volume "$project_root:$project_root")
fi

docker_args+=("$image_name" /bin/bash)

say "creating container: $container_name"
docker "${docker_args[@]}"

quoted_project_root="$(printf '%q' "$project_root")"
say 'configuring the repository-local Python environment'
docker exec -u root "$container_name" bash -c "
  set -e
  cd $quoted_project_root
  if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    source /usr/local/Ascend/cann/set_env.sh
  elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  else
    echo 'CANN set_env.sh is missing' >&2
    exit 1
  fi
  python3 -m venv --system-site-packages .codex-remote/venv
  source .codex-remote/venv/bin/activate
  python -m pip install -e .
"

say 'verifying the visible NPU count and device name'
docker exec -u root "$container_name" bash -c "
  set -e
  cd $quoted_project_root
  if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    source /usr/local/Ascend/cann/set_env.sh
  else
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  fi
  source .codex-remote/venv/bin/activate
  python - <<'PY'
import torch
import torch_npu

count = torch.npu.device_count()
print('visible NPU count:', count)
if count != 1:
    raise SystemExit(f'expected exactly one visible NPU, got {count}')
torch.npu.set_device(0)
print('logical npu:0:', torch.npu.get_device_name(0))
PY
"

say 'setup complete'
say "enter the container with: docker exec -u root -it $container_name /bin/bash"
