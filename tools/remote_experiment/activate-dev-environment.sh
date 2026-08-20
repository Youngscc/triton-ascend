#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' \
    'Source this file: source tools/remote_experiment/activate-dev-environment.sh' >&2
  exit 2
fi

_remote_activate_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/remote_experiment/load-cann-environment.sh
source "$_remote_activate_dir/load-cann-environment.sh"
# shellcheck source=tools/remote_experiment/config.sh
source "$_remote_activate_dir/config.sh"

if [[ ! -x "$REMOTE_VENV/bin/python" ]]; then
  printf 'development venv not found: %s\n' "$REMOTE_VENV" >&2
  return 1
fi
if [[ ! -x "$REMOTE_COMPILER_BUILD/bin/bishengir-compile" ]]; then
  printf 'custom bishengir-compile not found: %s\n' \
    "$REMOTE_COMPILER_BUILD/bin/bishengir-compile" >&2
  return 1
fi
if [[ ! -x "$REMOTE_COMPILER_BUILD/bin/bishengir-opt" ]]; then
  printf 'custom bishengir-opt not found: %s\n' \
    "$REMOTE_COMPILER_BUILD/bin/bishengir-opt" >&2
  printf '%s\n' \
    'Run ./tools/remote_experiment/rebuild-compiler.sh inside the container.' >&2
  return 1
fi

# shellcheck disable=SC1091
source "$REMOTE_VENV/bin/activate"
export PYTHONPATH="$REMOTE_PROJECT/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$REMOTE_COMPILER_BUILD/bin:$PATH"
export TRITON_NPU_COMPILER_PATH="$REMOTE_COMPILER_BUILD/bin"
mkdir -p "$REMOTE_TMP_DIR"
export TMPDIR="$REMOTE_TMP_DIR"
export TMP="$REMOTE_TMP_DIR"
export TEMP="$REMOTE_TMP_DIR"

# The repository-built A5 driver contains the merged native RegBase pipeline.
# Inheriting this legacy switch delegates to CANN's older compiler, whose HIVM
# schema does not include DynamicCV's SSBUF address space.
unset BISHENGIR_LEGACY_A5_REGBASE

TRITON_ASCEND_SOC_NAME="$("$REMOTE_VENV/bin/python" - <<'PY'
import torch
import torch_npu  # noqa: F401

print(torch.npu.get_device_name(torch.npu.current_device()))
PY
)"
export TRITON_ASCEND_SOC_NAME

# Clear the retired A5 text-IR workaround from shells that sourced an older
# version of this script.
unset TRITON_ASCEND_USE_BYTECODE

case "$TRITON_ASCEND_SOC_NAME" in
  *Ascend910_95*|*Ascend950*|*910_958*)
    export TRITON_ASCEND_BITCODE_ARCH=c310
    export BISHENGIR_NATIVE_A5_REGBASE=1
    _hivmc_a5_path="$(command -v hivmc-a5 || true)"
    if [[ -z "$_hivmc_a5_path" ]]; then
      printf '%s\n' 'hivmc-a5 was not found in the active CANN environment.' >&2
      return 1
    fi
    _hivmc_a5_path="$(realpath "$_hivmc_a5_path")"
    # BishengIR consults BISHENG_INSTALL_PATH before PATH. Pin it to the same
    # CANN tool that this shell resolves so an inherited older path cannot win.
    BISHENG_INSTALL_PATH="$(dirname "$_hivmc_a5_path")"
    export BISHENG_INSTALL_PATH
    _hivmc_a5_version="$("$_hivmc_a5_path" --version 2>&1 | head -n 1)"
    ;;
  *Ascend910B*|*Ascend910_93*)
    export TRITON_ASCEND_BITCODE_ARCH=c220
    unset BISHENGIR_NATIVE_A5_REGBASE
    ;;
  *)
    printf 'unsupported or unknown experiment SoC: %s\n' \
      "$TRITON_ASCEND_SOC_NAME" >&2
    return 1
    ;;
esac

printf 'DEV_ENVIRONMENT_OK soc=%s bitcode_arch=%s native_a5_regbase=%s tmpdir=%s\n' \
  "$TRITON_ASCEND_SOC_NAME" "$TRITON_ASCEND_BITCODE_ARCH" \
  "${BISHENGIR_NATIVE_A5_REGBASE:-0}" "$TMPDIR"
if [[ -n "${_hivmc_a5_path:-}" ]]; then
  printf 'A5_HIVMC_OK path=%s version=%s\n' \
    "$_hivmc_a5_path" "$_hivmc_a5_version"
fi
unset _remote_activate_dir _hivmc_a5_path _hivmc_a5_version
