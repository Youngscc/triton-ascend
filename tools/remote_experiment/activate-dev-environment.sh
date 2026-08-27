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

# Sourcing this script always selects the repository development toolchain.
# Keep the mode visible to commands started from the activated shell as well.
REMOTE_MODE=dev
export REMOTE_MODE

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

compiler_source="$REMOTE_PROJECT/third_party/ascend/AscendNPU-IR"
build_revision_stamp="$REMOTE_COMPILER_BUILD/.source-revisions"
compiler_fingerprint() {
  sha256sum "$@" | sha256sum | awk '{print $1}'
}
if compiler_revision="$(git -C "$compiler_source" rev-parse HEAD 2>/dev/null)"; then
  compiler_revision="git:$compiler_revision"
else
  compiler_revision="content:$(compiler_fingerprint \
    "$compiler_source/CMakeLists.txt" \
    "$compiler_source/bishengir/include/bishengir/Dialect/HIVM/IR/CMakeLists.txt" \
    "$compiler_source/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td" \
    "$compiler_source/bishengir/tools/bishengir-compile/CMakeLists.txt")"
fi
llvm_project="$compiler_source/third-party/llvm-project"
if llvm_revision="$(git -C "$llvm_project" rev-parse HEAD 2>/dev/null)"; then
  llvm_revision="git:$llvm_revision"
else
  llvm_revision="content:$(compiler_fingerprint \
    "$llvm_project/llvm/CMakeLists.txt" \
    "$llvm_project/cmake/Modules/LLVMVersion.cmake" \
    "$llvm_project/mlir/CMakeLists.txt" \
    "$llvm_project/mlir/include/mlir/IR/Attributes.h")"
fi
expected_build_revisions="ascendnpu_ir=$compiler_revision
llvm_project=$llvm_revision"
actual_build_revisions="missing"
if [[ -f "$build_revision_stamp" ]]; then
  actual_build_revisions="$(cat "$build_revision_stamp")"
fi
if [[ "$actual_build_revisions" != "$expected_build_revisions" ]]; then
  printf 'stale compiler build: expected=%s actual=%s\n' \
    "${expected_build_revisions//$'\n'/,}" \
    "${actual_build_revisions//$'\n'/,}" >&2
  printf '%s\n' \
    'Run ./tools/remote_experiment/rebuild-compiler.sh inside the container.' >&2
  unset -f compiler_fingerprint
  unset compiler_source build_revision_stamp compiler_revision llvm_project \
    llvm_revision expected_build_revisions actual_build_revisions
  return 1
fi
unset -f compiler_fingerprint
unset compiler_source build_revision_stamp compiler_revision llvm_project \
  llvm_revision expected_build_revisions actual_build_revisions

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
