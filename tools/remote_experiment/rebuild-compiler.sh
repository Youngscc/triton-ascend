#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f /.dockerenv ]]; then
  printf '%s\n' 'Run rebuild-compiler.sh inside the experiment container.' >&2
  exit 2
fi

# shellcheck source=tools/remote_experiment/load-cann-environment.sh
source "$SCRIPT_DIR/load-cann-environment.sh"
# shellcheck source=tools/remote_experiment/config.sh
source "$SCRIPT_DIR/config.sh"

mkdir -p "$REMOTE_TMP_DIR"
export TMPDIR="$REMOTE_TMP_DIR"
export TMP="$REMOTE_TMP_DIR"
export TEMP="$REMOTE_TMP_DIR"
unset BISHENGIR_LEGACY_A5_REGBASE

if [[ ! -f "$REMOTE_VENV/bin/activate" || ! -x "$REMOTE_VENV/bin/python" ]]; then
  printf 'development venv not found: %s\n' "$REMOTE_VENV" >&2
  printf '%s\n' \
    'Inside the container, run: ./tools/remote_experiment/setup-dev-environment.sh' >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REMOTE_VENV/bin/activate"

jobs="${JOBS:-32}"
compiler_source="$REMOTE_PROJECT/third_party/ascend/AscendNPU-IR"
llvm_source="$compiler_source/third-party/llvm-project/llvm"
system_bc_lib="$REMOTE_SYSTEM_COMPILER_LIB"
build_lib="$REMOTE_COMPILER_BUILD/lib"
toolchain_bin="$REMOTE_COMPILER_BUILD/cann-toolchain-bin"
build_revision_stamp="$REMOTE_COMPILER_BUILD/.source-revisions"

content_fingerprint() {
  sha256sum "$@" | sha256sum | awk '{print $1}'
}

if compiler_revision="$(git -C "$compiler_source" rev-parse HEAD 2>/dev/null)"; then
  compiler_revision="git:$compiler_revision"
  compiler_identity_mode=git
else
  compiler_revision="content:$(content_fingerprint \
    "$compiler_source/CMakeLists.txt" \
    "$compiler_source/bishengir/include/bishengir/Dialect/HIVM/IR/CMakeLists.txt" \
    "$compiler_source/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td" \
    "$compiler_source/bishengir/tools/bishengir-compile/CMakeLists.txt")"
  compiler_identity_mode=content
fi
llvm_project="$compiler_source/third-party/llvm-project"
if llvm_revision="$(git -C "$llvm_project" rev-parse HEAD 2>/dev/null)"; then
  llvm_revision="git:$llvm_revision"
  llvm_identity_mode=git
else
  llvm_revision="content:$(content_fingerprint \
    "$llvm_project/llvm/CMakeLists.txt" \
    "$llvm_project/cmake/Modules/LLVMVersion.cmake" \
    "$llvm_project/mlir/CMakeLists.txt" \
    "$llvm_project/mlir/include/mlir/IR/Attributes.h")"
  llvm_identity_mode=content
fi
printf 'BISHENGIR_SOURCE_ID compiler_mode=%s llvm_mode=%s\n' \
  "$compiler_identity_mode" "$llvm_identity_mode"
expected_build_revisions="ascendnpu_ir=$compiler_revision
llvm_project=$llvm_revision"
actual_build_revisions=""
if [[ -f "$build_revision_stamp" ]]; then
  actual_build_revisions="$(cat "$build_revision_stamp")"
fi
if [[ -f "$REMOTE_COMPILER_BUILD/CMakeCache.txt" \
  && "$actual_build_revisions" != "$expected_build_revisions" ]]; then
  printf 'BISHENGIR_BUILD_CACHE_RESET old=%s new=%s\n' \
    "${actual_build_revisions//$'\n'/,}" \
    "${expected_build_revisions//$'\n'/,}"
  cmake --build "$REMOTE_COMPILER_BUILD" --target clean
fi

soc_name="$("$REMOTE_VENV/bin/python" - <<'PY'
import torch
import torch_npu  # noqa: F401

print(torch.npu.get_device_name(torch.npu.current_device()))
PY
)"
case "$soc_name" in
  *Ascend910_95*|*Ascend950*|*910_958*)
    bitcode_arch=c310
    hivmc_a5_path="$(command -v hivmc-a5 || true)"
    if [[ -z "$hivmc_a5_path" ]]; then
      printf '%s\n' 'hivmc-a5 was not found in the active CANN environment.' >&2
      exit 1
    fi
    hivmc_a5_path="$(realpath "$hivmc_a5_path")"
    BISHENG_INSTALL_PATH="$(dirname "$hivmc_a5_path")"
    export BISHENG_INSTALL_PATH
    hivmc_a5_version="$("$hivmc_a5_path" --version 2>&1 | head -n 1)"
    printf 'A5_HIVMC_OK path=%s version=%s\n' \
      "$hivmc_a5_path" "$hivmc_a5_version"
    ;;
  *Ascend910B*|*Ascend910_93*) bitcode_arch=c220 ;;
  *)
    printf 'unsupported or unknown SoC for BishengIR bitcode: %s\n' "$soc_name" >&2
    exit 1
    ;;
esac
printf 'BishengIR target: soc=%s bitcode_arch=%s\n' "$soc_name" "$bitcode_arch"

test -x "$REMOTE_HOST_CC" || {
  printf 'missing host C compiler: %s\n' "${REMOTE_HOST_CC:-not found}" >&2
  exit 1
}
test -x "$REMOTE_HOST_CXX" || {
  printf 'missing host C++ compiler: %s\n' "${REMOTE_HOST_CXX:-not found}" >&2
  exit 1
}
test -x "$REMOTE_CCEC" || {
  printf 'missing CANN ccec: %s\n' "$REMOTE_CCEC" >&2
  exit 1
}
test -x "$REMOTE_LLVM_LINK" || {
  printf 'missing CANN llvm-link: %s\n' "$REMOTE_LLVM_LINK" >&2
  exit 1
}

mkdir -p "$build_lib" "$toolchain_bin"
ln -sfn "$REMOTE_CCEC" "$toolchain_bin/ccec"
ln -sfn "$REMOTE_LLVM_LINK" "$toolchain_bin/llvm-link"

for name in meta_op.aic.$bitcode_arch.bc meta_op.aiv.$bitcode_arch.bc \
  meta_op.mix.aic.$bitcode_arch.bc meta_op.mix.aiv.$bitcode_arch.bc host.bc; do
  if [[ -L "$build_lib/$name" ]]; then
    unlink "$build_lib/$name"
  fi
done

cmake -S "$llvm_source" -B "$REMOTE_COMPILER_BUILD" -G Ninja \
  -DCMAKE_C_COMPILER="$REMOTE_HOST_CC" \
  -DCMAKE_CXX_COMPILER="$REMOTE_HOST_CXX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=bishengir \
  -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$compiler_source" \
  -DLLVM_BSPUB_DAVINCI_BISHENGIR=ON \
  -DBSPUB_DAVINCI_BISHENGIR=ON \
  -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON \
  -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON \
  -DBISHENGIR_BUILD_TEMPLATE=ON \
  -DBISHENG_COMPILER_PATH="$toolchain_bin" \
  -DBISHENGIR_ENABLE_TRITON_COMPILE=ON

cmake --build "$REMOTE_COMPILER_BUILD" \
  --target bishengir-compile bishengir-opt -j "$jobs"

generated_hivm_enums="$(find "$REMOTE_COMPILER_BUILD" \
  -type f -name HIVMEnums.cpp.inc -print -quit)"
if [[ -z "$generated_hivm_enums" ]] \
  || ! grep -q 'ssbuf' "$generated_hivm_enums"; then
  printf 'generated HIVM enum schema does not contain SSBUF: %s\n' \
    "${generated_hivm_enums:-not found}" >&2
  exit 1
fi
printf 'HIVM_TABLEGEN_SSBUF_OK file=%s\n' "$generated_hivm_enums"

custom_bishengir_opt="$REMOTE_COMPILER_BUILD/bin/bishengir-opt"
test -x "$custom_bishengir_opt" || {
  printf 'custom bishengir-opt not found: %s\n' "$custom_bishengir_opt" >&2
  exit 1
}
custom_bishengir_compile="$REMOTE_COMPILER_BUILD/bin/bishengir-compile"
test -x "$custom_bishengir_compile" || {
  printf 'custom bishengir-compile not found: %s\n' \
    "$custom_bishengir_compile" >&2
  exit 1
}

# Commit the source identity as soon as both compiler targets and their
# generated schema are known-good. A later packaging or compatibility-probe
# failure must not force Ninja to rebuild thousands of unchanged targets on
# the next invocation.
build_revision_stamp_tmp="$build_revision_stamp.tmp.$$"
printf '%s\n' "$expected_build_revisions" >"$build_revision_stamp_tmp"
mv -f -- "$build_revision_stamp_tmp" "$build_revision_stamp"
printf '%s\n' 'BISHENGIR_BUILD_STAMP_UPDATED'

# DynamicCV introduces the SSBUF address space. CANN's older bishengir-opt
# does not know that enum value, so validate the project-built MLIR 19 reader
# against bytecode emitted by this checkout's MLIR 22 writer.
triton_mlir_opt="$REMOTE_PROJECT/python/triton/_C/triton-mlir-opt"
test -x "$triton_mlir_opt" || {
  printf 'project triton-mlir-opt not found: %s\n' "$triton_mlir_opt" >&2
  exit 1
}
ssbuf_check_dir="$(mktemp -d)"
trap 'rm -rf -- "$ssbuf_check_dir"' EXIT
printf '%s\n' \
  'module {' \
  '  func.func @ssbuf_roundtrip(%arg0: memref<16xf16, #hivm.address_space<ssbuf>>) {' \
  '    return' \
  '  }' \
  '}' >"$ssbuf_check_dir/input.mlir"
"$triton_mlir_opt" "$ssbuf_check_dir/input.mlir" --emit-bytecode \
  -o "$ssbuf_check_dir/input.mlirbc"
"$custom_bishengir_opt" "$ssbuf_check_dir/input.mlirbc" \
  -o "$ssbuf_check_dir/roundtrip.mlir"
grep -q '#hivm.address_space<ssbuf>' "$ssbuf_check_dir/roundtrip.mlir"
printf '%s\n' 'HIVM_SSBUF_BYTECODE_ROUNDTRIP_OK'
rm -rf -- "$ssbuf_check_dir"
trap - EXIT

for stem in aic aiv mix.aic mix.aiv; do
  target_bc="$build_lib/meta_op.$stem.$bitcode_arch.bc"
  if [[ ! -e "$target_bc" ]]; then
    source_bc="$system_bc_lib/meta_op.$stem.$bitcode_arch.bc"
    if [[ ! -e "$source_bc" ]]; then
      source_bc="$system_bc_lib/meta_op.$stem.bc"
    fi
    if [[ ! -e "$source_bc" ]]; then
      printf 'missing %s bitcode for meta_op.%s\n' "$bitcode_arch" "$stem" >&2
      exit 1
    fi
    ln -s "$source_bc" "$target_bc"
  fi
  test -s "$target_bc" || {
    printf 'empty %s bitcode: %s\n' "$bitcode_arch" "$target_bc" >&2
    exit 1
  }
done

if [[ ! -e "$build_lib/host.bc" ]]; then
  if [[ ! -e "$system_bc_lib/host.bc" ]]; then
    printf 'missing generated and CANN host.bc: %s\n' "$system_bc_lib/host.bc" >&2
    exit 1
  fi
  ln -s "$system_bc_lib/host.bc" "$build_lib/host.bc"
fi
test -s "$build_lib/host.bc" || {
  printf 'empty host bitcode: %s\n' "$build_lib/host.bc" >&2
  exit 1
}

printf 'BISHENGIR_PACKAGE_OK soc=%s bitcode_arch=%s tools=compile,opt\n' \
  "$soc_name" "$bitcode_arch"
