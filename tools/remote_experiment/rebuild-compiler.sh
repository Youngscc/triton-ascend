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

soc_name="$("$REMOTE_VENV/bin/python" - <<'PY'
import torch
import torch_npu  # noqa: F401

print(torch.npu.get_device_name(torch.npu.current_device()))
PY
)"
case "$soc_name" in
  *Ascend910_95*|*Ascend950*|*910_958*) bitcode_arch=c310 ;;
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

cmake --build "$REMOTE_COMPILER_BUILD" --target bishengir-compile -j "$jobs"

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

printf 'BISHENGIR_PACKAGE_OK soc=%s bitcode_arch=%s\n' "$soc_name" "$bitcode_arch"
