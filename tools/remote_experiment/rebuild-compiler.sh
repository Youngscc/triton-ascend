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
toolchain_bin="$REMOTE_COMPILER_BUILD/a3-toolchain-bin"

test -x "$REMOTE_HOST_CLANG" || {
  printf 'missing host compiler: %s\n' "$REMOTE_HOST_CLANG" >&2
  exit 1
}
test -x "$REMOTE_HOST_CLANGXX" || {
  printf 'missing host compiler: %s\n' "$REMOTE_HOST_CLANGXX" >&2
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

for name in meta_op.aic.c220.bc meta_op.aiv.c220.bc \
  meta_op.mix.aic.c220.bc meta_op.mix.aiv.c220.bc host.bc; do
  if [[ -L "$build_lib/$name" ]]; then
    unlink "$build_lib/$name"
  fi
done

cmake -S "$llvm_source" -B "$REMOTE_COMPILER_BUILD" -G Ninja \
  -DCMAKE_C_COMPILER="$REMOTE_HOST_CLANG" \
  -DCMAKE_CXX_COMPILER="$REMOTE_HOST_CLANGXX" \
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
  target_bc="$build_lib/meta_op.$stem.c220.bc"
  if [[ ! -e "$target_bc" ]]; then
    source_bc="$system_bc_lib/meta_op.$stem.c220.bc"
    if [[ ! -e "$source_bc" ]]; then
      source_bc="$system_bc_lib/meta_op.$stem.bc"
    fi
    if [[ ! -e "$source_bc" ]]; then
      printf 'missing generated and CANN bitcode for meta_op.%s\n' "$stem" >&2
      exit 1
    fi
    ln -s "$source_bc" "$target_bc"
  fi
done

if [[ ! -e "$build_lib/host.bc" ]]; then
  if [[ ! -e "$system_bc_lib/host.bc" ]]; then
    printf 'missing generated and CANN host.bc: %s\n' "$system_bc_lib/host.bc" >&2
    exit 1
  fi
  ln -s "$system_bc_lib/host.bc" "$build_lib/host.bc"
fi
