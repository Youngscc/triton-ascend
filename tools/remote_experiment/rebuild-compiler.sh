#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/remote_experiment/config.sh
source "$SCRIPT_DIR/config.sh"

jobs="${JOBS:-32}"
printf -v container_q '%q' "$REMOTE_CONTAINER"
printf -v build_q '%q' "$REMOTE_COMPILER_BUILD"
printf -v venv_bin_q '%q' "$REMOTE_VENV/bin"
compiler_source="$REMOTE_PROJECT/third_party/ascend/AscendNPU-IR"
llvm_source="$compiler_source/third-party/llvm-project/llvm"
system_bc_lib="$REMOTE_SYSTEM_COMPILER_LIB"
build_lib="$REMOTE_COMPILER_BUILD/lib"
printf -v compiler_source_q '%q' "$compiler_source"
printf -v llvm_source_q '%q' "$llvm_source"
printf -v system_bc_lib_q '%q' "$system_bc_lib"
printf -v build_lib_q '%q' "$build_lib"
printf -v bisheng_compiler_bin_q '%q' "$REMOTE_BISHENG_COMPILER_BIN"

ssh "$REMOTE_HOST" \
  "docker exec $container_q bash -c 'export PATH=$venv_bin_q:\$PATH; \
    mkdir -p $build_lib_q; \
    for name in meta_op.aic.c220.bc meta_op.aiv.c220.bc \
      meta_op.mix.aic.c220.bc meta_op.mix.aiv.c220.bc host.bc; do \
      if [[ -L $build_lib_q/\$name ]]; then unlink $build_lib_q/\$name; fi; \
    done; \
    cmake -S $llvm_source_q -B $build_q -G Ninja \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_EXTERNAL_PROJECTS=bishengir \
      -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR=$compiler_source_q \
      -DBSPUB_DAVINCI_BISHENGIR=ON \
      -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON \
      -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON \
      -DBISHENGIR_BUILD_TEMPLATE=ON \
      -DBISHENG_COMPILER_PATH=$bisheng_compiler_bin_q \
      -DBISHENGIR_ENABLE_TRITON_COMPILE=ON && \
    cmake --build $build_q --target bishengir-compile -j $jobs && \
    mkdir -p $build_lib_q && \
    for stem in aic aiv mix.aic mix.aiv; do \
      target_bc=$build_lib_q/meta_op.\$stem.c220.bc; \
      if [[ ! -e \$target_bc ]]; then \
        source_bc=$system_bc_lib_q/meta_op.\$stem.c220.bc; \
        if [[ ! -e \$source_bc ]]; then \
          source_bc=$system_bc_lib_q/meta_op.\$stem.bc; \
        fi; \
        if [[ ! -e \$source_bc ]]; then \
          echo \"missing generated and CANN bitcode for meta_op.\$stem\" >&2; \
          exit 1; \
        fi; \
        ln -s \$source_bc \$target_bc; \
      fi; \
    done && \
    if [[ ! -e $build_lib_q/host.bc ]]; then \
      if [[ ! -e $system_bc_lib_q/host.bc ]]; then \
        echo \"missing generated and CANN host.bc\" >&2; \
        exit 1; \
      fi; \
      ln -s $system_bc_lib_q/host.bc $build_lib_q/host.bc; \
    fi'"
