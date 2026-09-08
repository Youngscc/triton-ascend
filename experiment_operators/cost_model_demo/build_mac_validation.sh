#!/usr/bin/env bash
# Build the repository-pinned compiler pair for host-only UB validation.
set -euo pipefail

model_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$model_dir/../.." && pwd)"
validation_root="${1:?usage: build_mac_validation.sh OUTPUT_DIRECTORY LLVM_PACKAGE [JOBS]}"
llvm_package="${2:?pass the repository-selected prebuilt LLVM package}"
jobs="${3:-6}"
mkdir -p "$validation_root"
validation_root="$(cd "$validation_root" && pwd)"
llvm_package="$(cd "$llvm_package" && pwd)"
cache_root="${COMPILER_BUILD_CACHE_ROOT:-$validation_root}"
mkdir -p "$cache_root"
cache_root="$(cd "$cache_root" && pwd)"
python_bin="${PYTHON_FOR_BUILD:-python3}"
cmake_bin="${CMAKE_FOR_BUILD:-cmake}"
npuir_source="$project_root/third_party/ascend/AscendNPU-IR"

if [[ "$(uname -s)" != Darwin ]]; then
  printf 'This build entry is for macOS.\n' >&2
  exit 1
fi
"$python_bin" -m venv "$cache_root/venv"
"$cache_root/venv/bin/python" -m pip install --disable-pip-version-check \
  'pybind11==3.0.1' 'ninja==1.13.2' 'numpy==2.4.6' 'packaging==26.3' 'filelock==3.32.5'
pybind_dir="$("$cache_root/venv/bin/python" -m pybind11 --cmakedir)"
pybind_include="$("$cache_root/venv/bin/python" -c 'import pybind11; print(pybind11.get_include())')"
python_include="$("$cache_root/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
ninja_bin="$cache_root/venv/bin/ninja"

"$cmake_bin" -S "$npuir_source/third-party/llvm-project/llvm" \
  -B "$cache_root/build-npuir" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$ninja_bin" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_EXTERNAL_PROJECTS=bishengir \
  -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$npuir_source" \
  -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_TESTS=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_BSPUB_DAVINCI_BISHENGIR=ON -DBSPUB_DAVINCI_BISHENGIR=ON \
  -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON \
  -DBISHENGIR_ENABLE_TRITON_COMPILE=ON -DBISHENGIR_BUILD_TEMPLATE=OFF \
  -DLLVM_PARALLEL_LINK_JOBS=1
"$cmake_bin" --build "$cache_root/build-npuir" --target bishengir-compile bishengir-opt -j "$jobs"

# Apply the same checked-in adapters as setup_ascend.py, without resetting files.
for patch_name in triton-ascend-3.6.0.patch triton-ascend-dev-3.6.0.patch; do
  patch_file="$project_root/third_party/ascend/patch/$patch_name"
  if ! git -C "$project_root" apply --reverse --check "$patch_file" 2>/dev/null; then
    git -C "$project_root" apply --check "$patch_file"
    git -C "$project_root" apply "$patch_file"
  fi
done
"$cache_root/venv/bin/python" "$model_dir/build_plan_memory_oracle.py" "$cache_root/build-npuir"

"$cmake_bin" -S "$project_root" -B "$cache_root/build-triton" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$ninja_bin" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_SYSPATH="$llvm_package" -DLLVM_LIBRARY_DIR="$llvm_package/lib" \
  -DLLVM_INCLUDE_DIRS="$llvm_package/include" -DLLVM_DIR="$llvm_package/lib/cmake/llvm" \
  -DMLIR_DIR="$llvm_package/lib/cmake/mlir" -DLLVM_MAJOR_VERSION_22_COMPATIBLE=ON \
  "-DCMAKE_CXX_FLAGS=-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__ -I$pybind_include -I$python_include" \
  -DPython3_EXECUTABLE="$cache_root/venv/bin/python" -Dpybind11_DIR="$pybind_dir" \
  -DTRITON_BUILD_PYTHON_MODULE=ON -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_UT=OFF \
  '-DTRITON_CODEGEN_BACKENDS=nvidia;amd' \
  -DTRITON_PLUGIN_DIRS="$project_root/third_party/ascend" \
  -DTRITON_WHEEL_DIR="$project_root/python/triton" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$project_root/python/triton/_C" \
  -DTRITON_PARALLEL_LINK_JOBS=1
"$cmake_bin" --build "$cache_root/build-triton" --target triton triton-mlir-opt -j "$jobs"

"$cache_root/venv/bin/python" - "$project_root" "$validation_root" "$cache_root" <<'PY'
from pathlib import Path
import hashlib, json, subprocess, sys
root, build, cache = map(Path, sys.argv[1:])
for link, source in {
    root/'python/triton/backends/ascend': root/'third_party/ascend/backend',
    root/'python/triton/language/extra/cann': root/'third_party/ascend/language/cann',
}.items():
    if not link.exists(): link.symlink_to(source, target_is_directory=True)
    if link.resolve() != source.resolve(): raise RuntimeError(f'Unexpected import target: {link}')
revisions={name: subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
    for name,path in {'triton':root,'npuir':root/'third_party/ascend/AscendNPU-IR',
        'llvm':root/'third_party/ascend/AscendNPU-IR/third-party/llvm-project'}.items()}
revisions['build_patch_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'third_party/ascend/patch').glob('triton-ascend*3.6.0.patch')}
revisions['pybind11_version']='3.0.1'
revisions['oracle_source_sha256']=hashlib.sha256((root/'experiment_operators/cost_model_demo/plan_memory_oracle.cpp').read_bytes()).hexdigest()
revisions['compiler_build_cache']=str(cache)
revisions['python']=str(cache/'venv/bin/python')
revisions['upstream_main_revision']=subprocess.check_output(['git','rev-parse','upstream/main'],cwd=root,text=True).strip()
revisions['libtriton_sha256']=hashlib.sha256((root/'python/triton/_C/libtriton.so').read_bytes()).hexdigest()
tool_dir=build/'toolchain/bin'; tool_dir.mkdir(parents=True,exist_ok=True)
for name,source in {**{n:cache/'build-npuir/bin'/n for n in ('bishengir-compile','bishengir-opt','plan-memory-oracle')},
                    'triton-mlir-opt':cache/'build-triton/third_party/ascend/bin/triton-mlir-opt'}.items():
    target=tool_dir/name
    subprocess.run(['/bin/cp','-c',str(source),str(target)],check=True)
revisions['tools']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tool_dir.iterdir()}
subprocess.run(['/bin/cp','-c',str(root/'python/triton/_C/libtriton.so'),str(build/'toolchain/libtriton.so')],check=True)
(build/'build-source.json').write_text(json.dumps(revisions,indent=2)+'\n')
PY
"$cache_root/build-npuir/bin/bishengir-compile" --version
