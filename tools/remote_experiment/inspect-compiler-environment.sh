#!/usr/bin/env bash

# Print the compiler environment and run small MLIR compatibility probes.
# This script never launches an NPU kernel or changes installed software.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=tools/remote_experiment/config.sh
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.sh"

TIMEOUT_BIN="$(command -v timeout || command -v gtimeout || true)"
PROBE_TIMEOUT="${ENV_REPORT_TIMEOUT:-20}"
PROJECT_COMPILER_BUILD="${ENV_REPORT_COMPILER_BUILD:-$PROJECT_ROOT/.codex-remote/ascendnpu-ir-build-explicit}"

section() { printf '\n===== %s =====\n' "$1"; }
item() { printf '%-36s %s\n' "$1:" "${2:-<empty>}"; }
check() { printf 'CHECK %-38s %-5s %s\n' "$1" "$2" "$3"; }

run_limited() {
  if [[ -n "$TIMEOUT_BIN" ]]; then
    "$TIMEOUT_BIN" "$PROBE_TIMEOUT" "$@"
  else
    "$@"
  fi
}

real_path() {
  realpath "$1" 2>/dev/null || readlink -f "$1" 2>/dev/null || printf '%s\n' "$1"
}

file_hash() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" 2>/dev/null | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'
  else
    printf HASH_TOOL_NOT_FOUND
  fi
}

tool_path() {
  local preferred="$1"
  local name="$2"
  if [[ -x "$preferred" ]]; then
    real_path "$preferred"
  else
    command -v "$name" 2>/dev/null || true
  fi
}

show_tool() {
  local name="$1"
  local path
  path="$(command -v "$name" 2>/dev/null || true)"
  if [[ -z "$path" ]]; then
    item "$name" NOT_FOUND
    return
  fi
  item "$name" "$(real_path "$path")"
  run_limited "$path" --version 2>&1 | sed -n '1,3{s/^/  version: /;p;}' || true
}

show_linkage() {
  local name="$1"
  local path="$2"
  [[ -x "$path" ]] || return
  printf -- '-- %s linkage: %s\n' "$name" "$path"
  if command -v ldd >/dev/null 2>&1; then
    ldd "$path" 2>&1 | grep -Ei 'llvm|mlir|bisheng|hivm|not found' | sed -n '1,40p' || \
      printf '%s\n' '  no matching shared-library entries'
  elif command -v otool >/dev/null 2>&1; then
    otool -L "$path" 2>&1 | grep -Ei 'llvm|mlir|bisheng|hivm|not found' | sed -n '1,40p' || \
      printf '%s\n' '  no matching shared-library entries'
  fi
}

git_head() {
  git -C "$1" rev-parse HEAD 2>/dev/null || printf NO_GIT_METADATA
}

section "Report identity"
item generated_at "$(date -Iseconds 2>/dev/null || date)"
item project_root "$PROJECT_ROOT"
item project_compiler_build "$PROJECT_COMPILER_BUILD"
item configured_compiler_build "$REMOTE_COMPILER_BUILD"
item probe_timeout_seconds "$PROBE_TIMEOUT"
item temporary_files "project tmp; removed on exit"

section "Operating system and container"
item hostname "$(hostname 2>/dev/null || true)"
item uname "$(uname -a 2>/dev/null || true)"
item architecture "$(uname -m 2>/dev/null || true)"
item container "$([[ -f /.dockerenv ]] && printf yes || printf no)"
[[ -f /etc/os-release ]] && grep -E '^(PRETTY_NAME|NAME|VERSION|ID|VERSION_ID)=' /etc/os-release || true
command -v ldd >/dev/null 2>&1 && item libc "$(ldd --version 2>&1 | sed -n '1p')"
command -v docker >/dev/null 2>&1 && item docker "$(docker --version 2>&1 | sed -n '1p')"

section "Relevant environment variables"
variables=(
  ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME
  ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES ASCEND_SOC_VERSION
  BISHENG_INSTALL_PATH BISHENGIR_NATIVE_A5_REGBASE BISHENGIR_LEGACY_A5_REGBASE
  LLVM_SYSPATH LLVM_DIR MLIR_DIR LLD_DIR CMAKE_PREFIX_PATH
  TRITON_NPU_COMPILER_PATH TRITON_ASCEND_ARCH TRITON_ASCEND_SOC_NAME
  TRITON_ASCEND_BITCODE_ARCH TRITON_ASCEND_USE_BYTECODE
  PYTHONPATH PATH LD_LIBRARY_PATH
)
for name in "${variables[@]}"; do
  item "$name" "${!name-<unset>}"
done

section "Repository identities"
ASCEND_IR="$PROJECT_ROOT/third_party/ascend/AscendNPU-IR"
ASCEND_LLVM="$ASCEND_IR/third-party/llvm-project"
item triton_ascend_head "$(git_head "$PROJECT_ROOT")"
item ascend_npu_ir_head "$(git_head "$ASCEND_IR")"
item ascend_npu_ir_llvm_head "$(git_head "$ASCEND_LLVM")"
item triton_ascend_gitlink \
  "$(git -C "$PROJECT_ROOT" rev-parse HEAD:third_party/ascend/AscendNPU-IR 2>/dev/null || printf NO_GIT_METADATA)"
item ascend_npu_ir_llvm_gitlink \
  "$(git -C "$ASCEND_IR" rev-parse HEAD:third-party/llvm-project 2>/dev/null || printf NO_GIT_METADATA)"

section "Python packages and resolved modules"
PYTHON_BIN="$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)"
item python_command "${PYTHON_BIN:-NOT_FOUND}"
if [[ -n "$PYTHON_BIN" ]]; then
  run_limited "$PYTHON_BIN" - <<'PY' 2>&1 || true
import hashlib
import importlib
import importlib.metadata
import sys

print(f"python_version: {sys.version.replace(chr(10), ' ')}")
print(f"sys_executable: {sys.executable}")
print(f"sys_prefix: {sys.prefix}")
print(f"sys_base_prefix: {sys.base_prefix}")
for name in ("triton", "torch", "torch_npu", "attrs", "numpy", "pandas", "cmake", "ninja"):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None) or importlib.metadata.version(name)
        print(f"module {name}: version={version} file={getattr(module, '__file__', '<none>')}")
    except Exception as error:
        print(f"module {name}: IMPORT_ERROR {type(error).__name__}: {error}")
try:
    from triton._C import libtriton
    print(f"module triton._C.libtriton: file={libtriton.__file__}")
    digest = hashlib.sha256()
    with open(libtriton.__file__, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    print(f"libtriton_sha256: {digest.hexdigest()}")
    print(f"libtriton_has_getenv: {hasattr(libtriton, 'getenv')}")
except Exception as error:
    print(f"module triton._C.libtriton: IMPORT_ERROR {type(error).__name__}: {error}")
try:
    from triton.backends.ascend import utils
    print(f"ascend_backend_utils: {utils.__file__}")
    print(f"selected_bishengir_compile: {utils._get_npucompiler_path()[0]}")
    print(f"selected_bishengir_opt: {utils._get_bishengir_opt_path()[0]}")
except Exception as error:
    print(f"ascend_backend_resolution: ERROR {type(error).__name__}: {error}")
PY
fi

section "Build and compiler tools"
for name in cmake ninja gcc g++ clang clang++ lld ld.lld triton-opt triton-compile \
  llvm-config llvm-link mlir-opt FileCheck bishengir-opt bishengir-compile \
  hivmc hivmc-a5 ccec npu-smi; do
  show_tool "$name"
done

TRITON_MLIR_OPT="$(tool_path "$PROJECT_ROOT/python/triton/_C/triton-mlir-opt" triton-mlir-opt)"
COMPILER_BIN="${TRITON_NPU_COMPILER_PATH:-$PROJECT_COMPILER_BUILD/bin}"
BISHENGIR_OPT="$(tool_path "$COMPILER_BIN/bishengir-opt" bishengir-opt)"
BISHENGIR_COMPILE="$(tool_path "$COMPILER_BIN/bishengir-compile" bishengir-compile)"

section "LLVM and MLIR 22 writer"
item project_triton_mlir_opt "${TRITON_MLIR_OPT:-NOT_FOUND}"
[[ -n "$TRITON_MLIR_OPT" ]] && item project_triton_mlir_opt_sha256 "$(file_hash "$TRITON_MLIR_OPT")"
[[ -n "$TRITON_MLIR_OPT" ]] && run_limited "$TRITON_MLIR_OPT" --version 2>&1 | sed -n '1,8p' || true
item LLVM_SYSPATH "${LLVM_SYSPATH:-<unset>}"
if [[ -n "${LLVM_SYSPATH:-}" ]]; then
  for file in lib/cmake/llvm/LLVMConfig.cmake lib/cmake/mlir/MLIRConfig.cmake lib/cmake/lld/LLDConfig.cmake; do
    item offline_llvm_file "$LLVM_SYSPATH/$file $([[ -e "$LLVM_SYSPATH/$file" ]] && printf EXISTS || printf MISSING)"
  done
  if [[ -x "$LLVM_SYSPATH/bin/llvm-config" ]]; then
    item offline_llvm_config "$(real_path "$LLVM_SYSPATH/bin/llvm-config")"
    run_limited "$LLVM_SYSPATH/bin/llvm-config" --version 2>&1 | sed -n '1,3p' || true
  fi
  for file in "$LLVM_SYSPATH/lib/cmake/llvm/LLVMConfigVersion.cmake" \
    "$LLVM_SYSPATH/lib/cmake/mlir/MLIRConfigVersion.cmake"; do
    if [[ -f "$file" ]]; then
      printf -- '-- %s version entries\n' "$file"
      grep -E 'PACKAGE_VERSION|LLVM_VERSION_(MAJOR|MINOR|PATCH)' "$file" | sed -n '1,20p' || true
    fi
  done
fi
if [[ -n "$TRITON_MLIR_OPT" ]] \
  && run_limited "$TRITON_MLIR_OPT" --version 2>&1 | grep -Eq 'LLVM version 22|MLIR.*22|22\.0\.0'; then
  check TOP_LEVEL_MLIR_WRITER_22 PASS "project triton-mlir-opt reports LLVM/MLIR 22"
else
  check TOP_LEVEL_MLIR_WRITER_22 WARN "expected project triton-mlir-opt to report LLVM/MLIR 22"
fi

section "LLVM and MLIR 19 BishengIR reader/compiler"
LLVM_VERSION_FILE="$ASCEND_LLVM/cmake/Modules/LLVMVersion.cmake"
item llvm_19_version_file "$LLVM_VERSION_FILE"
[[ -f "$LLVM_VERSION_FILE" ]] && grep -E '^[[:space:]]*set\(LLVM_VERSION_(MAJOR|MINOR|PATCH)' "$LLVM_VERSION_FILE" || true
item project_bishengir_opt "${BISHENGIR_OPT:-NOT_FOUND}"
item project_bishengir_compile "${BISHENGIR_COMPILE:-NOT_FOUND}"
[[ -n "$BISHENGIR_OPT" ]] && item project_bishengir_opt_sha256 "$(file_hash "$BISHENGIR_OPT")"
[[ -n "$BISHENGIR_COMPILE" ]] && item project_bishengir_compile_sha256 "$(file_hash "$BISHENGIR_COMPILE")"
[[ -n "$BISHENGIR_OPT" ]] && run_limited "$BISHENGIR_OPT" --version 2>&1 | sed -n '1,8p' || true
[[ -n "$BISHENGIR_COMPILE" ]] && run_limited "$BISHENGIR_COMPILE" --version 2>&1 | sed -n '1,8p' || true
if [[ -n "$BISHENGIR_OPT" ]] \
  && run_limited "$BISHENGIR_OPT" --version 2>&1 | grep -Eq '19\.1\.7|LLVM version 19|MLIR.*19'; then
  check BISHENGIR_MLIR_READER_19 PASS "project bishengir-opt reports LLVM/MLIR 19"
else
  check BISHENGIR_MLIR_READER_19 WARN "expected repository reader built with LLVM 19.1.7"
fi

if [[ -f "$PROJECT_COMPILER_BUILD/CMakeCache.txt" ]]; then
  printf -- '-- %s selected entries\n' "$PROJECT_COMPILER_BUILD/CMakeCache.txt"
  grep -E '^(CMAKE_HOME_DIRECTORY|LLVM_DIR|MLIR_DIR|LLD_DIR|LLVM_PACKAGE_VERSION|BISHENGIR_ENABLE_TRITON_COMPILE|LLVM_BSPUB_DAVINCI_BISHENGIR_A5)' \
    "$PROJECT_COMPILER_BUILD/CMakeCache.txt" | sed -n '1,80p' || true
else
  item bishengir_cmake_cache "NOT_FOUND at $PROJECT_COMPILER_BUILD/CMakeCache.txt"
fi
if [[ -f "$PROJECT_COMPILER_BUILD/.source-revisions" ]]; then
  printf -- '-- %s\n' "$PROJECT_COMPILER_BUILD/.source-revisions"
  sed -n '1,20p' "$PROJECT_COMPILER_BUILD/.source-revisions"
else
  item bishengir_source_revision_stamp NOT_FOUND
fi

HIVM_ATTRS="$ASCEND_IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
item hivm_source_schema "$HIVM_ATTRS"
if [[ -f "$HIVM_ATTRS" ]] && grep -q 'HIVM_AddressSpace_SSBUF' "$HIVM_ATTRS"; then
  check HIVM_SOURCE_HAS_SSBUF PASS "AscendNPU-IR source declares ssbuf"
else
  check HIVM_SOURCE_HAS_SSBUF FAIL "the checked-out source itself lacks ssbuf"
fi

GENERATED_ENUMS="$(find "$PROJECT_COMPILER_BUILD" -type f -name HIVMEnums.cpp.inc -print -quit 2>/dev/null || true)"
item generated_hivm_enums "${GENERATED_ENUMS:-NOT_FOUND}"
[[ -n "$GENERATED_ENUMS" ]] && item generated_hivm_enums_sha256 "$(file_hash "$GENERATED_ENUMS")"
if [[ -n "$GENERATED_ENUMS" ]] && grep -q ssbuf "$GENERATED_ENUMS"; then
  check HIVM_TABLEGEN_HAS_SSBUF PASS "generated MLIR 19 schema contains ssbuf"
else
  check HIVM_TABLEGEN_HAS_SSBUF FAIL "DynamicCV needs the ssbuf address-space enum"
fi

section "Resolved compiler linkage"
[[ -n "$TRITON_MLIR_OPT" ]] && show_linkage triton-mlir-opt "$TRITON_MLIR_OPT"
[[ -n "$BISHENGIR_OPT" ]] && show_linkage bishengir-opt "$BISHENGIR_OPT"
[[ -n "$BISHENGIR_COMPILE" ]] && show_linkage bishengir-compile "$BISHENGIR_COMPILE"

section "CANN and NPU"
for file in /usr/local/Ascend/cann/version.cfg \
  /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
  /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info \
  /etc/ascend_install.info; do
  if [[ -f "$file" ]]; then
    printf -- '-- %s\n' "$file"
    sed -n '1,20p' "$file"
  fi
done
item REMOTE_SYSTEM_COMPILER_BIN "$REMOTE_SYSTEM_COMPILER_BIN"
item REMOTE_SYSTEM_COMPILER_LIB "$REMOTE_SYSTEM_COMPILER_LIB"
if command -v npu-smi >/dev/null 2>&1; then
  printf -- '-- npu-smi info (first 35 lines)\n'
  run_limited npu-smi info 2>&1 | sed -n '1,35p' || true
fi

section "DynamicCV and MLIR bytecode v4 probes"
item expected_writer "MLIR 22 project triton-mlir-opt"
item expected_reader "MLIR 19.1.7 project bishengir-opt with ssbuf"
item expected_bytecode_version 4
item BISHENGIR_LEGACY_A5_REGBASE "${BISHENGIR_LEGACY_A5_REGBASE-<unset>} (expected unset)"
item BISHENGIR_NATIVE_A5_REGBASE "${BISHENGIR_NATIVE_A5_REGBASE-<unset>} (expected 1 on A5)"
item TRITON_ASCEND_USE_BYTECODE "${TRITON_ASCEND_USE_BYTECODE-<unset>} (expected unset)"

if [[ -z "$TRITON_MLIR_OPT" || -z "$BISHENGIR_OPT" ]]; then
  check BYTECODE_V4_GENERIC_ROUNDTRIP SKIP "writer or reader is missing"
  check BYTECODE_V4_SSBUF_ROUNDTRIP SKIP "writer or reader is missing"
  check BISHENGIR_COMPILE_SSBUF_PARSE SKIP "required tools are missing"
else
  REPORT_TMP_ROOT="${TMPDIR:-$PROJECT_ROOT/tmp}"
  mkdir -p "$REPORT_TMP_ROOT"
  PROBE_DIR="$(mktemp -d "$REPORT_TMP_ROOT/compiler-env-report.XXXXXX")"
  trap 'rm -rf -- "${PROBE_DIR:-}"' EXIT

  printf '%s\n' 'module {' \
    '  llvm.func @bytecode_roundtrip(%arg0: i64) {' \
    '    %0 = llvm.inttoptr %arg0 : i64 to !llvm.ptr' \
    '    llvm.return' '  }' '}' >"$PROBE_DIR/generic.mlir"
  writer_rc=0
  run_limited "$TRITON_MLIR_OPT" "$PROBE_DIR/generic.mlir" --emit-bytecode \
    --emit-bytecode-version=4 -o "$PROBE_DIR/generic.mlirbc" \
    >"$PROBE_DIR/generic-writer.log" 2>&1 || writer_rc=$?
  reader_rc=0
  if (( writer_rc == 0 )); then
    run_limited "$BISHENGIR_OPT" "$PROBE_DIR/generic.mlirbc" -o "$PROBE_DIR/generic-out.mlir" \
      >"$PROBE_DIR/generic-reader.log" 2>&1 || reader_rc=$?
  else
    reader_rc=125
  fi
  if (( writer_rc == 0 && reader_rc == 0 )) && grep -q llvm.inttoptr "$PROBE_DIR/generic-out.mlir"; then
    check BYTECODE_V4_GENERIC_ROUNDTRIP PASS "MLIR 22 writer -> MLIR 19 reader"
  else
    check BYTECODE_V4_GENERIC_ROUNDTRIP FAIL "writer_rc=$writer_rc reader_rc=$reader_rc"
    sed -n '1,30p' "$PROBE_DIR/generic-writer.log" "$PROBE_DIR/generic-reader.log" 2>/dev/null || true
  fi

  printf '%s\n' 'module {' \
    '  func.func @ssbuf_roundtrip(%arg0: memref<16xf16, #hivm.address_space<ssbuf>>) {' \
    '    return' '  }' '}' >"$PROBE_DIR/ssbuf.mlir"
  ssbuf_writer_rc=0
  run_limited "$TRITON_MLIR_OPT" "$PROBE_DIR/ssbuf.mlir" --emit-bytecode \
    --emit-bytecode-version=4 -o "$PROBE_DIR/ssbuf.mlirbc" \
    >"$PROBE_DIR/ssbuf-writer.log" 2>&1 || ssbuf_writer_rc=$?
  ssbuf_reader_rc=0
  if (( ssbuf_writer_rc == 0 )); then
    run_limited "$BISHENGIR_OPT" "$PROBE_DIR/ssbuf.mlirbc" -o "$PROBE_DIR/ssbuf-out.mlir" \
      >"$PROBE_DIR/ssbuf-reader.log" 2>&1 || ssbuf_reader_rc=$?
  else
    ssbuf_reader_rc=125
  fi
  if (( ssbuf_writer_rc == 0 && ssbuf_reader_rc == 0 )) \
    && grep -q '#hivm.address_space<ssbuf>' "$PROBE_DIR/ssbuf-out.mlir"; then
    check BYTECODE_V4_SSBUF_ROUNDTRIP PASS "DynamicCV ssbuf survives MLIR 22 -> 19"
  else
    check BYTECODE_V4_SSBUF_ROUNDTRIP FAIL "writer_rc=$ssbuf_writer_rc reader_rc=$ssbuf_reader_rc"
    sed -n '1,40p' "$PROBE_DIR/ssbuf-writer.log" "$PROBE_DIR/ssbuf-reader.log" 2>/dev/null || true
  fi

  SOC_NAME="${ENV_REPORT_SOC_NAME:-${TRITON_ASCEND_SOC_NAME:-${ASCEND_SOC_VERSION:-}}}"
  item compile_probe_soc "${SOC_NAME:-NOT_DETECTED; set ENV_REPORT_SOC_NAME}"
  if [[ -z "$BISHENGIR_COMPILE" || -z "$SOC_NAME" || ! -f "$PROBE_DIR/ssbuf-out.mlir" ]]; then
    check BISHENGIR_COMPILE_SSBUF_PARSE SKIP "compiler, SoC, or roundtrip text is missing"
  else
    compile_rc=0
    run_limited "$BISHENGIR_COMPILE" "$PROBE_DIR/ssbuf-out.mlir" --target="$SOC_NAME" \
      -o "$PROBE_DIR/compiler-check" >"$PROBE_DIR/compiler.stdout" \
      2>"$PROBE_DIR/compiler.stderr" || compile_rc=$?
    if (( compile_rc == 124 )); then
      check BISHENGIR_COMPILE_SSBUF_PARSE FAIL "returncode=124; compiler probe timed out"
      sed -n '1,60p' "$PROBE_DIR/compiler.stdout" "$PROBE_DIR/compiler.stderr"
    elif grep -Fq '[ERROR] Failed to parse input file:' \
      "$PROBE_DIR/compiler.stdout" "$PROBE_DIR/compiler.stderr"; then
      check BISHENGIR_COMPILE_SSBUF_PARSE FAIL "returncode=$compile_rc; ssbuf parse rejected"
      sed -n '1,60p' "$PROBE_DIR/compiler.stdout" "$PROBE_DIR/compiler.stderr"
    else
      check BISHENGIR_COMPILE_SSBUF_PARSE PASS \
        "returncode=$compile_rc; parse accepted (later pipeline failure is allowed)"
      grep -Ei 'Using merged native A5 regbase pipeline|error:|failed' \
        "$PROBE_DIR/compiler.stdout" "$PROBE_DIR/compiler.stderr" | sed -n '1,30p' || true
    fi
  fi

  rm -rf -- "$PROBE_DIR"
  trap - EXIT
fi

section "Interpretation"
cat <<'EOF'
Expected on the working A5 development environment:
  1. TOP_LEVEL_MLIR_WRITER_22 is PASS.
  2. BISHENGIR_MLIR_READER_19 is PASS and points into the project build.
  3. HIVM_TABLEGEN_HAS_SSBUF is PASS.
  4. Both BYTECODE_V4_*_ROUNDTRIP checks are PASS.
  5. BISHENGIR_COMPILE_SSBUF_PARSE is PASS when a SoC name is available.
  6. BISHENGIR_LEGACY_A5_REGBASE and TRITON_ASCEND_USE_BYTECODE are unset;
     BISHENGIR_NATIVE_A5_REGBASE is 1 on A5.

Compare exact executable paths, real paths, LLVM/MLIR versions, linked
libraries, repository commits, CMake cache entries, and every CHECK line. A
generic roundtrip PASS with an SSBUF roundtrip FAIL isolates the problem to the
DynamicCV HIVM enum/schema rather than bytecode v4 in general.
EOF
