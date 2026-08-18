#!/usr/bin/env bash

# Probe an already-installed Triton-Ascend wheel with the compiler tools from
# the current environment. All environment changes stay inside main's subshell.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'Execute this script; do not source it.' >&2
  return 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
REPORT_ROOT="$PROJECT_ROOT/tmp"
mkdir -p "$REPORT_ROOT"
REPORT_FILE="$REPORT_ROOT/installed-wheel-toolchain-$(date +%Y%m%dT%H%M%S).log"

main() (
  set -uo pipefail

  local python_bin timeout_bin probe_dir candidate
  local probe_timeout full_timeout intra vf warmup active
  local failures=0

  section() { printf '\n===== %s =====\n' "$1"; }
  item() { printf '%-30s %s\n' "$1:" "${2:-<empty>}"; }
  check() {
    printf 'CHECK %-36s %-5s %s\n' "$1" "$2" "$3"
    [[ "$2" == PASS || "$2" == WARN ]] || failures=$((failures + 1))
  }
  real_path() {
    realpath "$1" 2>/dev/null || readlink -f "$1" 2>/dev/null || printf '%s\n' "$1"
  }
  run_limited() {
    local seconds="$1"
    shift
    if [[ -n "$timeout_bin" ]]; then
      "$timeout_bin" "$seconds" "$@"
    else
      "$@"
    fi
  }
  field() {
    awk -F '\t' -v key="$1" '$1 == key {sub(/^[^\t]*\t/, ""); print; exit}' "$probe_dir/tool-paths.tsv"
  }
  show_tool() {
    local label="$1"
    local path="$2"
    if [[ -z "$path" || ! -x "$path" ]]; then
      item "$label" NOT_FOUND
      return
    fi
    item "$label" "$(real_path "$path")"
    run_limited "$probe_timeout" "$path" --version 2>&1 \
      | sed -n '1,3{s/^/  version: /;p;}' || true
  }
  print_excerpt() {
    local path="$1"
    local pattern="$2"
    grep -Ein -- "$pattern" "$path" 2>/dev/null | tail -40 || true
    printf '%s\n' '-- last 40 lines --'
    tail -40 "$path" 2>/dev/null || true
  }

  python_bin="${SYSTEM_PROBE_PYTHON:-$(command -v python || command -v python3 || true)}"
  timeout_bin="$(command -v timeout || command -v gtimeout || true)"
  probe_timeout="${SYSTEM_PROBE_STEP_TIMEOUT:-30}"
  full_timeout="${SYSTEM_PROBE_FULL_TIMEOUT:-300}"
  intra="${SYSTEM_PROBE_INTRA_CACHE_NUM:-1}"
  vf="${SYSTEM_PROBE_VF_MERGE_LEVEL:-0}"
  warmup="${SYSTEM_PROBE_WARMUP:-1}"
  active="${SYSTEM_PROBE_ACTIVE:-1}"
  candidate="${SYSTEM_PROBE_OPERATOR:-$PROJECT_ROOT/experiment_operators/candidates/fused_attention.py}"
  probe_dir="$(mktemp -d "$REPORT_ROOT/installed-wheel-probe.XXXXXX")"
  trap 'rm -rf -- "${probe_dir:-}"' EXIT

  section "Isolation"
  item project_root "$PROJECT_ROOT"
  item selected_python "${python_bin:-NOT_FOUND}"
  item temporary_environment "subshell only"
  item inherited_device_filter "${ASCEND_RT_VISIBLE_DEVICES-${ASCEND_VISIBLE_DEVICES-<unset>}}"
  item report "$REPORT_FILE"

  if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    check PYTHON_AVAILABLE FAIL 'set SYSTEM_PROBE_PYTHON to the wheel environment Python'
    return 1
  fi

  # Remove development-checkout overrides. Sourcing CANN here affects only the
  # subshell and supplies runtime libraries and system compiler paths.
  unset PYTHONPATH TRITON_BUILD_DIR TRITON_NPU_COMPILER_PATH
  unset BISHENGIR_NATIVE_A5_REGBASE BISHENGIR_LEGACY_A5_REGBASE
  unset TRITON_ASCEND_USE_BYTECODE EXPERIMENT_MULTIBUFFER_NUM
  if ! source "$SCRIPT_DIR/load-cann-environment.sh"; then
    check CANN_ENVIRONMENT FAIL 'CANN set_env.sh was not found or failed'
    return 1
  fi
  unset PYTHONPATH TRITON_BUILD_DIR TRITON_NPU_COMPILER_PATH

  export TMPDIR="$probe_dir/tmp"
  export TMP="$TMPDIR"
  export TEMP="$TMPDIR"
  export TRITON_CACHE_DIR="$probe_dir/triton-cache"
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

  section "Installed Python packages and resolved tools"
  if ! "$python_bin" - "$probe_dir/tool-paths.tsv" <<'PY'
import importlib.metadata
import pathlib
import sys

output = pathlib.Path(sys.argv[1])

try:
    import torch
    import torch_npu  # noqa: F401
    import triton
    import triton._C.libtriton as libtriton
    from triton._C.libtriton import ascend
    from triton._C.libtriton.ascend import ir as ascend_ir  # noqa: F401
    from triton.backends.ascend import utils
except Exception as exc:
    print(f"IMPORT_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)
    raise

def version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "UNKNOWN"

compiler, _ = utils._get_npucompiler_path()
reader, _ = utils._get_bishengir_opt_path()
rows = {
    "PYTHON": sys.executable,
    "PYTHON_PREFIX": sys.prefix,
    "TRITON_FILE": triton.__file__,
    "TRITON_VERSION": version("triton", "triton-ascend"),
    "TORCH_FILE": torch.__file__,
    "TORCH_VERSION": torch.__version__,
    "TORCH_NPU_FILE": torch_npu.__file__,
    "TORCH_NPU_VERSION": version("torch-npu", "torch_npu"),
    "LIBTRITON": libtriton.__file__,
    "ASCEND_UTILS_FILE": utils.__file__,
    "TRITON_MLIR_OPT": utils._get_triton_mlir_opt_path(),
    "TRITON_OPT": utils._get_triton_opt_path(),
    "BISHENGIR_OPT": reader,
    "BISHENGIR_COMPILE": compiler,
    "DYNAMIC_CV_API": str(
        hasattr(ascend.passes.ttir, "add_dynamic_cv_pipeline")
        and hasattr(ascend.passes.ttir, "set_buffer_count")
    ),
}
try:
    rows["DEVICE_COUNT"] = str(torch.npu.device_count())
    rows["DEVICE_NAME"] = torch.npu.get_device_name(torch.npu.current_device())
except Exception as exc:
    rows["DEVICE_COUNT"] = "ERROR"
    rows["DEVICE_NAME"] = f"ERROR: {type(exc).__name__}: {exc}"

with output.open("w", encoding="utf-8") as stream:
    for key, value in rows.items():
        stream.write(f"{key}\t{value}\n")
PY
  then
    check INSTALLED_WHEEL_IMPORT FAIL 'triton/torch_npu or Ascend libtriton import failed'
    return 1
  fi
  check INSTALLED_WHEEL_IMPORT PASS 'wheel and triton._C.libtriton.ascend import successfully'

  local python_prefix triton_file libtriton ascend_utils_file
  local triton_mlir_opt triton_opt bishengir_opt bishengir_compile
  python_prefix="$(field PYTHON_PREFIX)"
  triton_file="$(field TRITON_FILE)"
  libtriton="$(field LIBTRITON)"
  ascend_utils_file="$(field ASCEND_UTILS_FILE)"
  triton_mlir_opt="$(field TRITON_MLIR_OPT)"
  triton_opt="$(field TRITON_OPT)"
  bishengir_opt="$(field BISHENGIR_OPT)"
  bishengir_compile="$(field BISHENGIR_COMPILE)"

  item python "$(field PYTHON)"
  item python_prefix "$python_prefix"
  item triton "${triton_file} ($(field TRITON_VERSION))"
  item torch "$(field TORCH_FILE) ($(field TORCH_VERSION))"
  item torch_npu "$(field TORCH_NPU_FILE) ($(field TORCH_NPU_VERSION))"
  item libtriton "$libtriton"
  item ascend_backend_utils "$ascend_utils_file"
  item dynamic_cv_api "$(field DYNAMIC_CV_API)"
  item device_count "$(field DEVICE_COUNT)"
  item device_name "$(field DEVICE_NAME)"
  show_tool triton-mlir-opt "$triton_mlir_opt"
  show_tool triton-opt "$triton_opt"
  show_tool bishengir-opt "$bishengir_opt"
  show_tool bishengir-compile "$bishengir_compile"
  show_tool hivmc-a5 "$(command -v hivmc-a5 2>/dev/null || true)"
  show_tool hivmc "$(command -v hivmc 2>/dev/null || true)"
  show_tool bisheng "$(command -v bisheng 2>/dev/null || true)"

  local wrong_source=0 path
  for path in "$triton_file" "$libtriton" "$ascend_utils_file" \
    "$triton_mlir_opt" "$triton_opt" "$bishengir_opt" "$bishengir_compile"; do
    case "$(real_path "$path")" in
      "$PROJECT_ROOT/python"/*|\
      "$PROJECT_ROOT/third_party/ascend"/*|\
      "$PROJECT_ROOT/.codex-remote/venv"/*|\
      "$PROJECT_ROOT/.codex-remote/ascendnpu-ir-build-explicit"/*)
        wrong_source=1
        ;;
    esac
  done
  for path in "$triton_file" "$libtriton" "$ascend_utils_file"; do
    case "$(real_path "$path")" in
      "$(real_path "$python_prefix")"/*) ;;
      *) wrong_source=1 ;;
    esac
  done
  if (( wrong_source )); then
    check TOOL_SOURCE FAIL 'selected components resolve to the development checkout or outside the wheel Python prefix'
    printf '%s\n' 'Set SYSTEM_PROBE_PYTHON to the isolated environment containing the installed wheel.'
    return 1
  fi
  check TOOL_SOURCE PASS 'Triton comes from the selected wheel environment; no project development tool was selected'

  if [[ ! -x "$triton_mlir_opt" || ! -x "$bishengir_opt" || ! -x "$bishengir_compile" ]]; then
    check REQUIRED_TOOLS FAIL 'writer, reader, or compiler is missing'
    return 1
  fi
  check REQUIRED_TOOLS PASS 'writer, reader, and compiler are executable'

  if [[ "$(field DYNAMIC_CV_API)" == True ]]; then
    check DYNAMIC_CV_FRONTEND_API PASS 'libtriton exposes DynamicCV and buffer-count APIs'
  else
    check DYNAMIC_CV_FRONTEND_API FAIL 'installed libtriton lacks DynamicCV or buffer-count APIs'
  fi

  local compiler_help_rc=0
  run_limited "$probe_timeout" "$bishengir_compile" --help \
    >"$probe_dir/compiler-help.log" 2>&1 || compiler_help_rc=$?
  if grep -q -- '--enable-vf-merge-level' "$probe_dir/compiler-help.log"; then
    check VF_MERGE_COMPILER_OPTION PASS "compiler help advertises --enable-vf-merge-level"
  else
    check VF_MERGE_COMPILER_OPTION FAIL "returncode=$compiler_help_rc; compiler option is missing"
  fi

  section "MLIR bytecode v4 compatibility"
  printf '%s\n' 'module {' \
    '  llvm.func @bytecode_roundtrip(%arg0: i64) {' \
    '    %0 = llvm.inttoptr %arg0 : i64 to !llvm.ptr' \
    '    llvm.return' '  }' '}' >"$probe_dir/generic.mlir"
  local writer_rc=0 reader_rc=0
  run_limited "$probe_timeout" "$triton_mlir_opt" "$probe_dir/generic.mlir" \
    --emit-bytecode --emit-bytecode-version=4 -o "$probe_dir/generic.mlirbc" \
    >"$probe_dir/generic-writer.log" 2>&1 || writer_rc=$?
  if (( writer_rc == 0 )); then
    run_limited "$probe_timeout" "$bishengir_opt" "$probe_dir/generic.mlirbc" \
      -o "$probe_dir/generic-out.mlir" >"$probe_dir/generic-reader.log" 2>&1 || reader_rc=$?
  else
    reader_rc=125
  fi
  if (( writer_rc == 0 && reader_rc == 0 )) \
    && grep -q llvm.inttoptr "$probe_dir/generic-out.mlir"; then
    check BYTECODE_V4_GENERIC_ROUNDTRIP PASS "writer_rc=$writer_rc reader_rc=$reader_rc"
  else
    check BYTECODE_V4_GENERIC_ROUNDTRIP FAIL "writer_rc=$writer_rc reader_rc=$reader_rc"
    print_excerpt "$probe_dir/generic-reader.log" 'error|failed|bytecode|version'
  fi

  printf '%s\n' 'module {' \
    '  func.func @ssbuf_roundtrip(%arg0: memref<16xf16, #hivm.address_space<ssbuf>>) {' \
    '    return' '  }' '}' >"$probe_dir/ssbuf.mlir"
  local ssbuf_writer_rc=0 ssbuf_reader_rc=0
  run_limited "$probe_timeout" "$triton_mlir_opt" "$probe_dir/ssbuf.mlir" \
    --emit-bytecode --emit-bytecode-version=4 -o "$probe_dir/ssbuf.mlirbc" \
    >"$probe_dir/ssbuf-writer.log" 2>&1 || ssbuf_writer_rc=$?
  if (( ssbuf_writer_rc == 0 )); then
    run_limited "$probe_timeout" "$bishengir_opt" "$probe_dir/ssbuf.mlirbc" \
      -o "$probe_dir/ssbuf-out.mlir" >"$probe_dir/ssbuf-reader.log" 2>&1 || ssbuf_reader_rc=$?
  else
    ssbuf_reader_rc=125
  fi
  if (( ssbuf_writer_rc == 0 && ssbuf_reader_rc == 0 )) \
    && grep -q '#hivm.address_space<ssbuf>' "$probe_dir/ssbuf-out.mlir"; then
    check BYTECODE_V4_SSBUF_ROUNDTRIP PASS "writer_rc=$ssbuf_writer_rc reader_rc=$ssbuf_reader_rc"
  else
    check BYTECODE_V4_SSBUF_ROUNDTRIP FAIL "writer_rc=$ssbuf_writer_rc reader_rc=$ssbuf_reader_rc"
    print_excerpt "$probe_dir/ssbuf-reader.log" 'error|failed|address|ssbuf|bytecode'
  fi

  section "BishengIR SSBUF parser"
  local soc compile_rc=0
  soc="${SYSTEM_PROBE_SOC_NAME:-${TRITON_ASCEND_SOC_NAME:-${ASCEND_SOC_VERSION:-$(field DEVICE_NAME)}}}"
  item target "$soc"
  if [[ ! -f "$probe_dir/ssbuf-out.mlir" || -z "$soc" || "$soc" == ERROR:* ]]; then
    check BISHENGIR_COMPILE_SSBUF_PARSE FAIL 'SSBUF text or SoC name is unavailable'
  else
    run_limited "$probe_timeout" "$bishengir_compile" "$probe_dir/ssbuf-out.mlir" \
      --target="$soc" -o "$probe_dir/compiler-check" \
      >"$probe_dir/compiler-check.log" 2>&1 || compile_rc=$?
    if grep -Eiq 'Failed to parse input|fail to parse HIVM_AddressSpaceAttr|expect.*hivm.*addressSpace' \
      "$probe_dir/compiler-check.log"; then
      check BISHENGIR_COMPILE_SSBUF_PARSE FAIL "returncode=$compile_rc; parser rejected ssbuf"
      print_excerpt "$probe_dir/compiler-check.log" 'error|failed|address|ssbuf'
    elif (( compile_rc == 124 || compile_rc == 126 || compile_rc == 127 )) \
      || grep -Eiq 'error while loading shared libraries|symbol lookup error|command not found|No such file or directory' \
        "$probe_dir/compiler-check.log"; then
      check BISHENGIR_COMPILE_SSBUF_PARSE FAIL "returncode=$compile_rc; compiler did not reach parsing"
      print_excerpt "$probe_dir/compiler-check.log" 'error|failed|library|symbol|timeout|not found'
    else
      check BISHENGIR_COMPILE_SSBUF_PARSE PASS \
        "returncode=$compile_rc; parser accepted ssbuf (a later minimal-IR failure is allowed)"
    fi
  fi

  section "Full DynamicCV compile, correctness, and NPU run"
  item operator "$candidate"
  item dynamic_cv true
  item intra_cache_num "$intra"
  item inter_cache_num 1
  item load_cache_num 1
  item vf_merge_level "$vf"
  item multibuffer_num '<omitted; compiler default remains active>'
  item warmup_active "$warmup/$active"
  item timeout_seconds "$full_timeout"

  if [[ ! -f "$candidate" ]]; then
    check FULL_PIPELINE FAIL "operator file not found: $candidate"
  else
    export EXPERIMENT_DYNAMIC_CV=1
    export EXPERIMENT_INTRA_CACHE_NUM="$intra"
    export EXPERIMENT_VF_MERGE_LEVEL="$vf"
    export EXPERIMENT_WARMUP="$warmup"
    export EXPERIMENT_ACTIVE="$active"
    export ENABLE_PRINT_UB_BITS=true
    export TRITON_ALWAYS_COMPILE=1
    export TRITON_PRINT_AUTOTUNING=1
    export TRITON_PRINT_IR_AFTER_FAILURE=0
    unset EXPERIMENT_MULTIBUFFER_NUM

    local full_rc=0
    (
      cd "$probe_dir" || exit 1
      run_limited "$full_timeout" "$python_bin" -u "$candidate"
    ) >"$probe_dir/full-pipeline.log" 2>&1 || full_rc=$?

    if grep -q -- '--set-local-multibuffer' "$probe_dir/full-pipeline.log"; then
      check MULTIBUFFER_OMITTED FAIL 'full command unexpectedly contains --set-local-multibuffer'
    else
      check MULTIBUFFER_OMITTED PASS 'no custom ordinary multibuffer option was requested or observed'
    fi
    if (( full_rc == 0 )) \
      && grep -q 'Test Passed' "$probe_dir/full-pipeline.log" \
      && grep -q 'BENCHMARK operator=' "$probe_dir/full-pipeline.log"; then
      check FULL_PIPELINE PASS 'DynamicCV compiled, correctness passed, and NPU benchmark returned'
      grep -E 'Test Passed|BENCHMARK operator=|required_ub_bits|memory allocated|UB.*bits' \
        "$probe_dir/full-pipeline.log" | tail -20 || true
    else
      check FULL_PIPELINE FAIL "returncode=$full_rc"
      print_excerpt "$probe_dir/full-pipeline.log" \
        'error|failed|traceback|timeout|address_space|ssbuf|unsupported|mismatch|incorrect'
    fi

    if grep -Eiq 'required_ub_bits[^0-9]*[1-9]|maximum memory allocated size|UB[^0-9]*[1-9][0-9]*.*bits' \
      "$probe_dir/full-pipeline.log"; then
      check UB_OBSERVATION PASS 'nonzero UB information appears in the full-pipeline output'
    else
      check UB_OBSERVATION WARN 'compile/run feasibility can pass, but formal UB measurement still needs verification'
    fi
  fi

  section "Conclusion"
  if (( failures == 0 )); then
    printf '%s\n' 'RESULT=PASS'
    printf '%s\n' 'The installed wheel and its resolved system toolchain can run this DynamicCV case.'
    return 0
  fi
  printf 'RESULT=FAIL failed_checks=%d\n' "$failures"
  printf '%s\n' 'Use the first failed CHECK line to locate the incompatible layer.'
  return 1
)

set +e
main "$@" 2>&1 | tee "$REPORT_FILE"
status=${PIPESTATUS[0]}
set -e
printf '\nSaved report: %s\n' "$REPORT_FILE"
exit "$status"
