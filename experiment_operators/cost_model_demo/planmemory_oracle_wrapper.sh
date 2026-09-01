#!/usr/bin/env bash

set -u

real_compiler=${SHAPE_ORACLE_REAL_COMPILER:?set SHAPE_ORACLE_REAL_COMPILER}

# Queries such as --version retain the compiler's native behavior.
output=""
previous=""
for argument in "$@"; do
  if [[ "$previous" == "-o" ]]; then
    output=$argument
    break
  fi
  previous=$argument
done

if [[ -z "$output" ]]; then
  exec "$real_compiler" "$@"
fi

case_log=${SHAPE_ORACLE_CASE_LOG:-}
stdout_log="${case_log}.stdout"
stderr_log="${case_log}.stderr"
if [[ -n "$case_log" ]]; then
  {
    printf 'COMMAND:'
    printf ' %q' "$real_compiler" "$@" --enable-cpu-runner-after=hivm-plan-memory-regbase
    printf '\n'
  } >"$case_log"
fi

BISHENGIR_STOP_AFTER_PASS_ONLY=1 \
  BISHENGIR_STOP_AFTER_PASS_NAME=hivm-plan-memory-regbase \
  BISHENGIR_STOP_AFTER_PASS_OCCURRENCE=3 \
  "$real_compiler" "$@" --enable-cpu-runner-after=hivm-plan-memory-regbase \
  >"$stdout_log" 2>"$stderr_log"
status=$?

cat "$stdout_log"
cat "$stderr_log" >&2
if [[ -n "$case_log" ]]; then
  {
    printf '\nSTDOUT:\n'
    cat "$stdout_log"
    printf '\nSTDERR:\n'
    cat "$stderr_log"
  } >>"$case_log"
fi
rm -f "$stdout_log" "$stderr_log"

if [[ $status -ne 0 ]]; then
  exit "$status"
fi

# CPU-runner mode writes post-PlanMemory MLIR to the requested output. The
# Triton driver only needs the expected filenames to finish UB metadata parsing.
if [[ -f "$output" ]]; then
  cp "$output" "${output}.o"
  cp "$output" "${output}_reloc.o"
fi
