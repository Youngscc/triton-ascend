#!/usr/bin/env python3
"""Run small A5 controls and print only hand-reportable mismatch summaries."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CASE_MATRIX = {
    "fused": (("F-DYN", True, None), ("F-DEF", False, None),
              ("F-D4", False, 4), ("F-D3", False, 3),
              ("F-D2", False, 2), ("F-D1", False, 1)),
    "hstu": (("H-DYN", True, None), ("H-D2", False, 2),
             ("H-D1", False, 1)),
    "unified": (("U-DYN", True, None), ("U-D4", False, 4),
                 ("U-D3", False, 3)),
}
MODULES = {
    "fused": "experiment_operators.candidates.fused_attention",
    "hstu": "experiment_operators.candidates.hstu_attention",
    "unified": "experiment_operators.candidates.unified_attention",
}
MISMATCH_RE = re.compile(r"DIAG_MISMATCH\s+(.*)")
DOMINANCE_RE = re.compile(r"DIAG_DOMINANCE\s+operand=(\d+)")
IMPORT_ERROR_RE = re.compile(r"DIAG_IMPORT\s+message=(.*)")
MLIR_ERROR_RE = re.compile(r"DIAG_MLIR\s+message=(.*)")
DEV_ENV_READY = "TRITON_ASCEND_DIAG_DEV_ENV_READY"


class DiagnosticMismatch(Exception):
    pass


def activate_and_reexec() -> int:
    activate = ROOT / "tools/remote_experiment/activate-dev-environment.sh"
    command = (
        'source "$1"; '
        f'export {DEV_ENV_READY}=1; '
        'exec python -u "$2" "${@:3}"'
    )
    completed = subprocess.run(
        ["bash", "-c", command, "diagnose-a5", str(activate),
         str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=ROOT,
        env=os.environ.copy(),
        check=False,
    )
    return completed.returncode


def compact_mlir_error(message: str) -> str:
    markers = (
        "error:",
        "failed",
        "doesn't dominate",
        "does not dominate",
        "unknown operation",
        "llvm error",
    )
    lines = [" ".join(line.split()) for line in message.splitlines()]
    useful = [line for line in lines if any(marker in line.lower()
                                              for marker in markers)]
    compact = useful[-1] if useful else " ".join(message.split())
    return compact[-300:]


def install_assert_probe(torch):
    original = torch.testing.assert_close

    def assert_close(actual, expected, *args, **kwargs):
        try:
            return original(actual, expected, *args, **kwargs)
        except AssertionError:
            lhs = actual.detach().to(dtype=torch.float32, device="cpu")
            rhs = expected.detach().to(dtype=torch.float32, device="cpu")
            atol = float(kwargs.get("atol", 1e-5))
            rtol = float(kwargs.get("rtol", 1.3e-6))
            equal_nan = bool(kwargs.get("equal_nan", False))
            bad = ~torch.isclose(lhs, rhs, atol=atol, rtol=rtol,
                                 equal_nan=equal_nan)
            flat = bad.reshape(-1)
            total = flat.numel()
            bounds = [total * index // 4 for index in range(5)]
            chunks = [
                int(flat[bounds[index]:bounds[index + 1]].sum().item())
                for index in range(4)
            ]
            absolute = (lhs - rhs).abs()
            finite_absolute = absolute[torch.isfinite(absolute)]
            max_abs = (float(finite_absolute.max().item())
                       if finite_absolute.numel() else float("inf"))
            print(
                "DIAG_MISMATCH "
                f"count={int(flat.sum().item())} total={total} "
                f"max_abs={max_abs:.6g} "
                f"lhs_zero={int((lhs == 0).sum().item())} "
                f"rhs_zero={int((rhs == 0).sum().item())} "
                f"chunks={','.join(map(str, chunks))}",
                flush=True,
            )
            raise DiagnosticMismatch from None

    torch.testing.assert_close = assert_close


def run_operator_case(operator: str) -> None:
    import numpy as np
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    install_assert_probe(torch)
    module = importlib.import_module(MODULES[operator])

    dynamic = os.environ["DIAG_DYNAMIC_CV"] == "1"
    depth_text = os.environ.get("DIAG_DEPTH")
    options = {
        "enable_dynamic_cv_pipeline": dynamic,
        "multibuffer_num": 1,
        "vf_merge_level": 1,
    }
    if depth_text:
        options["set_workspace_multibuffer"] = int(depth_text)
    module._experiment_compile_options = lambda: dict(options)

    if operator == "fused":
        module.test_attention_fused()
    elif operator == "hstu":
        module.test_hstu_attention_fwd(2, 1024, 2, 32, torch.float32)
    else:
        module.test_triton_unified_attn(
            seq_lens=[(1, 1328), (5, 18), (129, 463)],
            num_heads=(8, 2),
            head_size=128,
            sliding_window=None,
            dtype=torch.float16,
            block_size=32,
            soft_cap=None,
            num_blocks=2048,
            q_dtype=None,
        )
    print("DIAG_PASS", flush=True)


def child_main(operator: str) -> int:
    try:
        run_operator_case(operator)
        return 0
    except DiagnosticMismatch:
        return 20
    except BaseException as error:
        message = str(error)
        if isinstance(error, ImportError):
            compact = " ".join(message.split())[:300]
            print(f"DIAG_IMPORT message={compact}", flush=True)
            return 22
        if type(error).__name__ == "MLIRCompilationError":
            print(f"DIAG_MLIR message={compact_mlir_error(message)}",
                  flush=True)
            return 23
        dominance = re.search(
            r"operand\s+#(\d+)\s+does(?:n't| not)\s+dominate\s+this\s+use",
            message,
            re.IGNORECASE,
        )
        if dominance:
            print(f"DIAG_DOMINANCE operand={dominance.group(1)}", flush=True)
        elif "buildFinalHIVMPipelines" in message:
            print("DIAG_COMPILE stage=buildFinalHIVMPipelines", flush=True)
        else:
            print(f"DIAG_ERROR type={type(error).__name__}", flush=True)
        return 21


def classify(output: str, returncode: int, timed_out: bool) -> str:
    if timed_out:
        return "TIMEOUT"
    if "DIAG_PASS" in output and returncode == 0:
        return "PASS"
    mismatch = MISMATCH_RE.search(output)
    if mismatch:
        return "MISMATCH " + mismatch.group(1).strip()
    dominance = DOMINANCE_RE.search(output)
    if dominance:
        return f"COMPILE_DOMINANCE operand={dominance.group(1)}"
    import_error = IMPORT_ERROR_RE.search(output)
    if import_error:
        return f"IMPORT_ERROR message={import_error.group(1).strip()}"
    mlir_error = MLIR_ERROR_RE.search(output)
    if mlir_error:
        return f"MLIR_ERROR message={mlir_error.group(1).strip()}"
    if "DIAG_COMPILE stage=" in output:
        return output[output.index("DIAG_COMPILE stage="):].splitlines()[0]
    error = re.search(r"DIAG_ERROR\s+type=(\w+)", output)
    return f"ERROR type={error.group(1) if error else 'unknown'}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operator",
        choices=("fused", "hstu", "unified", "all"),
        default="fused",
    )
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child:
        return child_main(args.operator)
    if not args.dry_run and os.environ.get(DEV_ENV_READY) != "1":
        return activate_and_reexec()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    operators = tuple(CASE_MATRIX) if args.operator == "all" else (args.operator,)
    selected = [
        (operator, *case)
        for operator in operators
        for case in CASE_MATRIX[operator]
    ]
    print(
        f"A5_MISMATCH_DIAG cases={len(selected)} "
        "fixed_multibuffer=1 fixed_vf_merge=1"
    )
    if args.dry_run:
        for operator, label, dynamic, depth in selected:
            print(
                f"CASE {label} operator={operator} dynamic={int(dynamic)} "
                f"depth={depth if depth is not None else 'default'}"
            )
        return 0

    diagnostic_root = ROOT / ".codex-remote" / "diagnostics"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix="a5-mismatch-",
                                     dir=diagnostic_root) as temp:
        temp_path = Path(temp)
        for operator, label, dynamic, depth in selected:
            print(
                f"RUN {label} dynamic={int(dynamic)} "
                f"depth={depth if depth is not None else 'default'}",
                flush=True,
            )
            env = os.environ.copy()
            env.update({
                "DIAG_DYNAMIC_CV": "1" if dynamic else "0",
                "TRITON_ALWAYS_COMPILE": "1",
                "TRITON_PRINT_AUTOTUNING": "0",
                "TRITON_PRINT_IR_AFTER_FAILURE": "0",
                "TRITON_CACHE_DIR": str(temp_path / label),
            })
            if depth is None:
                env.pop("DIAG_DEPTH", None)
            else:
                env["DIAG_DEPTH"] = str(depth)
            try:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()),
                     "--operator", operator, "--child"],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                output = completed.stdout or ""
                result = classify(output, completed.returncode, False)
            except subprocess.TimeoutExpired as error:
                output = error.stdout or ""
                result = classify(output, 124, True)
            results[label] = result
            print(f"CASE {label} result={result}", flush=True)

    dynamic_results = [value for key, value in results.items() if key.endswith("DYN")]
    static_results = [value for key, value in results.items() if not key.endswith("DYN")]
    if any(value.startswith("IMPORT_ERROR") for value in results.values()):
        conclusion = "ENVIRONMENT_IMPORT_ERROR"
    elif (results.get("F-DEF") == "PASS"
          and results.get("F-D2", "").startswith("MISMATCH")):
        conclusion = "STATIC_DEFAULT_DIFFERS_FROM_EXPLICIT_DEPTH_2"
    elif (results.get("F-DYN") == "PASS"
          and results.get("F-D4") == "PASS"
          and results.get("F-D1", "").startswith("MISMATCH")
          and results.get("F-D2", "").startswith("MISMATCH")
          and results.get("F-D3", "").startswith("MISMATCH")):
        conclusion = "STATIC_DEPTH_BELOW_4_MISMATCH"
    elif dynamic_results and all(value == "PASS" for value in dynamic_results) \
            and any(value.startswith("MISMATCH") for value in static_results):
        conclusion = "STATIC_CV_REGBASE_TRIGGER"
    elif any(value.startswith("MISMATCH") for value in dynamic_results):
        conclusion = "NOT_LIMITED_TO_STATIC_CV"
    else:
        conclusion = "NEEDS_PASS_LEVEL_COMPARISON"
    print(f"CONCLUSION {conclusion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
