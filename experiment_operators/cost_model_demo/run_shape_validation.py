#!/usr/bin/env python3
"""Validate the DynamicCV UB model against real compiler shape variants.

The script recompiles a Python Triton kernel for every shape and DynamicCV
count, captures the real PlanComputeBlock and DynamicCV pass snapshots, runs
the host-only real PlanMemory oracle, and compares those results with the
parametric model.  No NPU runtime is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_operators.cost_model_demo import (  # noqa: E402
    evaluate_configuration,
    normalized_ir_sha256,
    prepare_cost_model,
)


TARGET = "Ascend950PR_9579"
STATIC_MEMREF_RESULT_RE = re.compile(
    r":\s*memref<((?:[0-9]+x)+(?:bf16|f[0-9]+|i[0-9]+))(?=[,>])")
BLOCK_ID_RE = re.compile(r"ssbuffer\.block_id\s*=\s*([0-9]+)")
RESULT_FIELDS = (
    "case_id",
    "case_class",
    "source_variant",
    "source_sha256",
    "dynamic_cv",
    "status",
    "normalized_plan_sha256",
    "plan_invariant_across_d",
    "model_blockers",
    "model_cuts",
    "model_families",
    "block_rewrite_match",
    "raw_alloc_expected",
    "raw_alloc_compiler",
    "raw_alloc_match",
    "compiler_ub_bits",
    "compiler_ub_bytes",
    "compiler_delta_bytes",
    "model_delta_bytes",
    "delta_match",
    "validation_outcome",
    "compile_time_s",
    "plan_ir_path",
    "pass_log_path",
    "oracle_stage_dir",
    "compiler_log_path",
    "error",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(path: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _extract_stage(log_text: str, pass_name: str) -> str:
    lines = log_text.splitlines()
    starts = [
        index for index, line in enumerate(lines)
        if "IR Dump Before" in line and pass_name in line
    ]
    if not starts:
        raise RuntimeError(f"pass snapshot not found: {pass_name}")
    start = starts[-1] + 1
    end = next(
        (
            index for index in range(start, len(lines))
            if "IR Dump " in lines[index] and "// -----//" in lines[index]
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _alloc_counter(stage_ir: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for line in stage_ir.splitlines():
        if "memref.alloc" not in line:
            continue
        match = STATIC_MEMREF_RESULT_RE.search(line)
        if match:
            result[match.group(1)] += 1
    return result


def _inner_alloc_counter(log_text: str) -> Counter[str]:
    before = _alloc_counter(_extract_stage(log_text, "AddMultiBufferInnerScopePass"))
    after = _alloc_counter(_extract_stage(log_text, "AddMultiBufferOuterScopePass"))
    return after - before


def _family_type(family: Any) -> str:
    return "x".join((*map(str, family.logical_shape), family.dtype))


def _expected_inner_allocs(families: Iterable[Any], dynamic: int) -> Counter[str]:
    expected: Counter[str] = Counter()
    for family in families:
        expected[_family_type(family)] += dynamic
    return expected


def _definition_block(stage_ir: str, value: str) -> Optional[int]:
    definition = re.compile(rf"^\s*{re.escape(value)}(?:\s*:\s*[0-9]+)?\s*=")
    matches = [line for line in stage_ir.splitlines() if definition.search(line)]
    if len(matches) != 1:
        return None
    block = BLOCK_ID_RE.search(matches[0])
    return int(block.group(1)) if block else None


def _block_rewrites_match(log_text: str, families: Iterable[Any]) -> bool:
    stage = _extract_stage(log_text, "BroadcastUBOptPass")
    return all(
        _definition_block(stage, family.origin) == family.producer_block
        for family in families
    )


def _load_corpus(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("shape corpus must use schema_version=1")
    if value.get("operator") != "fused_attention":
        raise ValueError("the first shape oracle supports fused_attention only")
    dynamic_values = value.get("dynamic_values")
    if dynamic_values != [1, 2, 3]:
        raise ValueError("shape corpus dynamic_values must be [1,2,3]")
    fixed = value.get("fixed_controls", {})
    if fixed.get("multibuffer_num") != 1 or fixed.get("vf_merge_level") != 0:
        raise ValueError("shape validation isolates DynamicCV with m=1 and vf=0")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("shape corpus has no cases")
    ids = [case.get("case_id") for case in cases]
    if not all(isinstance(case_id, str) and case_id for case_id in ids):
        raise ValueError("every shape case needs a non-empty case_id")
    if len(ids) != len(set(ids)):
        raise ValueError("shape case_id values must be unique")
    for case in cases:
        if not isinstance(case.get("constants"), dict):
            raise ValueError(f"shape case {case['case_id']} constants must be an object")
        edits = case.get("source_edits", [])
        if not isinstance(edits, list):
            raise ValueError(f"shape case {case['case_id']} source_edits must be a list")
        for index, edit in enumerate(edits):
            if not isinstance(edit, dict):
                raise ValueError(
                    f"shape case {case['case_id']} source_edits[{index}] must be an object")
            if not isinstance(edit.get("old"), str) or not edit["old"]:
                raise ValueError(
                    f"shape case {case['case_id']} source_edits[{index}].old must be non-empty")
            if not isinstance(edit.get("new"), str):
                raise ValueError(
                    f"shape case {case['case_id']} source_edits[{index}].new must be a string")
            if not isinstance(edit.get("count", 1), int) or edit.get("count", 1) <= 0:
                raise ValueError(
                    f"shape case {case['case_id']} source_edits[{index}].count must be positive")
    return value


def _materialize_kernel_source(
    case: dict[str, Any],
    worktree: Path,
    case_dir: Path,
) -> tuple[Path, str]:
    source = (
        worktree / "experiment_operators" / "candidates" / "fused_attention.py"
    ).read_text(encoding="utf-8")
    for index, edit in enumerate(case.get("source_edits", [])):
        old = edit["old"]
        count = edit.get("count", 1)
        actual = source.count(old)
        if actual != count:
            raise ValueError(
                f"shape case {case['case_id']} source edit {index} expected "
                f"{count} exact occurrence(s), found {actual}")
        source = source.replace(old, edit["new"], count)
    source_path = case_dir / "kernel_variant.py"
    source_path.write_text(source, encoding="utf-8")
    return source_path, hashlib.sha256(source.encode()).hexdigest()


def _select_cases(
    corpus: dict[str, Any],
    selected: list[str],
    max_cases: Optional[int],
) -> list[dict[str, Any]]:
    cases = corpus["cases"]
    if selected:
        requested = set(selected)
        known = {case["case_id"] for case in cases}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown shape cases: {', '.join(unknown)}")
        cases = [case for case in cases if case["case_id"] in requested]
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def _ensure_tools(compiler: Path, triton_mlir_opt: Path) -> None:
    required = {
        "bishengir-compile": compiler,
        "bishengir-opt": compiler.parent / "bishengir-opt",
        "triton-mlir-opt": triton_mlir_opt,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing compiler tools: {', '.join(missing)}")
    non_executable = [name for name, path in required.items() if not os.access(path, os.X_OK)]
    if non_executable:
        raise PermissionError(f"compiler tools are not executable: {', '.join(non_executable)}")


def _frontend_case(
    *,
    python: Path,
    worktree: Path,
    output_dir: Path,
    source_path: Path,
    constants_path: Path,
    dynamic: int,
    environment: dict[str, str],
) -> tuple[bool, str]:
    dump_script = worktree / "experiment_operators/plan_compute_block_ir/dump_plan_compute_block.py"
    command = [
        str(python),
        str(dump_script),
        "--worktree",
        str(worktree),
        "--operator",
        "fused_attention",
        "--source-path",
        str(source_path),
        "--dynamic-cv",
        str(dynamic),
        "--constants-json",
        str(constants_path),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(command, text=True, capture_output=True, env=environment)
    (output_dir / "driver.stdout.log").write_text(result.stdout, encoding="utf-8")
    (output_dir / "driver.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        return False, (result.stderr or result.stdout)[-8000:]
    return True, ""


def _compiler_options(dynamic: int):
    from triton.backends.ascend.compiler import NPUOptions

    fields = NPUOptions.__dataclass_fields__
    dynamic_counts = (
        {
            "buf_slot_num_of_veccore": dynamic,
            "buf_slot_num_of_crosscore": 1,
            "buf_slot_num_of_gm": 1,
        }
        if "buf_slot_num_of_veccore" in fields
        else {
            "intra_cache_num": dynamic,
            "inter_cache_num": 1,
            "load_cache_num": 1,
        }
    )
    return NPUOptions(
        arch=TARGET,
        enable_dynamic_cv_pipeline=True,
        cv_pipeline_mode="off",
        set_workspace_multibuffer=0,
        multibuffer=True,
        multibuffer_num=1,
        vf_merge_level=0,
        enable_mixed_cv=True,
        disable_auto_inject_block_sync=True,
        **dynamic_counts,
    )


def _compile_planmemory(
    *,
    ttadapter: str,
    case_id: str,
    dynamic: int,
    log_path: Path,
) -> tuple[str, int, float, str]:
    import triton.backends.ascend.compiler as ascend_compiler
    from triton.backends.compiler import GPUTarget

    options = _compiler_options(dynamic)
    metadata = {**options.__dict__}
    metadata["target"] = GPUTarget("npu", TARGET, 32)
    metadata["hash"] = hashlib.sha256(f"{case_id}:d={dynamic}".encode()).hexdigest()
    os.environ["SHAPE_ORACLE_CASE_LOG"] = str(log_path)
    started = time.perf_counter()
    try:
        ascend_compiler.linalg_to_bin_enable_npu_compile_910_95(ttadapter, metadata, options)
        elapsed = time.perf_counter() - started
        ub_bits = int(metadata.get("required_ub_bits") or 0)
        if ub_bits <= 0:
            return "ub_missing", 0, elapsed, "compiler did not report required_ub_bits"
        return "measured", ub_bits, elapsed, ""
    except Exception as error:  # compiler diagnostics are retained in the case log
        elapsed = time.perf_counter() - started
        return "compile_failed", int(metadata.get("required_ub_bits") or 0), elapsed, str(error)


def _configure_compiler_environment(
    *,
    output_dir: Path,
    compiler: Path,
    triton_mlir_opt: Path,
    wrapper_source: Path,
) -> dict[str, str]:
    wrapper_dir = output_dir / ".compiler-wrapper-bin"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper = wrapper_dir / "bishengir-compile"
    if wrapper.exists() or wrapper.is_symlink():
        wrapper.unlink()
    wrapper.symlink_to(wrapper_source)
    path_parts = (
        str(wrapper_dir),
        str(compiler.parent),
        str(triton_mlir_opt.parent),
        os.environ.get("PATH", ""),
    )
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(path_parts)
    environment["TRITON_NPU_COMPILER_PATH"] = str(compiler.parent)
    environment["SHAPE_ORACLE_REAL_COMPILER"] = str(compiler)
    environment["ENABLE_PRINT_UB_BITS"] = "true"
    environment["TRITON_PRINT_AUTOTUNING"] = "0"
    environment["TRITON_PRINT_IR_AFTER_FAILURE"] = "0"
    environment.setdefault("PYTHONPYCACHEPREFIX", "/tmp/cost-model-shape-pycache")
    os.environ.update({
        key: environment[key]
        for key in (
            "PATH",
            "TRITON_NPU_COMPILER_PATH",
            "SHAPE_ORACLE_REAL_COMPILER",
            "ENABLE_PRINT_UB_BITS",
            "TRITON_PRINT_AUTOTUNING",
            "TRITON_PRINT_IR_AFTER_FAILURE",
        )
    })
    return environment


def _empty_row(case: dict[str, Any], dynamic: int) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "case_class": case.get("class", ""),
        "source_variant": case.get("source_variant", "original"),
        "source_sha256": "",
        "dynamic_cv": dynamic,
        "status": "pending",
        "normalized_plan_sha256": "",
        "plan_invariant_across_d": "",
        "model_blockers": "",
        "model_cuts": "",
        "model_families": "",
        "block_rewrite_match": "",
        "raw_alloc_expected": "",
        "raw_alloc_compiler": "",
        "raw_alloc_match": "",
        "compiler_ub_bits": "",
        "compiler_ub_bytes": "",
        "compiler_delta_bytes": "",
        "model_delta_bytes": "",
        "delta_match": "",
        "validation_outcome": "",
        "compile_time_s": "",
        "plan_ir_path": "",
        "pass_log_path": "",
        "oracle_stage_dir": "",
        "compiler_log_path": "",
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, default=REPO_ROOT)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--triton-mlir-opt", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).with_name("shape_corpus.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Retain only the three pass snapshots used by the oracle.",
    )
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    compiler = args.compiler.resolve()
    triton_mlir_opt = args.triton_mlir_opt.resolve()
    corpus_path = args.corpus.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    _ensure_tools(compiler, triton_mlir_opt)

    wrapper_source = Path(__file__).with_name("planmemory_oracle_wrapper.sh").resolve()
    if not os.access(wrapper_source, os.X_OK):
        raise PermissionError(f"oracle wrapper is not executable: {wrapper_source}")
    environment = _configure_compiler_environment(
        output_dir=output_dir,
        compiler=compiler,
        triton_mlir_opt=triton_mlir_opt,
        wrapper_source=wrapper_source,
    )

    import triton.backends.ascend.compiler as ascend_compiler

    class HostOnlyNPUUtils:
        @staticmethod
        def has_device_limit() -> bool:
            return False

    ascend_compiler.NPUUtils = HostOnlyNPUUtils

    corpus = _load_corpus(corpus_path)
    cases = _select_cases(corpus, args.case, args.max_cases)
    dynamic_values = corpus["dynamic_values"]
    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        print(f"[{case_index}/{len(cases)}] shape={case_id}", flush=True)
        case_dir = output_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        constants_path = case_dir / "constants.json"
        constants_path.write_text(
            json.dumps(case["constants"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            source_path, source_sha256 = _materialize_kernel_source(
                case, worktree, case_dir)
        except Exception as error:
            for dynamic in dynamic_values:
                row = _empty_row(case, dynamic)
                row["status"] = "source_variant_failed"
                row["error"] = str(error).replace("\n", " ")[:4000]
                rows.append(row)
            _write_rows(output_dir / "shape_validation.csv", rows)
            case_summaries.append({
                "case_id": case_id,
                "class": case.get("class", ""),
                "description": case.get("description", ""),
                "source_variant": case.get("source_variant", "original"),
                "all_dynamic_values_measured": False,
                "all_block_rewrites_match": False,
                "all_raw_allocs_match": False,
                "exact_prediction": False,
                "correct_fail_open": False,
                "validation_outcome": "mismatch",
                "validation_pass": False,
                "error": str(error),
            })
            continue

        artifacts: dict[int, dict[str, Any]] = {}
        case_rows: dict[int, dict[str, Any]] = {}
        for dynamic in dynamic_values:
            row = _empty_row(case, dynamic)
            row["source_sha256"] = source_sha256
            rows.append(row)
            case_rows[dynamic] = row
            dynamic_dir = case_dir / f"dynamic_{dynamic}"
            dynamic_dir.mkdir(exist_ok=True)
            print(f"  frontend d={dynamic}", flush=True)
            ok, error = _frontend_case(
                python=Path(sys.executable),
                worktree=worktree,
                output_dir=dynamic_dir,
                source_path=source_path,
                constants_path=constants_path,
                dynamic=dynamic,
                environment=environment,
            )
            if not ok:
                row["status"] = "frontend_failed"
                row["error"] = error.replace("\n", " ")[:4000]
                _write_rows(output_dir / "shape_validation.csv", rows)
                continue
            plan_path = dynamic_dir / "after-plan-compute-block.mlir"
            pass_log_path = dynamic_dir / "mlir-pass-dump.log"
            adapter_path = dynamic_dir / "final.ttadapter.mlir"
            plan_text = plan_path.read_text(encoding="utf-8")
            artifacts[dynamic] = {
                "plan_path": plan_path,
                "pass_log_path": pass_log_path,
                "adapter_path": adapter_path,
                "plan_text": plan_text,
                "normalized_hash": normalized_ir_sha256(plan_text),
            }
            row["normalized_plan_sha256"] = artifacts[dynamic]["normalized_hash"]
            row["plan_ir_path"] = str(plan_path.relative_to(output_dir))
            row["pass_log_path"] = str(pass_log_path.relative_to(output_dir))

        normalized_hashes = {
            artifact["normalized_hash"] for artifact in artifacts.values()
        }
        plan_invariant = len(artifacts) == len(dynamic_values) and len(normalized_hashes) == 1
        for row in case_rows.values():
            row["plan_invariant_across_d"] = str(plan_invariant).lower()

        prepared = None
        if 1 in artifacts:
            try:
                prepared = prepare_cost_model(artifacts[1]["plan_text"], vf_merge_level=0)
            except Exception as error:
                for row in case_rows.values():
                    if row["status"] == "pending":
                        row["status"] = "model_failed"
                        row["error"] = str(error).replace("\n", " ")[:4000]

        compiler_baseline_bytes: Optional[int] = None
        measured_ub: dict[int, int] = {}
        for dynamic in dynamic_values:
            row = case_rows[dynamic]
            if dynamic not in artifacts or prepared is None:
                continue
            artifact = artifacts[dynamic]
            blockers = list(prepared.blockers)
            cuts = [
                {
                    "seed": cut.seed_value,
                    "frontier": [cut.origin_frontier_bytes, cut.selected_frontier_bytes],
                    "moved": list(cut.moved_values),
                    "families": list(cut.family_values),
                }
                for cut in prepared.dynamic_cuts
            ]
            families = [
                {
                    "origin": family.origin,
                    "logical_type": _family_type(family),
                    "physical_shape": list(family.physical_shape),
                    "aligned_bytes": family.aligned_bytes,
                    "producer_block": family.producer_block,
                    "consumer_blocks": list(family.consumer_blocks),
                }
                for family in prepared.dynamic_buffers
            ]
            row["model_blockers"] = _json_cell(blockers)
            row["model_cuts"] = _json_cell(cuts)
            row["model_families"] = _json_cell(families)

            try:
                pass_log = artifact["pass_log_path"].read_text(
                    encoding="utf-8", errors="replace")
                oracle_stage_dir = artifact["pass_log_path"].parent / "oracle-stages"
                oracle_stage_dir.mkdir(exist_ok=True)
                stage_snapshots = {
                    "before-broadcast-ub-opt.mlir": _extract_stage(
                        pass_log, "BroadcastUBOptPass"),
                    "before-add-inner-scope.mlir": _extract_stage(
                        pass_log, "AddMultiBufferInnerScopePass"),
                    "before-add-outer-scope.mlir": _extract_stage(
                        pass_log, "AddMultiBufferOuterScopePass"),
                }
                for filename, stage_text in stage_snapshots.items():
                    (oracle_stage_dir / filename).write_text(
                        stage_text.rstrip() + "\n", encoding="utf-8")
                row["oracle_stage_dir"] = str(oracle_stage_dir.relative_to(output_dir))
                block_match = _block_rewrites_match(pass_log, prepared.dynamic_buffers)
                compiler_allocs = _inner_alloc_counter(pass_log)
                expected_allocs = _expected_inner_allocs(prepared.dynamic_buffers, dynamic)
                alloc_match = compiler_allocs == expected_allocs
                row["block_rewrite_match"] = str(block_match).lower()
                row["raw_alloc_expected"] = _json_cell(dict(expected_allocs))
                row["raw_alloc_compiler"] = _json_cell(dict(compiler_allocs))
                row["raw_alloc_match"] = str(alloc_match).lower()
                if args.compact:
                    artifact["pass_log_path"].unlink()
                    row["pass_log_path"] = ""
            except Exception as error:
                block_match = False
                alloc_match = False
                row["error"] = f"intermediate oracle: {error}"[:4000]

            estimate = evaluate_configuration(
                prepared,
                intra_cache_num=dynamic,
                multibuffer_num=1,
                baseline=None,
            )
            model_delta = estimate.total_from_11_bytes
            row["model_delta_bytes"] = "" if model_delta is None else model_delta
            compiler_log = output_dir / "logs" / f"{case_id}__dynamic-{dynamic}.log"
            row["compiler_log_path"] = str(compiler_log.relative_to(output_dir))
            print(f"  PlanMemory d={dynamic}", flush=True)
            status, ub_bits, elapsed, error = _compile_planmemory(
                ttadapter=artifact["adapter_path"].read_text(encoding="utf-8"),
                case_id=case_id,
                dynamic=dynamic,
                log_path=compiler_log,
            )
            row["status"] = status
            row["compile_time_s"] = f"{elapsed:.6f}"
            if error:
                existing = f"{row['error']} | " if row["error"] else ""
                row["error"] = (existing + error.replace("\n", " "))[:4000]
            if ub_bits:
                measured_ub[dynamic] = ub_bits
                row["compiler_ub_bits"] = ub_bits
                if ub_bits % 8 == 0:
                    row["compiler_ub_bytes"] = ub_bits // 8
            if dynamic == 1 and status == "measured" and ub_bits % 8 == 0:
                compiler_baseline_bytes = ub_bits // 8
            if status == "measured" and compiler_baseline_bytes is not None and ub_bits % 8 == 0:
                compiler_delta = ub_bits // 8 - compiler_baseline_bytes
                row["compiler_delta_bytes"] = compiler_delta
                delta_match = (
                    plan_invariant
                    and not blockers
                    and block_match
                    and alloc_match
                    and model_delta == compiler_delta
                )
                row["delta_match"] = str(delta_match).lower()
            _write_rows(output_dir / "shape_validation.csv", rows)

        compiler_deltas = (
            {
                str(dynamic): measured_ub[dynamic] // 8 - measured_ub[1] // 8
                for dynamic in dynamic_values
                if 1 in measured_ub and dynamic in measured_ub
                and measured_ub[dynamic] % 8 == 0 and measured_ub[1] % 8 == 0
            }
        )
        model_deltas = {
            str(dynamic): case_rows[dynamic]["model_delta_bytes"]
            for dynamic in dynamic_values
            if case_rows[dynamic]["model_delta_bytes"] != ""
        }
        successful_rows = [
            case_rows[dynamic] for dynamic in dynamic_values
            if case_rows[dynamic]["status"] == "measured"
        ]
        model_blockers = list(prepared.blockers) if prepared else ["model unavailable"]
        predictive_eligible = plan_invariant and prepared is not None and not model_blockers
        exact_prediction = (
            predictive_eligible
            and len(successful_rows) == len(dynamic_values)
            and all(row["block_rewrite_match"] == "true" for row in successful_rows)
            and all(row["raw_alloc_match"] == "true" for row in successful_rows)
            and all(row["delta_match"] == "true" for row in successful_rows)
        )
        correct_fail_open = (
            prepared is not None
            and (bool(model_blockers) or not plan_invariant)
            and len(successful_rows) == len(dynamic_values)
            and all(row["model_delta_bytes"] == "" for row in successful_rows)
        )
        if exact_prediction:
            validation_outcome = "exact_prediction"
        elif correct_fail_open:
            validation_outcome = "correct_fail_open"
        else:
            validation_outcome = "mismatch"
        for row in case_rows.values():
            row["validation_outcome"] = validation_outcome
        case_summary = {
            "case_id": case_id,
            "class": case.get("class", ""),
            "description": case.get("description", ""),
            "source_variant": case.get("source_variant", "original"),
            "source_sha256": source_sha256,
            "source_path": str(source_path.relative_to(output_dir)),
            "constant_overrides": case["constants"],
            "normalized_plan_hashes": sorted(normalized_hashes),
            "plan_invariant_across_d": plan_invariant,
            "model_blockers": model_blockers,
            "family_count": len(prepared.dynamic_buffers) if prepared else None,
            "family_bytes": (
                sum(family.aligned_bytes for family in prepared.dynamic_buffers)
                if prepared else None
            ),
            "compiler_delta_bytes": compiler_deltas,
            "model_delta_bytes": model_deltas,
            "all_dynamic_values_measured": len(successful_rows) == len(dynamic_values),
            "all_block_rewrites_match": (
                len(successful_rows) == len(dynamic_values)
                and all(row["block_rewrite_match"] == "true" for row in successful_rows)
            ),
            "all_raw_allocs_match": (
                len(successful_rows) == len(dynamic_values)
                and all(row["raw_alloc_match"] == "true" for row in successful_rows)
            ),
            "predictive_eligible": predictive_eligible,
            "exact_prediction": exact_prediction,
            "correct_fail_open": correct_fail_open,
            "validation_outcome": validation_outcome,
            "validation_pass": exact_prediction or correct_fail_open,
        }
        case_summaries.append(case_summary)
        (output_dir / "shape_validation.json").write_text(
            json.dumps({"cases": case_summaries}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    compiler_version = subprocess.run(
        [str(compiler), "--version"], text=True, capture_output=True)
    import triton
    from triton._C import libtriton

    runtime_source_root = Path(triton.__file__).resolve().parents[2]
    compiler_source_root = compiler.parents[2]
    overall = {
        "schema_version": 1,
        "target": TARGET,
        "operator": corpus["operator"],
        "fixed_controls": corpus["fixed_controls"],
        "dynamic_values": dynamic_values,
        "case_count": len(cases),
        "row_count": len(rows),
        "compiler": str(compiler),
        "compiler_version": (compiler_version.stdout + compiler_version.stderr).strip(),
        "compiler_source_root": str(compiler_source_root),
        "compiler_ascend_npu_ir_revision": _git_revision(compiler_source_root),
        "compiler_llvm_revision": _git_revision(
            compiler_source_root / "third-party/llvm-project"),
        "triton_mlir_opt": str(triton_mlir_opt),
        "kernel_source_worktree": str(worktree),
        "kernel_source_worktree_revision": _git_revision(worktree),
        "kernel_source_ascend_npu_ir_revision": _git_revision(
            worktree / "third_party/ascend/AscendNPU-IR"),
        "runtime_python": sys.executable,
        "runtime_triton": triton.__file__,
        "runtime_libtriton": libtriton.__file__,
        "runtime_python_source_root": str(runtime_source_root),
        "runtime_python_source_revision": _git_revision(runtime_source_root),
        "corpus": str(corpus_path),
        "corpus_sha256": _sha256_file(corpus_path),
        "cases": case_summaries,
        "summary": {
            "all_dynamic_values_measured": sum(
                case["all_dynamic_values_measured"] for case in case_summaries),
            "all_block_rewrites_match": sum(
                case["all_block_rewrites_match"] for case in case_summaries),
            "all_raw_allocs_match": sum(case["all_raw_allocs_match"] for case in case_summaries),
            "exact_predictions": sum(case["exact_prediction"] for case in case_summaries),
            "correct_fail_open": sum(case["correct_fail_open"] for case in case_summaries),
            "validation_pass": sum(case["validation_pass"] for case in case_summaries),
        },
    }
    (output_dir / "shape_validation.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_rows(output_dir / "shape_validation.csv", rows)
    print(json.dumps(overall["summary"], sort_keys=True), flush=True)

    complete = all(case["validation_pass"] for case in case_summaries)
    return 0 if complete or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
