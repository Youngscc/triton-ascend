#!/usr/bin/env python3
"""Run and retain every requested three-axis compiler configuration.

This is deliberately a small controller around the existing operator wrappers.
It does not choose a winner and it never drops a failed configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    "fused_attention": ROOT / "experiment_operators/candidates/fused_attention.py",
    "unified_attention": ROOT / "experiment_operators/candidates/unified_attention.py",
    "hstu_attention": ROOT / "experiment_operators/candidates/hstu_attention.py",
}
PASS_MARKERS = {
    "fused_attention": "Fused Attention Test Passed!",
    "unified_attention": "Unified Attention Test Passed!",
    "hstu_attention": "HSTU Attention Forward Test Passed!",
}
BENCHMARK_RE = re.compile(
    r"BENCHMARK operator=(?P<operator>\S+) latency_ms=(?P<latency>[0-9.]+) "
    r"warmup=(?P<warmup>\d+) active=(?P<active>\d+)"
)
MISMATCHED_ELEMENTS_RE = re.compile(
    r"Mismatched elements:\s*(?P<count>[0-9,]+)\s*/", re.IGNORECASE
)
DOMINANCE_ERROR_RE = re.compile(
    r"operand\s+#(?P<operand>\d+)\s+does(?:n't| not)\s+dominate\s+this\s+use",
    re.IGNORECASE,
)
DEPTH_VALUES = (1, 2, 3, 4)
MULTIBUFFER_VALUES = (1, 2, 3, 4)
VF_MERGE_VALUES = (0, 1, 2)
DEFAULT_VF_MERGE_VALUES = (0, 1)
EXPERIMENT_SCHEMA = "native-cv-depth+no-dynamic-cv+independent-local-multibuffer-v4"
OPERATOR_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SOURCE_BENCHMARK_OPERATOR_RE = re.compile(
    r"BENCHMARK\s+operator=([A-Za-z0-9][A-Za-z0-9_.-]*)"
)
CSV_FIELDNAMES = [
    "operator",
    "depth",
    "multibuffer_num",
    "vf_merge_level",
    "latency_ms",
    "required_ub_kib",
    "required_ub_bytes",
    "required_ub_bits",
    "status",
    "correctness_status",
    "diagnostic",
    "experiment_schema",
    "warmup",
    "active",
    "cache_key",
    "compiler_hash",
    "compile_time_ms",
    "ttir_hash",
    "binary_hash",
    "metadata_path",
    "binary_path",
    "wall_time_s",
    "timed_out",
    "returncode",
    "log_path",
]
SIMPLE_CSV_FIELDNAMES = [
    "序号",
    "depth",
    "multibuffer_num",
    "vf_merge_level",
    "结果",
    "原因",
    "运行延迟_ms",
    "UB使用_KiB",
    "本轮总耗时_s",
]


class SweepProgress:
    """Render a compact sweep dashboard without adding a dependency."""

    BAR_WIDTH = 32

    def __init__(self, operator: str, total: int) -> None:
        self.operator = operator
        self.total = total
        self.completed = 0
        self.current = "waiting"
        self.success = 0
        self.failed = 0
        self.unsupported = 0
        self._rendered = False
        self._owns_stream = False
        self.mode = os.environ.get("SWEEP_PROGRESS_MODE", "auto").lower()
        if self.mode not in {"auto", "terminal", "plain", "off"}:
            raise SystemExit(
                "SWEEP_PROGRESS_MODE must be auto, terminal, plain, or off"
            )
        self.stream, self.interactive = self._select_stream()

    def _select_stream(self) -> tuple[TextIO, bool]:
        if self.mode in {"off", "plain"}:
            return sys.stdout, False
        if sys.stdout.isatty():
            return sys.stdout, True
        try:
            terminal = open("/dev/tty", "w", encoding="utf-8")
        except OSError:
            if self.mode == "terminal":
                raise SystemExit(
                    "SWEEP_PROGRESS_MODE=terminal requires a controlling terminal"
                )
            return sys.stdout, False
        self._owns_stream = True
        return terminal, True

    def _lines(self) -> tuple[str, str]:
        ratio = self.completed / self.total if self.total else 1.0
        filled = min(self.BAR_WIDTH, round(ratio * self.BAR_WIDTH))
        bar = "#" * filled + "-" * (self.BAR_WIDTH - filled)
        progress = (
            f"[{self.operator}] [{bar}] {self.completed}/{self.total} "
            f"({ratio * 100:5.1f}%)"
        )
        details = (
            f"current: {self.current} | success={self.success} "
            f"failed={self.failed} unsupported={self.unsupported}"
        )
        return progress, details

    def _clear_interactive(self) -> None:
        if not self.interactive or not self._rendered:
            return
        self.stream.write("\r\033[2K\033[1A\r\033[2K")
        self.stream.flush()
        self._rendered = False

    def render(self) -> None:
        if self.mode == "off":
            return
        progress, details = self._lines()
        if self.interactive:
            self._clear_interactive()
            self.stream.write(f"{progress}\n{details}")
            self.stream.flush()
            self._rendered = True
        else:
            print(f"PROGRESS {progress}", flush=True)
            print(f"PROGRESS {details}", flush=True)

    def begin(self, key: str, depth: int, multibuffer_num: int, merge: int) -> None:
        self.current = (
            f"{key} depth={depth} multibuffer_num={multibuffer_num} "
            f"vf_merge_level={merge}"
        )
        self.render()

    def finish_candidate(self, key: str, status: str) -> None:
        self.completed += 1
        if status == "measured":
            self.success += 1
            result = "success"
        elif status == "unsupported":
            self.unsupported += 1
            result = "unsupported"
        else:
            self.failed += 1
            result = "failed"
        self.current = f"completed {key} result={result}"
        self.render()

    def log(self, message: str) -> None:
        self._clear_interactive()
        print(message, flush=True)
        if self.interactive:
            self.render()

    def close(self) -> None:
        if self.interactive and self._rendered:
            self.stream.write("\n")
            self.stream.flush()
            self._rendered = False
        if self._owns_stream:
            self.stream.close()


def requested_configs(vf_merge_values):
    for depth in DEPTH_VALUES:
        for multibuffer_num in MULTIBUFFER_VALUES:
            for merge in vf_merge_values:
                yield depth, multibuffer_num, merge


def sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    command = ["git"]
    top_git = ROOT / ".codex-remote/top-git"
    if not (ROOT / ".git").exists() and (top_git / "HEAD").is_file():
        command.extend([f"--git-dir={top_git}", f"--work-tree={ROOT}"])
    result = subprocess.run(
        [*command, *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def matching_metadata(
    cache_dir: Path,
    depth: int,
    multibuffer_num: int,
    merge: int,
    min_mtime_ns: int,
):
    matches = []
    for path in cache_dir.glob("*/**/*.json"):
        if path.name.startswith("__grp__"):
            continue
        try:
            metadata = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        mtime_ns = path.stat().st_mtime_ns
        if mtime_ns < min_mtime_ns:
            continue
        if (
            metadata.get("set_workspace_multibuffer") == depth
            and metadata.get("multibuffer_num") == multibuffer_num
            and metadata.get("vf_merge_level") == merge
            and metadata.get("enable_dynamic_cv_pipeline") is False
            and metadata.get("limit_auto_multi_buffer_buffer") == "no-limit"
        ):
            matches.append((mtime_ns, path, metadata))
    return max(matches, default=(None, None, None), key=lambda item: item[0] or -1)[1:]


def artifact_row(
    cache_dir: Path,
    depth: int,
    multibuffer_num: int,
    merge: int,
    min_mtime_ns: int,
    include_audit: bool = True,
):
    metadata_path, metadata = matching_metadata(
        cache_dir, depth, multibuffer_num, merge, min_mtime_ns
    )
    if metadata_path is None:
        return {
            "metadata_constraints_matched": False,
            "required_ub_bits": None,
            "required_ub_bytes": None,
            "required_ub_kib": None,
            "cache_key": None,
            "compiler_hash": None,
            "compile_time_ms": None,
            "ttir_hash": None,
            "binary_hash": None,
            "metadata_path": None,
            "binary_path": None,
        }

    stem = metadata.get("name") or metadata_path.stem
    artifact_dir = metadata_path.parent
    ttir_path = artifact_dir / f"{stem}.ttir"
    binary_path = artifact_dir / f"{stem}.npubin"
    required_ub_bits = metadata.get("required_ub_bits") or None
    result = {
        "metadata_constraints_matched": True,
        "required_ub_bits": required_ub_bits,
        "required_ub_bytes": required_ub_bits / 8 if required_ub_bits else None,
        "required_ub_kib": required_ub_bits / 8192 if required_ub_bits else None,
    }
    if not include_audit:
        return result
    return {
        **result,
        "cache_key": artifact_dir.name,
        "compiler_hash": metadata.get("hash"),
        "compile_time_ms": metadata.get("compile_time_ms"),
        "ttir_hash": sha256(ttir_path),
        "binary_hash": sha256(binary_path),
        "metadata_path": str(metadata_path),
        "binary_path": str(binary_path) if binary_path.is_file() else None,
    }


def write_tables(rows: list[dict], result_dir: Path):
    jsonl = result_dir / "measurements.jsonl"
    csv_path = result_dir / "measurements.csv"
    with jsonl.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        # The public `depth` axis maps to BishengIR's native
        # set_workspace_multibuffer option and is not duplicated in the CSV.
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def simple_result(status: str) -> str:
    if status == "measured":
        return "成功"
    if status == "unsupported":
        return "不支持"
    return "失败"


def compact_diagnostic(status: str, output: str, default: str) -> str:
    if status == "incorrect":
        mismatch = MISMATCHED_ELEMENTS_RE.search(output)
        if mismatch:
            return f"mismatch={mismatch.group('count').replace(',', '')}"
    if status == "compile_failed":
        dominance = DOMINANCE_ERROR_RE.search(output)
        if dominance:
            return (
                "BuildFinalHIVMPipelines dominance error: "
                f"operand #{dominance.group('operand')}"
            )
    return default


def simple_reason(row: dict) -> str:
    status = row["status"]
    if status == "measured":
        return "编译成功、结果正确，且已记录性能和UB"
    if row["timed_out"]:
        return "执行超时"
    if status == "incorrect":
        if row["diagnostic"].startswith("mismatch="):
            return f"正确性验证失败（{row['diagnostic']}）"
        return "正确性验证失败"
    if status == "compile_failed":
        if row["diagnostic"].startswith("BuildFinalHIVMPipelines dominance error"):
            operand = row["diagnostic"].rsplit(" ", 1)[-1]
            return f"BuildFinalHIVMPipelines产生非法IR（{operand}不支配使用位置）"
        return "编译失败"
    if row["diagnostic"] == "required_ub_bits missing from compiler metadata":
        return "未得到UB使用量"
    if row["diagnostic"] == "compiler metadata missing or does not match fixed experiment options":
        return "未找到与参数匹配的编译结果"
    return "当前配置不支持"


def write_simple_table(rows: list[dict], result_dir: Path) -> Path:
    csv_path = result_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIMPLE_CSV_FIELDNAMES)
        writer.writeheader()
        for number, row in enumerate(rows, 1):
            result = simple_result(row["status"])
            writer.writerow(
                {
                    "序号": number,
                    "depth": row["depth"],
                    "multibuffer_num": row["multibuffer_num"],
                    "vf_merge_level": row["vf_merge_level"],
                    "结果": result,
                    "原因": simple_reason(row),
                    "运行延迟_ms": row["latency_ms"],
                    "UB使用_KiB": row["required_ub_kib"],
                    "本轮总耗时_s": row["wall_time_s"],
                }
            )
    return csv_path


def print_candidate_failure(
    *,
    key: str,
    status: str,
    returncode: int,
    diagnostic: str,
    output: str,
    log_path: Path,
    emit=print,
) -> None:
    """Make failed candidates visible in the foreground sweep output."""
    emit(f"[{key}] FAILED status={status} returncode={returncode}: {diagnostic}")
    if output.strip():
        emit(
            f"----- {key} subprocess output begin -----\n"
            f"{output.rstrip()}\n"
            f"----- {key} subprocess output end -----"
        )
    else:
        emit(f"[{key}] subprocess produced no output")
    emit(f"[{key}] full_log={log_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    operator_group = parser.add_mutually_exclusive_group(required=True)
    operator_group.add_argument(
        "--operator",
        choices=sorted(CANDIDATES),
        help="registered operator name (backward-compatible interface)",
    )
    operator_group.add_argument(
        "--operator-file",
        type=Path,
        help="Python operator wrapper to sweep",
    )
    parser.add_argument(
        "--operator-name",
        help="result name override for --operator-file; normally inferred",
    )
    parser.add_argument(
        "--pass-marker",
        help="optional correctness marker required in operator stdout",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=30)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="maximum seconds allowed for one candidate subprocess",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        help="development smoke-test only; a formal sweep must omit this option",
    )
    parser.add_argument(
        "--simple-output",
        action="store_true",
        help="write one readable results.csv without audit artifacts",
    )
    parser.add_argument(
        "--include-vf-merge-level-2",
        action="store_true",
        help="include the currently disabled vf_merge_level=2 configurations",
    )
    return parser.parse_args()


def normalized_operator_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    if not normalized:
        raise SystemExit(f"cannot derive an operator name from {value!r}")
    if not normalized[0].isalnum():
        normalized = f"operator_{normalized}"
    return normalized


def source_benchmark_operator(candidate: Path) -> str | None:
    try:
        source = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    match = SOURCE_BENCHMARK_OPERATOR_RE.search(source)
    return match.group(1) if match else None


def resolve_operator(args: argparse.Namespace) -> tuple[str, Path, str | None, str]:
    if args.operator:
        candidate = CANDIDATES[args.operator].resolve()
        operator = args.operator
        pass_marker = args.pass_marker or PASS_MARKERS[operator]
        correctness_evidence = f"stdout marker: {pass_marker}"
        return operator, candidate, pass_marker, correctness_evidence

    candidate = args.operator_file.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise SystemExit(f"operator file does not exist: {candidate}")
    if candidate.suffix != ".py":
        raise SystemExit(f"operator file must be a Python file: {candidate}")

    registered_operator = next(
        (
            name
            for name, registered_path in CANDIDATES.items()
            if registered_path.resolve() == candidate
        ),
        None,
    )
    inferred_name = source_benchmark_operator(candidate)
    operator = normalized_operator_name(
        args.operator_name or registered_operator or inferred_name or candidate.stem
    )
    if not OPERATOR_NAME_RE.fullmatch(operator):
        raise SystemExit(f"invalid operator name: {operator!r}")

    pass_marker = args.pass_marker
    if pass_marker is None and registered_operator is not None:
        pass_marker = PASS_MARKERS[registered_operator]
    correctness_evidence = (
        f"stdout marker: {pass_marker}"
        if pass_marker is not None
        else "successful process exit followed by BENCHMARK output"
    )
    return operator, candidate, pass_marker, correctness_evidence


def main() -> int:
    args = parse_args()
    if args.warmup < 1 or args.active < 1 or args.timeout <= 0:
        raise SystemExit("--warmup, --active, and --timeout must be positive")
    operator, candidate, pass_marker, correctness_evidence = resolve_operator(args)

    # The experiment container does not necessarily ship the IANA tzdata
    # database.  A fixed UTC+8 offset gives the desired local run ID without
    # adding a package or changing the shared container environment.
    experiment_timezone = timezone(timedelta(hours=8), name="UTC+08:00")
    run_id = datetime.now(experiment_timezone).strftime("%Y%m%dT%H%M%S%z")
    result_dir = (
        args.output_dir
        or ROOT / ".codex-remote/results" / f"{run_id}-{operator}"
    ).resolve()
    if args.simple_output:
        result_dir.mkdir(parents=True, exist_ok=False)
        logs_dir = None
    else:
        logs_dir = result_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton/cache"))

    vf_merge_values = (
        VF_MERGE_VALUES
        if args.include_vf_merge_level_2
        else DEFAULT_VF_MERGE_VALUES
    )
    configs = list(requested_configs(vf_merge_values))
    requested_configuration_count = (
        len(DEPTH_VALUES) * len(MULTIBUFFER_VALUES) * len(vf_merge_values)
    )
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        configs = configs[: args.limit]

    manifest = {
        "run_id": run_id,
        "timezone": "UTC+08:00",
        "experiment_schema": EXPERIMENT_SCHEMA,
        "operator": operator,
        "candidate": str(candidate),
        "correctness_evidence": correctness_evidence,
        "git_commit": git_value("rev-parse", "HEAD"),
        "ascend_npu_ir_commit": git_value(
            "rev-parse", "HEAD:third_party/ascend/AscendNPU-IR"
        ),
        "git_status": git_value("status", "--short", "--ignore-submodules=dirty") or "",
        "python": sys.version,
        "warmup": args.warmup,
        "active": args.active,
        "timeout_s": args.timeout,
        "benchmark_method": os.environ.get("TRITON_BENCH_METHOD", "npu/default"),
        "requested_configuration_count": requested_configuration_count,
        "executed_configuration_count": len(configs),
        "limited_smoke_run": args.limit is not None,
        "axes": {
            "depth": list(DEPTH_VALUES),
            "multibuffer_num": list(MULTIBUFFER_VALUES),
            "vf_merge_level": list(vf_merge_values),
        },
        "resolved_cv_constraint": "set_workspace_multibuffer == depth",
        "dynamic_cv_pipeline_constraint": "enable_dynamic_cv_pipeline == false",
        "ordinary_multibuffer_constraint": "multibuffer_num is independent of depth",
        "ordinary_multibuffer_strategy": "limit_auto_multi_buffer_buffer == no-limit",
    }
    if not args.simple_output:
        (result_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    rows = []
    progress = SweepProgress(operator, len(configs))
    for index, (depth, multibuffer_num, merge) in enumerate(configs, 1):
        key = f"d{depth}-b{multibuffer_num}-m{merge}"
        progress.begin(key, depth, multibuffer_num, merge)
        log_path = logs_dir / f"{key}.log" if logs_dir is not None else None
        env = os.environ.copy()
        env.update(
            {
                "ENABLE_PRINT_UB_BITS": "true",
                "TRITON_ALWAYS_COMPILE": "1",
                "TRITON_PRINT_AUTOTUNING": "1",
                "EXPERIMENT_DEPTH": str(depth),
                "EXPERIMENT_MULTIBUFFER_NUM": str(multibuffer_num),
                "EXPERIMENT_VF_MERGE_LEVEL": str(merge),
                "EXPERIMENT_WARMUP": str(args.warmup),
                "EXPERIMENT_ACTIVE": str(args.active),
            }
        )
        requested_parameters = {
            "operator": operator,
            "candidate": str(candidate),
            "experiment_schema": EXPERIMENT_SCHEMA,
            "depth": depth,
            "multibuffer_num": multibuffer_num,
            "vf_merge_level": merge,
            "enable_dynamic_cv_pipeline": False,
            "limit_auto_multi_buffer_buffer": "no-limit",
            "enable_print_ub_bits": True,
            "warmup": args.warmup,
            "active": args.active,
            "timeout_s": args.timeout,
        }
        if not args.simple_output:
            progress.log(
                f"[{index}/{len(configs)}] {key} requested_parameters="
                + json.dumps(requested_parameters, sort_keys=True)
            )
        started_epoch_ns = time.time_ns()
        started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(
                [sys.executable, "-u", str(candidate)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
            output = completed.stdout or ""
            returncode = completed.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            returncode = 124
        wall_time = time.monotonic() - started
        if log_path is not None:
            log_header = (
                "[EXPERIMENT] requested_parameters="
                + json.dumps(requested_parameters, sort_keys=True)
                + "\n"
            )
            log_path.write_text(log_header + output)
        audit_lines = tuple(
            dict.fromkeys(
                line
                for line in output.splitlines()
                if line.startswith("[EXPERIMENT] operator_parameters=")
                or line.startswith("[EXPERIMENT] resolved_npu_options=")
                or line.startswith("[DEBUG] cmd_list:")
            )
        )
        if not args.simple_output:
            for line in audit_lines:
                progress.log(f"[{key}] {line}")
        benchmark = BENCHMARK_RE.search(output)
        correctness = (
            pass_marker in output
            if pass_marker is not None
            else returncode == 0 and benchmark is not None
        )
        artifacts = artifact_row(
            cache_dir,
            depth,
            multibuffer_num,
            merge,
            started_epoch_ns - 2_000_000_000,
            include_audit=not args.simple_output,
        )

        diagnostic = ""
        if timed_out:
            status = "unsupported"
            diagnostic = f"candidate timed out after {args.timeout:g} seconds"
        elif returncode != 0:
            status = "incorrect" if "AssertionError" in output else "compile_failed"
            diagnostic = output.strip().splitlines()[-1] if output.strip() else "subprocess failed"
        elif pass_marker is not None and not correctness:
            status = "incorrect"
            diagnostic = "correctness pass marker missing"
        elif benchmark is None:
            status = "compile_failed"
            diagnostic = "benchmark result missing"
        elif not correctness:
            status = "incorrect"
            diagnostic = "successful exit and benchmark output required"
        elif not artifacts["metadata_constraints_matched"]:
            status = "unsupported"
            diagnostic = "compiler metadata missing or does not match fixed experiment options"
        elif artifacts["required_ub_bits"] is None:
            status = "unsupported"
            diagnostic = "required_ub_bits missing from compiler metadata"
        else:
            status = "measured"

        diagnostic = compact_diagnostic(status, output, diagnostic)

        if status != "measured":
            if args.simple_output:
                progress.log(f"{key} 结果={simple_result(status)}")
            else:
                print_candidate_failure(
                    key=key,
                    status=status,
                    returncode=returncode,
                    diagnostic=diagnostic,
                    output=output,
                    log_path=log_path,
                    emit=progress.log,
                )

        row = {
            "operator": operator,
            "experiment_schema": manifest["experiment_schema"],
            "depth": depth,
            "multibuffer_num": multibuffer_num,
            "set_workspace_multibuffer": depth,
            "enable_dynamic_cv_pipeline": False,
            "limit_auto_multi_buffer_buffer": "no-limit",
            "vf_merge_level": merge,
            "status": status,
            "diagnostic": diagnostic,
            "correctness_status": "passed" if correctness else "failed",
            "latency_ms": float(benchmark.group("latency")) if benchmark else None,
            "reported_operator": benchmark.group("operator") if benchmark else None,
            "warmup": int(benchmark.group("warmup")) if benchmark else args.warmup,
            "active": int(benchmark.group("active")) if benchmark else args.active,
            **artifacts,
            "wall_time_s": round(wall_time, 6),
            "timed_out": timed_out,
            "returncode": returncode,
            "log_path": str(log_path) if log_path is not None else None,
        }
        rows.append(row)
        if args.simple_output:
            result_path = write_simple_table(rows, result_dir)
        else:
            write_tables(rows, result_dir)
        progress.finish_candidate(key, status)

    progress.close()

    if args.simple_output:
        result_counts = Counter(simple_result(row["status"]) for row in rows)
        print(
            "实验完成："
            f"成功={result_counts['成功']} "
            f"失败={result_counts['失败']} "
            f"不支持={result_counts['不支持']}"
        )
        print(f"result_file={result_path}")
        return 0 if len(rows) == len(configs) else 1

    status_counts = Counter(row["status"] for row in rows)
    ttir_hashes = sorted({row["ttir_hash"] for row in rows if row["ttir_hash"]})
    binary_hashes = sorted({row["binary_hash"] for row in rows if row["binary_hash"]})
    summary = {
        "row_count": len(rows),
        "expected_row_count": requested_configuration_count,
        "complete": len(rows) == requested_configuration_count,
        "status_counts": dict(sorted(status_counts.items())),
        "distinct_ttir_hashes": len(ttir_hashes),
        "distinct_binary_hashes": len(binary_hashes),
        "ttir_frozen": len(ttir_hashes) == 1,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"results={result_dir}")
    print(json.dumps(summary, sort_keys=True))
    return 0 if len(rows) == len(configs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
