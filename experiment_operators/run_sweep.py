#!/usr/bin/env python3
"""Run a complete three-axis sweep or refill one case in the latest result."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import NamedTuple, TextIO

try:
    from . import experiment_config as experiment
except ImportError:
    import experiment_config as experiment

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / ".codex-remote/results"
BENCHMARK_RE = re.compile(r"BENCHMARK operator=(?P<operator>\S+) latency_ms=(?P<latency>[0-9.]+) "
                          r"warmup=(?P<warmup>\d+) active=(?P<active>\d+)")
BENCHMARK_METHOD_RE = re.compile(r"NPU_BENCHMARK_METHOD=(?P<method>[A-Za-z0-9_.-]+)")
MISMATCHED_ELEMENTS_RE = re.compile(r"Mismatched elements:\s*(?P<count>[0-9,]+)\s*/", re.IGNORECASE)
DOMINANCE_ERROR_RE = re.compile(
    r"operand\s+#(?P<operand>\d+)\s+does(?:n't| not)\s+dominate\s+this\s+use",
    re.IGNORECASE,
)
OPERATOR_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SOURCE_BENCHMARK_OPERATOR_RE = re.compile(r"BENCHMARK\s+operator=([A-Za-z0-9][A-Za-z0-9_.-]*)")
OPERATOR_ALIASES = {"hstu_attention_fwd": "hstu_attention"}
A3_EXPERIMENT_SCHEMA = ("native-cv-depth+no-dynamic-cv+local-multibuffer-off-v11")
A5_EXPERIMENT_SCHEMA = ("dynamic-cv-slots+no-static-cv+local-multibuffer-off-v11")
OFF = "off"
RESULTS_CSV_SUFFIX_FIELDS = [
    "序号",
    "enable_dynamic_cv_pipeline",
    "enable_auto_multi_buffer",
    "multibuffer_num",
    "vf_merge_level",
    "结果",
    "原因",
    "运行延迟_ms",
    "测量方式",
    "UB使用_KiB",
    "本轮总耗时_s",
    "尝试次数",
    "自动补测次数",
    "手动补测次数",
    "首轮是否超时",
    "最终是否超时",
    "日志文件",
]


class SweepConfig(NamedTuple):
    dynamic_cv_pipeline: bool
    pipeline_value: int | None
    multibuffer_num: int | None
    vf_merge_level: int

    @property
    def auto_multibuffer(self) -> bool:
        return self.multibuffer_num is not None


def axis_value(value: int | None) -> int | str:
    return OFF if value is None else value


def validate_axis(name: str, values, minimum: int, *, allow_off: bool = False) -> tuple[int | str, ...]:
    normalized = tuple(OFF if isinstance(value, str) and value.lower() == OFF else value for value in values)
    if not normalized:
        raise SystemExit(f"{name} must contain at least one value")
    invalid = [
        value for value in normalized if (value == OFF and not allow_off) or (
            value != OFF and (isinstance(value, bool) or not isinstance(value, int)))
    ]
    if invalid:
        expected = f"integers or {OFF!r}" if allow_off else "integers"
        raise SystemExit(f"{name} must contain {expected}")
    if any(value != OFF and value < minimum for value in normalized):
        raise SystemExit(f"{name} values must be >= {minimum}")
    if len(set(normalized)) != len(normalized):
        raise SystemExit(f"{name} must not contain duplicate values")
    return normalized


def configured_values(is_a5: bool) -> tuple[tuple[int | str, ...], tuple[int | str, ...], tuple[int, ...]]:
    if is_a5:
        first = validate_axis(
            "A5_BUF_SLOT_NUM_OF_VECCORE_VALUES",
            experiment.A5_BUF_SLOT_NUM_OF_VECCORE_VALUES,
            1,
            allow_off=True,
        )
    else:
        first = validate_axis("A3_DEPTH_VALUES", experiment.A3_DEPTH_VALUES, 1)
    multibuffer = validate_axis(
        "MULTIBUFFER_NUM_VALUES",
        experiment.MULTIBUFFER_NUM_VALUES,
        1,
        allow_off=True,
    )
    vf_merge = validate_axis("VF_MERGE_LEVEL_VALUES", experiment.VF_MERGE_LEVEL_VALUES, 0)
    if (isinstance(experiment.WARMUP, bool) or not isinstance(experiment.WARMUP, int) or experiment.WARMUP < 1):
        raise SystemExit("WARMUP must be a positive integer")
    if (isinstance(experiment.ACTIVE, bool) or not isinstance(experiment.ACTIVE, int) or experiment.ACTIVE < 1):
        raise SystemExit("ACTIVE must be a positive integer")
    if experiment.CASE_TIMEOUT_SECONDS <= 0:
        raise SystemExit("CASE_TIMEOUT_SECONDS must be positive")
    if (isinstance(experiment.TIMEOUT_RETRIES, bool) or not isinstance(experiment.TIMEOUT_RETRIES, int)
            or experiment.TIMEOUT_RETRIES < 0):
        raise SystemExit("TIMEOUT_RETRIES must be a non-negative integer")
    return first, multibuffer, vf_merge


def requested_configs(
    is_a5: bool,
    first_values: tuple[int | str, ...],
    multibuffer_values: tuple[int | str, ...],
    vf_merge_values: tuple[int, ...],
):
    for requested_pipeline in first_values:
        dynamic_cv = is_a5 and requested_pipeline != OFF
        pipeline_value = None if requested_pipeline == OFF else requested_pipeline
        for requested_multibuffer in multibuffer_values:
            multibuffer_num = None if requested_multibuffer == OFF else requested_multibuffer
            for merge in vf_merge_values:
                yield SweepConfig(dynamic_cv, pipeline_value, multibuffer_num, merge)


def config_key(config: SweepConfig, is_a5: bool) -> str:
    multibuffer = axis_value(config.multibuffer_num)
    if is_a5 and not config.dynamic_cv_pipeline:
        return f"dynoff-b{multibuffer}-m{config.vf_merge_level}"
    if is_a5:
        return (f"dynon-v{config.pipeline_value}-b{multibuffer}"
                f"-m{config.vf_merge_level}")
    return (f"d{config.pipeline_value}-b{multibuffer}"
            f"-m{config.vf_merge_level}")


class SweepProgress:
    """Render one two-line dashboard, or append plain events without a TTY."""

    BAR_WIDTH = 24

    def __init__(self, operator: str, total: int) -> None:
        self.operator = operator
        self.total = total
        self.completed = 0
        self.success = 0
        self.failed = 0
        self.unsupported = 0
        self.current = "waiting"
        self.stream, self.interactive = self._select_stream()
        self.rendered = False

    @staticmethod
    def _select_stream() -> tuple[TextIO, bool]:
        if sys.stdout.isatty():
            return sys.stdout, True
        if sys.stderr.isatty():
            return sys.stderr, True
        return sys.stdout, False

    def _terminal_columns(self) -> int:
        try:
            return max(40, os.get_terminal_size(self.stream.fileno()).columns)
        except (AttributeError, OSError, ValueError):
            return 120

    @staticmethod
    def _fit_line(value: str, columns: int) -> str:
        if len(value) <= columns:
            return value
        return value[:max(0, columns - 3)] + "..."

    def _lines(self) -> tuple[str, str]:
        ratio = self.completed / self.total if self.total else 1.0
        columns = self._terminal_columns()
        counts = (f"{self.completed}/{self.total} {ratio * 100:5.1f}% "
                  f"ok={self.success} fail={self.failed} unsupported={self.unsupported}")
        fixed_width = len(self.operator) + len(counts) + 6
        bar_width = min(self.BAR_WIDTH, max(4, columns - fixed_width))
        filled = min(bar_width, round(ratio * bar_width))
        bar = "#" * filled + "-" * (bar_width - filled)
        progress = self._fit_line(f"[{self.operator}] [{bar}] {counts}", columns)
        details = self._fit_line(f"current: {self.current}", columns)
        return progress, details

    def _clear_dashboard(self) -> None:
        if not self.interactive or not self.rendered:
            return
        self.stream.write("\r\033[2K\033[1A\r\033[2K")
        self.stream.flush()
        self.rendered = False

    def _render_dashboard(self) -> None:
        progress, details = self._lines()
        self._clear_dashboard()
        self.stream.write(f"{progress}\n{details}")
        self.stream.flush()
        self.rendered = True

    def begin(
        self,
        index: int,
        config: SweepConfig,
        pipeline_axis: str,
        attempt_number: int,
        attempt_kind: str,
    ) -> None:
        value = axis_value(config.pipeline_value)
        self.current = (f"running {index}/{self.total} {pipeline_axis}={value} "
                        f"multibuffer_num={axis_value(config.multibuffer_num)} "
                        f"vf_merge_level={config.vf_merge_level} "
                        f"attempt={attempt_number}({attempt_kind})")
        if self.interactive:
            self._render_dashboard()
        else:
            print(f"CASE_START {self.current}", flush=True)

    def note(self, message: str) -> None:
        self._clear_dashboard()
        print(message, file=self.stream, flush=True)

    def defer_timeout(self, key: str, retry_number: int) -> None:
        self.note(f"{key}: timeout; queued automatic retry "
                  f"{retry_number}/{experiment.TIMEOUT_RETRIES} after the initial sweep")

    def finish(self, key: str, row: dict) -> None:
        status = row["status"]
        self.completed += 1
        if status == "measured":
            self.success += 1
        elif status == "unsupported":
            self.unsupported += 1
        else:
            self.failed += 1
        latency = row.get("latency_ms")
        ub_kib = row.get("required_ub_kib")
        details = (f"latency_ms={latency if latency is not None else '-'} "
                   f"ub_kib={ub_kib if ub_kib is not None else '-'}")
        if status != "measured":
            details += f" reason={simple_reason(row)}"
        self.current = (f"finished case={key} status={status} {details} "
                        f"log=logs/{Path(row['log_path']).name}")
        if self.interactive:
            self._render_dashboard()
        else:
            print(
                f"CASE_RESULT {self.completed}/{self.total} "
                f"ok={self.success} fail={self.failed} unsupported={self.unsupported} "
                f"{self.current}", flush=True)

    def close(self) -> None:
        if self.interactive and self.rendered:
            self.stream.write("\n")
            self.stream.flush()
            self.rendered = False


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
        [*command, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def matching_metadata(
    cache_dir: Path,
    pipeline_axis: str,
    config: SweepConfig,
    min_mtime_ns: int,
):
    matches = []
    closest_mismatch = None
    expected = {
        "multibuffer": config.auto_multibuffer,
        "multibuffer_num": config.multibuffer_num,
        "vf_merge_level": config.vf_merge_level,
        # UnitFlag is not an experiment axis. The pinned compatibility compiler
        # retains its false default on both architectures.
        "unit_flag": None,
        "limit_auto_multi_buffer_buffer": ("no-limit" if config.auto_multibuffer else None),
    }
    if pipeline_axis == "buf_slot_num_of_veccore" and config.dynamic_cv_pipeline:
        expected.update({
            "cv_pipeline_mode": "off",
            "enable_dynamic_cv_pipeline": True,
            "buf_slot_num_of_veccore": config.pipeline_value,
            "buf_slot_num_of_crosscore": 1,
            "buf_slot_num_of_gm": 1,
            "set_workspace_multibuffer": 0,
        })
    elif pipeline_axis == "buf_slot_num_of_veccore":
        expected.update({
            "cv_pipeline_mode": "off",
            "enable_dynamic_cv_pipeline": False,
            "set_workspace_multibuffer": 0,
        })
    else:
        expected.update({
            "enable_dynamic_cv_pipeline": False,
            "set_workspace_multibuffer": config.pipeline_value,
        })
    for path in cache_dir.glob("*/**/*.json"):
        if path.name.startswith("__grp__"):
            continue
        try:
            metadata = json.loads(path.read_text())
            mtime_ns = path.stat().st_mtime_ns
        except (OSError, json.JSONDecodeError):
            continue
        if mtime_ns < min_mtime_ns:
            continue
        mismatches = {
            name: (metadata.get(name), value)
            for name, value in expected.items()
            if metadata.get(name) != value
        }
        if not mismatches:
            matches.append((mtime_ns, path, metadata))
        mismatch_candidate = (
            len(mismatches),
            -mtime_ns,
            path,
            metadata,
            mismatches,
        )
        if closest_mismatch is None or mismatch_candidate[:2] < closest_mismatch[:2]:
            closest_mismatch = mismatch_candidate
    if matches:
        path, metadata = max(matches, key=lambda item: item[0])[1:]
        return path, metadata, None
    if closest_mismatch is None:
        return None, None, f"no recent compiler metadata found under {cache_dir}"
    _, _, path, metadata, mismatches = closest_mismatch
    details = ", ".join(f"{name}={actual!r} expected={wanted!r}" for name, (actual,
                                                                            wanted) in sorted(mismatches.items()))
    fallback_rc = metadata.get("dynamic_cv_pipeline_return_code")
    if fallback_rc is not None:
        details += f", dynamic_cv_pipeline_return_code={fallback_rc!r}"
    return None, None, f"closest metadata {path.name}: {details}"


def artifact_row(
    cache_dir: Path,
    pipeline_axis: str,
    config: SweepConfig,
    min_mtime_ns: int,
) -> dict:
    metadata_path, metadata, diagnostic = matching_metadata(cache_dir, pipeline_axis, config, min_mtime_ns)
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
            "metadata_diagnostic": diagnostic,
        }
    stem = metadata.get("name") or metadata_path.stem
    artifact_dir = metadata_path.parent
    ttir_path = artifact_dir / f"{stem}.ttir"
    binary_path = artifact_dir / f"{stem}.npubin"
    required_ub_bits = metadata.get("required_ub_bits") or None
    return {
        "metadata_constraints_matched": True,
        "required_ub_bits": required_ub_bits,
        "required_ub_bytes": required_ub_bits / 8 if required_ub_bits else None,
        "required_ub_kib": required_ub_bits / 8192 if required_ub_bits else None,
        "cache_key": artifact_dir.name,
        "compiler_hash": metadata.get("hash"),
        "compile_time_ms": metadata.get("compile_time_ms"),
        "ttir_hash": sha256(ttir_path),
        "binary_hash": sha256(binary_path),
        "metadata_path": str(metadata_path),
        "binary_path": str(binary_path) if binary_path.is_file() else None,
        "metadata_diagnostic": None,
    }


def compact_diagnostic(status: str, output: str, default: str) -> str:
    if status == "incorrect":
        mismatch = MISMATCHED_ELEMENTS_RE.search(output)
        if mismatch:
            return f"mismatch={mismatch.group('count').replace(',', '')}"
    if status == "compile_failed":
        dominance = DOMINANCE_ERROR_RE.search(output)
        if dominance:
            return ("BuildFinalHIVMPipelines dominance error: "
                    f"operand #{dominance.group('operand')}")
    return default


def result_label(status: str) -> str:
    if status == "measured":
        return "成功"
    if status == "unsupported":
        return "不支持"
    return "失败"


def simple_reason(row: dict) -> str:
    status = row.get("status")
    if status == "measured":
        if row.get("initial_timed_out"):
            return f"曾超时，补测后成功（共{row.get('attempt_count', 1)}次尝试）"
        if row.get("manual_rerun_count", 0):
            return "手动补测成功"
        return "编译成功、结果正确，且已记录性能和UB"
    if row.get("timed_out"):
        return f"执行超时（共{row.get('attempt_count', 1)}次尝试）"
    diagnostic = row.get("diagnostic", "")
    if status == "incorrect":
        return f"正确性验证失败（{diagnostic}）" if diagnostic else "正确性验证失败"
    if status == "compile_failed":
        if diagnostic.startswith("BuildFinalHIVMPipelines dominance error"):
            operand = diagnostic.rsplit(" ", 1)[-1]
            return f"BuildFinalHIVMPipelines产生非法IR（{operand}不支配使用位置）"
        return "编译失败"
    if diagnostic == "required_ub_bits missing from compiler metadata":
        return "未得到UB使用量"
    if diagnostic.startswith("compiler metadata mismatch: "):
        return "编译结果参数不匹配：" + diagnostic.removeprefix("compiler metadata mismatch: ")
    return "当前配置不支持"


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_measurements(rows: list[dict], result_dir: Path) -> Path:
    path = result_dir / "measurements.jsonl"
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    atomic_write(path, content)
    return path


def write_results(rows: list[dict], result_dir: Path, pipeline_axis: str) -> Path:
    path = result_dir / "results.csv"
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = [
        RESULTS_CSV_SUFFIX_FIELDS[0],
        RESULTS_CSV_SUFFIX_FIELDS[1],
        pipeline_axis,
        *RESULTS_CSV_SUFFIX_FIELDS[2:],
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for number, row in enumerate(rows, 1):
            pipeline_value = row.get(pipeline_axis)
            if pipeline_axis == "buf_slot_num_of_veccore" and not row.get("enable_dynamic_cv_pipeline", False):
                pipeline_value = OFF
            writer.writerow({
                "序号": number,
                "enable_dynamic_cv_pipeline": row.get("enable_dynamic_cv_pipeline", False),
                pipeline_axis: pipeline_value,
                "enable_auto_multi_buffer": row.get("enable_auto_multi_buffer", True),
                "multibuffer_num": row.get("multibuffer_num"),
                "vf_merge_level": row.get("vf_merge_level"),
                "结果": result_label(row.get("status", "missing")),
                "原因": simple_reason(row),
                "运行延迟_ms": row.get("latency_ms"),
                "测量方式": row.get("benchmark_method"),
                "UB使用_KiB": row.get("required_ub_kib"),
                "本轮总耗时_s": row.get("wall_time_s"),
                "尝试次数": row.get("attempt_count", 1),
                "自动补测次数": row.get("timeout_retries_used", 0),
                "手动补测次数": row.get("manual_rerun_count", 0),
                "首轮是否超时": row.get("initial_timed_out", False),
                "最终是否超时": row.get("timed_out", False),
                "日志文件": str(Path("logs") / Path(row.get("log_path", "unknown.log")).name),
            })
    temporary.replace(path)
    return path


def write_run_files(rows: list[dict], result_dir: Path, pipeline_axis: str) -> None:
    write_measurements(rows, result_dir)
    write_results(rows, result_dir, pipeline_axis)


def terminate_process_group(process: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def write_timeout_process_snapshot(log_handle, session_id: int) -> None:
    log_handle.write(b"\n[EXPERIMENT] TIMEOUT_PROCESS_SNAPSHOT\n")
    log_handle.flush()
    try:
        subprocess.run(
            [
                "ps",
                "-o",
                "pid,ppid,pgid,sid,stat,etime,comm,args",
                "--sid",
                str(session_id),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        log_handle.write(f"process snapshot unavailable: {error}\n".encode("utf-8"))
    log_handle.flush()


def requested_parameters(
    operator: str,
    candidate: Path,
    experiment_schema: str,
    pipeline_axis: str,
    config: SweepConfig,
    attempt_number: int,
    attempt_kind: str,
) -> dict:
    parameters = {
        "operator": operator,
        "candidate": str(candidate),
        "experiment_schema": experiment_schema,
        pipeline_axis: axis_value(config.pipeline_value),
        "enable_auto_multi_buffer": config.auto_multibuffer,
        "multibuffer_num": axis_value(config.multibuffer_num),
        "vf_merge_level": config.vf_merge_level,
        "enable_dynamic_cv_pipeline": config.dynamic_cv_pipeline,
        "cv_pipeline_mode": ("off" if pipeline_axis == "buf_slot_num_of_veccore" else None),
        "unit_flag": "compiler-default-false",
        "limit_auto_multi_buffer_buffer": ("no-limit" if config.auto_multibuffer else None),
        "enable_print_ub_bits": True,
        "warmup": experiment.WARMUP,
        "active": experiment.ACTIVE,
        "timeout_s": experiment.CASE_TIMEOUT_SECONDS,
        "attempt": attempt_number,
        "attempt_kind": attempt_kind,
    }
    if pipeline_axis == "buf_slot_num_of_veccore" and config.dynamic_cv_pipeline:
        parameters.update({
            "set_workspace_multibuffer": 0,
            "buf_slot_num_of_crosscore": 1,
            "buf_slot_num_of_gm": 1,
        })
    elif pipeline_axis == "buf_slot_num_of_veccore":
        parameters.update({
            "set_workspace_multibuffer": 0,
            "buf_slot_num_of_crosscore": None,
            "buf_slot_num_of_gm": None,
        })
    return parameters


def candidate_environment(config: SweepConfig, is_a5: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("EXPERIMENT_DEPTH", None)
    env.pop("EXPERIMENT_BUF_SLOT_NUM_OF_VECCORE", None)
    env.pop("EXPERIMENT_DISABLE_STATIC_CV", None)
    env.pop("EXPERIMENT_MULTIBUFFER_NUM", None)
    env.pop("EXPERIMENT_HIVM_UNIT_FLAG_SYNC", None)
    env.update({
        "ENABLE_PRINT_UB_BITS": "true",
        "TRITON_ALWAYS_COMPILE": "1",
        "TRITON_PRINT_AUTOTUNING": "1",
        "TRITON_PRINT_IR_AFTER_FAILURE": "0",
        "EXPERIMENT_DYNAMIC_CV": "1" if config.dynamic_cv_pipeline else "0",
        "EXPERIMENT_MULTIBUFFER": "1" if config.auto_multibuffer else "0",
        "EXPERIMENT_VF_MERGE_LEVEL": str(config.vf_merge_level),
        "EXPERIMENT_WARMUP": str(experiment.WARMUP),
        "EXPERIMENT_ACTIVE": str(experiment.ACTIVE),
    })
    if config.auto_multibuffer:
        env["EXPERIMENT_MULTIBUFFER_NUM"] = str(config.multibuffer_num)
    if is_a5:
        env["EXPERIMENT_DISABLE_STATIC_CV"] = "1"
        if config.dynamic_cv_pipeline:
            env["EXPERIMENT_BUF_SLOT_NUM_OF_VECCORE"] = str(config.pipeline_value)
    else:
        env["EXPERIMENT_DEPTH"] = str(config.pipeline_value)
    return env


def execute_case(
    *,
    operator: str,
    candidate: Path,
    experiment_schema: str,
    pipeline_axis: str,
    is_a5: bool,
    config: SweepConfig,
    cache_dir: Path,
    log_path: Path,
    previous_row: dict | None,
    attempt_kind: str,
) -> dict:
    previous_attempt_count = int((previous_row or {}).get("attempt_count", 0))
    attempt_number = previous_attempt_count + 1
    parameters = requested_parameters(
        operator,
        candidate,
        experiment_schema,
        pipeline_axis,
        config,
        attempt_number,
        attempt_kind,
    )
    started_epoch_ns = time.time_ns()
    started = time.monotonic()
    timed_out = False
    header = (("\n" if previous_row is not None else "") +
              f"{'=' * 24} attempt {attempt_number} ({attempt_kind}) {'=' * 24}\n" +
              "[EXPERIMENT] requested_parameters=" + json.dumps(parameters, sort_keys=True) + "\n")
    mode = "ab" if previous_row is not None else "wb"
    with log_path.open(mode) as log_handle:
        log_handle.write(header.encode("utf-8"))
        log_handle.flush()
        output_start = log_handle.tell()
        process = subprocess.Popen(
            [sys.executable, "-u", str(candidate)],
            cwd=ROOT,
            env=candidate_environment(config, is_a5),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=experiment.CASE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            write_timeout_process_snapshot(log_handle, process.pid)
            terminate_process_group(process)
            returncode = 124
        except KeyboardInterrupt:
            terminate_process_group(process)
            raise
    attempt_wall_time = time.monotonic() - started
    with log_path.open("rb") as log_handle:
        log_handle.seek(output_start)
        output = log_handle.read().decode("utf-8", errors="replace")

    benchmark = BENCHMARK_RE.search(output)
    benchmark_method_match = BENCHMARK_METHOD_RE.search(output)
    correctness = returncode == 0 and benchmark is not None
    artifacts = artifact_row(
        cache_dir,
        pipeline_axis,
        config,
        started_epoch_ns - 2_000_000_000,
    )
    diagnostic = ""
    if timed_out:
        status = "unsupported"
        diagnostic = (f"candidate timed out after {experiment.CASE_TIMEOUT_SECONDS:g} seconds")
    elif returncode != 0:
        status = "incorrect" if "AssertionError" in output else "compile_failed"
        diagnostic = (output.strip().splitlines()[-1] if output.strip() else "subprocess failed")
    elif benchmark is None:
        status = "compile_failed"
        diagnostic = "benchmark result missing"
    elif not artifacts["metadata_constraints_matched"]:
        status = "unsupported"
        diagnostic = "compiler metadata mismatch: " + artifacts["metadata_diagnostic"]
    elif artifacts["required_ub_bits"] is None:
        status = "unsupported"
        diagnostic = "required_ub_bits missing from compiler metadata"
    else:
        status = "measured"
    diagnostic = compact_diagnostic(status, output, diagnostic)

    previous_history = list((previous_row or {}).get("attempt_history", []))
    previous_wall_time = float((previous_row or {}).get("wall_time_s", 0.0))
    initial_timed_out = (bool(previous_row.get("initial_timed_out")) if previous_row is not None else timed_out)
    history = [
        *previous_history,
        {
            "attempt": attempt_number,
            "kind": attempt_kind,
            "status": status,
            "diagnostic": diagnostic,
            "timed_out": timed_out,
            "returncode": returncode,
            "wall_time_s": round(attempt_wall_time, 6),
        },
    ]
    return {
        "operator":
        operator,
        "experiment_schema":
        experiment_schema,
        "depth":
        None if is_a5 else config.pipeline_value,
        "buf_slot_num_of_veccore": (config.pipeline_value if is_a5 and config.dynamic_cv_pipeline else None),
        "enable_auto_multi_buffer":
        config.auto_multibuffer,
        "multibuffer_num":
        axis_value(config.multibuffer_num),
        "resolved_local_multibuffer_num":
        config.multibuffer_num,
        "set_workspace_multibuffer":
        0 if is_a5 else config.pipeline_value,
        "cv_pipeline_mode":
        "off" if is_a5 else None,
        "enable_dynamic_cv_pipeline":
        config.dynamic_cv_pipeline,
        "unit_flag":
        None,
        "unit_flag_policy":
        "compiler-default-false",
        "buf_slot_num_of_crosscore":
        1 if is_a5 and config.dynamic_cv_pipeline else None,
        "buf_slot_num_of_gm":
        1 if is_a5 and config.dynamic_cv_pipeline else None,
        "limit_auto_multi_buffer_buffer":
        "no-limit" if config.auto_multibuffer else None,
        "vf_merge_level":
        config.vf_merge_level,
        "status":
        status,
        "diagnostic":
        diagnostic,
        "correctness_status":
        "passed" if correctness else "failed",
        "latency_ms":
        float(benchmark.group("latency")) if benchmark else None,
        "benchmark_method":
        ((benchmark_method_match.group("method") if benchmark_method_match else "npu_profiler") if benchmark else None),
        "reported_operator":
        benchmark.group("operator") if benchmark else None,
        "warmup": (int(benchmark.group("warmup")) if benchmark else experiment.WARMUP),
        "active": (int(benchmark.group("active")) if benchmark else experiment.ACTIVE),
        **artifacts,
        "wall_time_s":
        round(previous_wall_time + attempt_wall_time, 6),
        "last_attempt_wall_time_s":
        round(attempt_wall_time, 6),
        "attempt_count":
        attempt_number,
        "timeout_retries_used":
        int((previous_row or {}).get("timeout_retries_used", 0)) + (1 if attempt_kind == "automatic_retry" else 0),
        "manual_rerun_count":
        int((previous_row or {}).get("manual_rerun_count", 0)) + (1 if attempt_kind == "manual" else 0),
        "initial_timed_out":
        initial_timed_out,
        "attempt_history":
        history,
        "timed_out":
        timed_out,
        "returncode":
        returncode,
        "log_path":
        str(log_path),
    }


def normalized_operator_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    if not normalized:
        raise SystemExit(f"cannot derive an operator name from {value!r}")
    if not normalized[0].isalnum():
        normalized = f"operator_{normalized}"
    return normalized


def resolve_operator(operator_file: Path) -> tuple[str, Path]:
    candidate = operator_file.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise SystemExit(f"operator file does not exist: {candidate}")
    if candidate.suffix != ".py":
        raise SystemExit(f"operator file must be a Python file: {candidate}")
    try:
        source = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        source = ""
    match = SOURCE_BENCHMARK_OPERATOR_RE.search(source)
    inferred = match.group(1) if match else candidate.stem
    operator = normalized_operator_name(OPERATOR_ALIASES.get(inferred, inferred))
    if not OPERATOR_NAME_RE.fullmatch(operator):
        raise SystemExit(f"invalid operator name: {operator!r}")
    return operator, candidate


def experiment_context() -> tuple[bool, str, str]:
    is_a5 = os.environ.get("BISHENGIR_NATIVE_A5_REGBASE") == "1"
    pipeline_axis = "buf_slot_num_of_veccore" if is_a5 else "depth"
    schema = A5_EXPERIMENT_SCHEMA if is_a5 else A3_EXPERIMENT_SCHEMA
    return is_a5, pipeline_axis, schema


def now_run_id() -> str:
    experiment_timezone = timezone(timedelta(hours=8), name="UTC+08:00")
    return datetime.now(experiment_timezone).strftime("%Y%m%dT%H%M%S%z")


def build_manifest(
    *,
    run_id: str,
    operator: str,
    candidate: Path,
    schema: str,
    pipeline_axis: str,
    is_a5: bool,
    first_values: tuple[int | str, ...],
    multibuffer_values: tuple[int | str, ...],
    vf_merge_values: tuple[int, ...],
    configuration_count: int,
) -> dict:
    dynamic_cv_values = [False]
    if is_a5:
        dynamic_cv_values = []
        if OFF in first_values:
            dynamic_cv_values.append(False)
        if any(value != OFF for value in first_values):
            dynamic_cv_values.append(True)
    auto_multibuffer_values = []
    if OFF in multibuffer_values:
        auto_multibuffer_values.append(False)
    if any(value != OFF for value in multibuffer_values):
        auto_multibuffer_values.append(True)
    return {
        "run_id":
        run_id,
        "timezone":
        "UTC+08:00",
        "experiment_schema":
        schema,
        "operator":
        operator,
        "candidate":
        str(candidate),
        "candidate_sha256":
        sha256(candidate),
        "correctness_evidence":
        "successful process exit followed by BENCHMARK output",
        "git_commit":
        git_value("rev-parse", "HEAD"),
        "ascend_npu_ir_commit":
        git_value("rev-parse", "HEAD:third_party/ascend/AscendNPU-IR"),
        "git_status":
        git_value("status", "--short", "--ignore-submodules=dirty") or "",
        "python":
        sys.version,
        "config_file":
        str(Path(experiment.__file__).resolve()),
        "warmup":
        experiment.WARMUP,
        "active":
        experiment.ACTIVE,
        "timeout_s":
        experiment.CASE_TIMEOUT_SECONDS,
        "timeout_retries":
        experiment.TIMEOUT_RETRIES,
        "timeout_retry_order":
        "after_initial_sweep",
        "benchmark_method":
        "npu_profiler_with_explicit_event_fallback",
        "requested_configuration_count":
        configuration_count,
        "executed_configuration_count":
        configuration_count,
        "axes": {
            pipeline_axis: list(first_values),
            "enable_dynamic_cv_pipeline": dynamic_cv_values,
            "enable_auto_multi_buffer": auto_multibuffer_values,
            "multibuffer_num": list(multibuffer_values),
            "vf_merge_level": list(vf_merge_values),
        },
        "resolved_cv_constraint":
        ("static CVPipeline is disabled for every A5 row; DynamicCV on: "
         "buf_slot_num_of_veccore is explicit and set_workspace_multibuffer=0; "
         "DynamicCV off: all CVPipeline modes are disabled" if is_a5 else "static CV: set_workspace_multibuffer=depth"),
        "ordinary_multibuffer_strategy": ("off: enable-auto-multi-buffer=false and no explicit count; "
                                          "numeric: limit_auto_multi_buffer_buffer=no-limit"),
        "fixed_dynamic_cv_buffer_counts": ({
            "buf_slot_num_of_crosscore": 1,
            "buf_slot_num_of_gm": 1,
        } if is_a5 else None),
        "static_cv_pipeline_policy": ("always-disabled" if is_a5 else "controlled-by-depth"),
        "hivm_unit_flag_sync_policy":
        "compiler default: disabled",
        "configuration_order": ("config-file axis order; off precedes numeric values by default"),
    }


def run_full_sweep(operator_file: Path) -> int:
    operator, candidate = resolve_operator(operator_file)
    is_a5, pipeline_axis, schema = experiment_context()
    first_values, multibuffer_values, vf_merge_values = configured_values(is_a5)
    configs = list(requested_configs(is_a5, first_values, multibuffer_values, vf_merge_values))
    run_id = now_run_id()
    result_dir = (RESULTS_ROOT / f"{run_id}-{operator}").resolve()
    logs_dir = result_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton/cache"))
    manifest = build_manifest(
        run_id=run_id,
        operator=operator,
        candidate=candidate,
        schema=schema,
        pipeline_axis=pipeline_axis,
        is_a5=is_a5,
        first_values=first_values,
        multibuffer_values=multibuffer_values,
        vf_merge_values=vf_merge_values,
        configuration_count=len(configs),
    )
    atomic_write(result_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")

    print(f"operator={operator} architecture={'A5' if is_a5 else 'A3'} "
          f"configurations={len(configs)} config={Path(experiment.__file__).resolve()}")
    row_slots: list[dict | None] = [None] * len(configs)
    attempt_queue = deque((index, config, 0) for index, config in enumerate(configs, 1))
    progress = SweepProgress(operator, len(configs))
    retry_phase_announced = False
    while attempt_queue:
        index, config, automatic_retry_number = attempt_queue.popleft()
        if automatic_retry_number and not retry_phase_announced:
            pending = 1 + sum(retry_number > 0 for _, _, retry_number in attempt_queue)
            progress.note(f"initial sweep complete; retrying {pending} timed-out configuration(s)")
            retry_phase_announced = True
        previous = row_slots[index - 1]
        attempt_kind = "automatic_retry" if automatic_retry_number else "initial"
        attempt_number = int((previous or {}).get("attempt_count", 0)) + 1
        progress.begin(index, config, pipeline_axis, attempt_number, attempt_kind)
        key = config_key(config, is_a5)
        row = execute_case(
            operator=operator,
            candidate=candidate,
            experiment_schema=schema,
            pipeline_axis=pipeline_axis,
            is_a5=is_a5,
            config=config,
            cache_dir=cache_dir,
            log_path=logs_dir / f"{key}.log",
            previous_row=previous,
            attempt_kind=attempt_kind,
        )
        row_slots[index - 1] = row
        rows = [stored for stored in row_slots if stored is not None]
        write_run_files(rows, result_dir, pipeline_axis)
        if row["timed_out"] and automatic_retry_number < experiment.TIMEOUT_RETRIES:
            next_retry = automatic_retry_number + 1
            attempt_queue.append((index, config, next_retry))
            progress.defer_timeout(key, next_retry)
        else:
            progress.finish(key, row)

    progress.close()
    rows = [row for row in row_slots if row is not None]
    counts = Counter(result_label(row["status"]) for row in rows)
    retried = sum(bool(row.get("initial_timed_out")) for row in rows)
    recovered = sum(bool(row.get("initial_timed_out")) and not row.get("timed_out") for row in rows)
    final_timeouts = sum(bool(row.get("timed_out")) for row in rows)
    print("实验完成："
          f"成功={counts['成功']} 失败={counts['失败']} "
          f"不支持={counts['不支持']} 超时补测={retried} "
          f"补测后不再超时={recovered} 最终仍超时={final_timeouts}")
    print(f"results={result_dir / 'results.csv'}")
    print(f"measurements={result_dir / 'measurements.jsonl'}")
    print(f"case_logs={logs_dir}")
    return 0 if len(rows) == len(configs) else 1


def optional_int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


def optional_axis_value(value: str | None) -> int | str | None:
    normalized = (value or "").strip().lower()
    if not normalized:
        return None
    return OFF if normalized == OFF else int(normalized)


def optional_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def csv_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes"}


def load_legacy_results(result_dir: Path, manifest: dict) -> list[dict]:
    path = result_dir / "results.csv"
    if not path.is_file():
        return []
    pipeline_axis = ("buf_slot_num_of_veccore" if "buf_slot_num_of_veccore" in manifest.get("axes", {}) else "depth")
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            result = raw.get("结果")
            reason = raw.get("原因", "")
            if result == "成功":
                status = "measured"
                correctness = "passed"
            elif result == "不支持":
                status = "unsupported"
                correctness = "missing"
            else:
                status = "incorrect" if "正确性" in reason else "compile_failed"
                correctness = "failed" if status == "incorrect" else "missing"
            requested_pipeline = optional_axis_value(raw.get(pipeline_axis))
            value = requested_pipeline if isinstance(requested_pipeline, int) else None
            multibuffer = optional_axis_value(raw.get("multibuffer_num"))
            enable_auto_multibuffer = (csv_bool(raw.get("enable_auto_multi_buffer"))
                                       if "enable_auto_multi_buffer" in raw else multibuffer != OFF)
            ub_kib = optional_float(raw.get("UB使用_KiB"))
            timed_out = csv_bool(raw.get("最终是否超时"))
            rows.append({
                "operator": manifest.get("operator"),
                "experiment_schema": manifest.get("experiment_schema"),
                "depth": value if pipeline_axis == "depth" else None,
                "buf_slot_num_of_veccore": value if pipeline_axis == "buf_slot_num_of_veccore" else None,
                "enable_dynamic_cv_pipeline": csv_bool(raw.get("enable_dynamic_cv_pipeline")),
                "enable_auto_multi_buffer": enable_auto_multibuffer,
                "multibuffer_num": multibuffer,
                "resolved_local_multibuffer_num": (multibuffer if isinstance(multibuffer, int) else None),
                "vf_merge_level": optional_int(raw.get("vf_merge_level")),
                "status": status,
                "correctness_status": correctness,
                "diagnostic": reason,
                "latency_ms": optional_float(raw.get("运行延迟_ms")),
                "benchmark_method": raw.get("测量方式") or None,
                "required_ub_kib": ub_kib,
                "required_ub_bytes": ub_kib * 1024 if ub_kib is not None else None,
                "required_ub_bits": ub_kib * 8192 if ub_kib is not None else None,
                "wall_time_s": optional_float(raw.get("本轮总耗时_s")) or 0.0,
                "last_attempt_wall_time_s": optional_float(raw.get("本轮总耗时_s")) or 0.0,
                "attempt_count": optional_int(raw.get("尝试次数")) or 1,
                "timeout_retries_used": optional_int(raw.get("自动补测次数")) or 0,
                "manual_rerun_count": optional_int(raw.get("手动补测次数")) or 0,
                "initial_timed_out": csv_bool(raw.get("首轮是否超时")),
                "timed_out": timed_out,
                "returncode": 124 if timed_out else None,
                "attempt_history": [],
                "log_path": str(result_dir / raw.get("日志文件", "")),
            })
    return rows


def load_measurements(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def parse_run_time(run_id: str) -> datetime:
    normalized = run_id[:-1] + "+0000" if run_id.endswith("Z") else run_id
    return datetime.strptime(normalized, "%Y%m%dT%H%M%S%z")


def find_latest_operator_run(operator: str, pipeline_axis: str,
                             experiment_schema: str) -> tuple[Path, dict, list[dict]]:
    latest = None
    if not RESULTS_ROOT.is_dir():
        raise SystemExit(f"results directory does not exist: {RESULTS_ROOT}")
    for result_dir in RESULTS_ROOT.iterdir():
        manifest_path = result_dir / "manifest.json"
        if not result_dir.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("operator") != operator:
            continue
        if manifest.get("experiment_schema") != experiment_schema:
            continue
        if pipeline_axis not in manifest.get("axes", {}):
            continue
        measurements_path = result_dir / "measurements.jsonl"
        rows = (load_measurements(measurements_path) if measurements_path.is_file() else load_legacy_results(
            result_dir, manifest))
        requested = manifest.get("requested_configuration_count")
        executed = manifest.get("executed_configuration_count")
        if not isinstance(requested, int) or requested < 1:
            continue
        if executed != requested or len(rows) != requested:
            continue
        try:
            run_time = parse_run_time(manifest["run_id"])
        except (KeyError, ValueError):
            continue
        candidate = (run_time, result_dir.name, result_dir, manifest, rows)
        if latest is None or candidate[:2] > latest[:2]:
            latest = candidate
    if latest is None:
        raise SystemExit(f"no complete {pipeline_axis} result with schema {experiment_schema!r} "
                         f"found for operator {operator!r}")
    return latest[2], latest[3], latest[4]


def parse_manual_config(first_axis: str, multibuffer: str, vf_merge: str, is_a5: bool) -> SweepConfig:
    if is_a5 and first_axis.lower() == OFF:
        dynamic_cv = False
        pipeline_value = None
    else:
        if not re.fullmatch(r"[0-9]+", first_axis):
            expected = "off or a positive integer" if is_a5 else "a positive integer"
            raise SystemExit(f"first axis must be {expected}")
        pipeline_value = int(first_axis)
        if pipeline_value < 1:
            raise SystemExit("first axis must be positive")
        dynamic_cv = is_a5
    if multibuffer.lower() == OFF:
        multibuffer_num = None
    else:
        try:
            multibuffer_num = int(multibuffer)
        except ValueError as error:
            raise SystemExit("multibuffer must be 'off' or a positive integer") from error
        if multibuffer_num < 1:
            raise SystemExit("multibuffer must be 'off' or a positive integer")
    try:
        vf_merge_level = int(vf_merge)
    except ValueError as error:
        raise SystemExit("vf_merge_level must be an integer") from error
    if vf_merge_level < 0:
        raise SystemExit("vf_merge_level must be >= 0")
    return SweepConfig(dynamic_cv, pipeline_value, multibuffer_num, vf_merge_level)


def row_matches_config(row: dict, config: SweepConfig, is_a5: bool) -> bool:
    pipeline_value = row.get("buf_slot_num_of_veccore") if is_a5 else row.get("depth")
    row_multibuffer = row.get("multibuffer_num")
    if row_multibuffer is None and row.get("enable_auto_multi_buffer") is False:
        row_multibuffer = OFF
    return (bool(row.get("enable_dynamic_cv_pipeline")) == config.dynamic_cv_pipeline
            and pipeline_value == config.pipeline_value and row_multibuffer == axis_value(config.multibuffer_num)
            and row.get("vf_merge_level") == config.vf_merge_level)


def row_is_timeout(row: dict) -> bool:
    if "timed_out" in row:
        return bool(row["timed_out"])
    return (row.get("returncode") == 124 or "timed out" in str(row.get("diagnostic", "")).lower()
            or "超时" in str(row.get("diagnostic", "")))


def rerun_case(operator_file: Path, first_axis: str, multibuffer: str, vf_merge: str) -> int:
    operator, candidate = resolve_operator(operator_file)
    is_a5, pipeline_axis, schema = experiment_context()
    configured_values(is_a5)
    config = parse_manual_config(first_axis, multibuffer, vf_merge, is_a5)
    result_dir, manifest, rows = find_latest_operator_run(operator, pipeline_axis, schema)
    matching_indexes = [index for index, row in enumerate(rows) if row_matches_config(row, config, is_a5)]
    if len(matching_indexes) != 1:
        raise SystemExit("the requested combination is not a unique row in the latest complete "
                         f"result: {result_dir}")
    index = matching_indexes[0]
    previous = rows[index]
    key = config_key(config, is_a5)
    if not row_is_timeout(previous):
        prompt = (f"{key} is currently {previous.get('status')}, not timeout. "
                  "Run it and overwrite this row? [y/N] ")
        try:
            confirmed = input(prompt).strip().lower() in {"y", "yes"}
        except EOFError:
            confirmed = False
        if not confirmed:
            print(f"case_update=skipped key={key}")
            return 0
    else:
        print(f"case_update=timeout_refill key={key}; running without prompt")

    cache_dir = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton/cache"))
    logs_dir = result_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    progress = SweepProgress(operator, 1)
    progress.begin(
        1,
        config,
        pipeline_axis,
        int(previous.get("attempt_count", 0)) + 1,
        "manual",
    )
    row = execute_case(
        operator=operator,
        candidate=candidate,
        experiment_schema=manifest.get("experiment_schema", schema),
        pipeline_axis=pipeline_axis,
        is_a5=is_a5,
        config=config,
        cache_dir=cache_dir,
        log_path=logs_dir / f"{key}.log",
        previous_row=previous,
        attempt_kind="manual",
    )
    rows[index] = row
    write_run_files(rows, result_dir, pipeline_axis)
    progress.finish(key, row)
    progress.close()

    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "key": key,
        "candidate": str(candidate),
        "previous_status": previous.get("status"),
        "new_status": row.get("status"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "ascend_npu_ir_commit": git_value("rev-parse", "HEAD:third_party/ascend/AscendNPU-IR"),
    }
    manifest.setdefault("manual_case_updates", []).append(event)
    atomic_write(result_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"case_update=written result={result_dir / 'results.csv'}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Run a complete sweep, or use --case to update one existing row in "
                                                  "the latest complete result."))
    parser.add_argument("--case", action="store_true", help="rerun one existing case")
    parser.add_argument("operator_file", type=Path)
    parser.add_argument(
        "case_values",
        nargs="*",
        metavar="VALUE",
        help="--case only: first_axis multibuffer vf_merge_level; the first two accept 'off' where supported",
    )
    args = parser.parse_args(argv)
    expected = 3 if args.case else 0
    if len(args.case_values) != expected:
        parser.error("--case requires exactly: first_axis multibuffer vf_merge_level" if args.
                     case else "a full sweep accepts only the operator file")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.case:
        return rerun_case(args.operator_file, *args.case_values)
    return run_full_sweep(args.operator_file)


if __name__ == "__main__":
    raise SystemExit(main())
