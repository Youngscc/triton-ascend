# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

VALID_KERNEL_TYPES = ("vector", "cube", "mixcv")
VALID_MODES = ("baseline", "shadow", "prune")

ALL_SEARCH_VALUES: dict[str, list[Any]] = {
    "num_stages": [1, 2],
    "unit_flag": [False, True],
    "limit_auto_multi_buffer_only_for_local_buffer": [False, True],
    "limit_auto_multi_buffer_of_local_buffer": ["no-l0c", "no-limit"],
    "set_workspace_multibuffer": [2, 4],
    "enable_hivm_auto_cv_balance": [False, True],
    "tile_mix_vector_loop": [2, 4, 8],
    "tile_mix_cube_loop": [2, 4, 8],
    "enable_ubuf_saving": [False, True],
}

KERNEL_TYPE_PARAMS: dict[str, tuple[str, ...]] = {
    "vector": ("num_stages", "enable_ubuf_saving"),
    "cube": ("num_stages", "unit_flag", "limit_auto_multi_buffer_of_local_buffer"),
    "mixcv": tuple(ALL_SEARCH_VALUES),
}

# Defaults are the current BiSheng product values. Parameters unsupported for a
# kernel type remain fixed to these values and are still emitted explicitly.
DEFAULT_TUNABLE_VALUES: dict[str, Any] = {
    "num_stages": 1,
    "unit_flag": False,
    "limit_auto_multi_buffer_only_for_local_buffer": False,
    "limit_auto_multi_buffer_of_local_buffer": "no-l0c",
    "set_workspace_multibuffer": 4,
    "enable_hivm_auto_cv_balance": False,
    "tile_mix_vector_loop": 2,
    "tile_mix_cube_loop": 2,
    "enable_ubuf_saving": False,
}

DEFAULT_FIXED_BISHENG_OPTIONS = (
    "--enable-hfusion-compile=true",
    "--enable-hivm-compile=true",
    "--enable-triton-kernel-compile=true",
    "--disable-auto-cv-work-space-manage=false",
    "--enable-preload=false",
    "--enable-code-motion=true",
    "--enable-auto-bind-sub-block=true",
    "--enable-hivm-auto-storage-align=true",
    "--limit-auto-multi-buffer-buffer=only-cube",
    "--enable-hivm-cross-core-gss=true",
    "--enable-hivm-inject-block-all-sync=false",
    "--disable-auto-inject-block-sync=false",
)

_RESERVED_OPTION_PREFIXES = (
    "--enable-ub-overflow-prediction",
    "--prune-predicted-ub-overflow",
    "--enable-auto-multi-buffer",
    "--enable-hivm-unit-flag-sync",
    "--limit-auto-multi-buffer-only-for-local-buffer",
    "--limit-auto-multi-buffer-of-local-buffer",
    "--set-workspace-multibuffer",
    "--enable-hivm-auto-cv-balance",
    "--tile-mix-vector-loop",
    "--tile-mix-cube-loop",
    "--enable-ubuf-saving",
)

_MODEL_RESULT_PREFIX = "BISHENGIR_UB_MODEL_RESULT "
_FALLBACK_RE = re.compile(r"^\[BISHENG\]\[FALLBACK\]\[RETRY\] (?P<cause>.+?) detected; "
                          r"automatically set (?P<option>[^ ]+) to (?P<value>[^ ]+) and retrying compilation\.$")
_NATIVE_UB_OVERFLOW_RE = re.compile(
    r"ub overflow, requires (?P<required_bits>\d+) bits while (?P<capacity_bits>\d+) bits available",
    re.IGNORECASE,
)
_UB_SIZE_RE = re.compile(r"UB\s+size\s*=\s*(?P<required_bits>\d+)\s*bits", re.IGNORECASE)


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = _stable_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_scalar(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "unknown":
        return None
    try:
        return int(value)
    except ValueError:
        return value


@dataclass(frozen=True)
class AdapterSpec:
    path: Path
    kernel_type: str

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        if self.kernel_type not in VALID_KERNEL_TYPES:
            raise ValueError(f"Unsupported kernel_type '{self.kernel_type}'. Expected one of {VALID_KERNEL_TYPES}.")


@dataclass(frozen=True)
class AdapterCompileConfig:
    kwargs: dict[str, Any]
    num_stages: int


@dataclass
class CompileOnlyOptions:
    adapters: Sequence[AdapterSpec]
    compiler: Path
    report_dir: Path
    modes: Sequence[str] = VALID_MODES
    repeat: int = 5
    jobs: int = 1
    timeout: float = 300.0
    resume: bool = False
    order_seed: int = 0
    fixed_bisheng_options: Sequence[str] = DEFAULT_FIXED_BISHENG_OPTIONS
    limit_configs: Optional[int] = None
    progress_interval: int = 100

    def validate(self) -> None:
        self.compiler = Path(self.compiler).expanduser().resolve()
        self.report_dir = Path(self.report_dir).expanduser().resolve()
        if not self.adapters:
            raise ValueError("At least one adapter is required.")
        adapter_keys = [(str(adapter.path.resolve()), adapter.kernel_type) for adapter in self.adapters]
        if len(adapter_keys) != len(set(adapter_keys)):
            raise ValueError("Duplicate adapter/kernel_type entries are not allowed.")
        if not self.modes:
            raise ValueError("At least one mode is required.")
        if len(self.modes) != len(set(self.modes)):
            raise ValueError("Duplicate modes are not allowed.")
        if self.repeat < 1:
            raise ValueError("repeat must be at least 1.")
        if self.jobs < 1:
            raise ValueError("jobs must be at least 1.")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0.")
        if self.limit_configs is not None and self.limit_configs < 1:
            raise ValueError("limit_configs must be at least 1 when provided.")
        if self.progress_interval < 0:
            raise ValueError("progress_interval cannot be negative.")
        invalid_modes = sorted(set(self.modes) - set(VALID_MODES))
        if invalid_modes:
            raise ValueError(f"Unsupported mode(s): {', '.join(invalid_modes)}")
        if not self.compiler.is_file():
            raise FileNotFoundError(f"bishengir-compile not found: {self.compiler}")
        if not os.access(self.compiler, os.X_OK):
            raise PermissionError(f"bishengir-compile is not executable: {self.compiler}")
        for adapter in self.adapters:
            if not adapter.path.is_file():
                raise FileNotFoundError(f"Adapter not found: {adapter.path}")
        for option in self.fixed_bisheng_options:
            if any(option.startswith(prefix) for prefix in _RESERVED_OPTION_PREFIXES):
                raise ValueError(f"fixed_bisheng_options must not override a mode or tunable option: {option}")


@dataclass(frozen=True)
class CompileTask:
    run_id: str
    candidate_id: str
    adapter: AdapterSpec
    adapter_digest: str
    mode: str
    repeat: int
    order_index: int
    config_id: str
    normalized_config: dict[str, Any]


@dataclass
class CompileOnlyCandidateResult:
    run_id: str
    candidate_id: str
    adapter_path: str
    adapter_digest: str
    kernel_type: str
    mode: str
    repeat: int
    order_index: int
    config_id: str
    normalized_config: dict[str, Any]
    bisheng_arguments: list[str]
    compiler_returncode: int
    timed_out: bool
    status: str
    reached_plan_memory: bool
    candidate_wall_ns: int
    model_serialize_ns: int
    model_ns: int
    model_status: Optional[str]
    precision: Optional[str]
    overflow: Optional[bool]
    ub_peak_bits: Optional[int]
    required_bits: Optional[int]
    capacity_bits: Optional[int]
    selected_seed: Optional[int]
    decision_path: Optional[str]
    non_overflow_upper_bound_proven: Optional[bool]
    conservative_upper_bound_bits: Optional[int]
    pipeline_fingerprint: Optional[str]
    attempt_count: int
    attempt_results: list[dict[str, Any]]
    fallback_count: int
    fallback_actions: list[dict[str, str]]
    diagnostic_category: Optional[str]
    stderr_digest: str


@dataclass
class CompileOnlySummary:
    run_id: str
    total_candidates: int
    executed_candidates: int
    resumed_candidates: int
    sweep_wall_ns: int
    report_path: str
    mode_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    best_config: None = None


def _build_standalone_configs(kernel_type: str, values: dict[str, list[Any]]) -> list[AdapterCompileConfig]:
    keys = list(values)
    configs = []
    for combination in itertools.product(*(values[key] for key in keys)):
        kwargs = {}
        num_stages = DEFAULT_TUNABLE_VALUES["num_stages"]
        for key, value in zip(keys, combination):
            if key == "num_stages":
                num_stages = value
            else:
                kwargs[key] = value
        configs.append(AdapterCompileConfig(kwargs=kwargs, num_stages=num_stages))
    return configs


def build_all_configs(kernel_type: str) -> list[Any]:
    if kernel_type not in VALID_KERNEL_TYPES:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    values = {name: ALL_SEARCH_VALUES[name] for name in KERNEL_TYPE_PARAMS[kernel_type]}
    if __package__:
        # Installed Triton uses the same Config expansion as normal autotune.
        from triton.runtime.autotuner import Config

        from .autotuner import get_max_configs

        base = Config({}, num_warps=4, num_stages=2, num_ctas=1)
        configs = get_max_configs(base, kernel_type=kernel_type, **values)
    else:
        # A source-tree Mac run must not require the Triton extension or torch.
        configs = _build_standalone_configs(kernel_type, values)

    unique: dict[str, Any] = {}
    for config in configs:
        normalized = normalize_config(config, kernel_type)
        unique.setdefault(_stable_json(normalized), config)
    return [unique[key] for key in sorted(unique)]


def normalize_config(config: Any, kernel_type: str) -> dict[str, Any]:
    values = dict(DEFAULT_TUNABLE_VALUES)
    values["num_stages"] = config.num_stages
    for name in KERNEL_TYPE_PARAMS[kernel_type]:
        if name != "num_stages" and name in config.kwargs:
            values[name] = config.kwargs[name]
    values["multibuffer"] = values["num_stages"] != 1
    return values


def config_to_bisheng_options(config: dict[str, Any]) -> list[str]:
    return [
        f"--enable-auto-multi-buffer={_bool(config['multibuffer'])}",
        f"--enable-hivm-unit-flag-sync={_bool(config['unit_flag'])}",
        "--limit-auto-multi-buffer-only-for-local-buffer="
        f"{_bool(config['limit_auto_multi_buffer_only_for_local_buffer'])}",
        "--limit-auto-multi-buffer-of-local-buffer="
        f"{config['limit_auto_multi_buffer_of_local_buffer']}",
        f"--set-workspace-multibuffer={config['set_workspace_multibuffer']}",
        f"--enable-hivm-auto-cv-balance={_bool(config['enable_hivm_auto_cv_balance'])}",
        f"--tile-mix-vector-loop={config['tile_mix_vector_loop']}",
        f"--tile-mix-cube-loop={config['tile_mix_cube_loop']}",
        f"--enable-ubuf-saving={_bool(config['enable_ubuf_saving'])}",
    ]


def mode_to_bisheng_options(mode: str) -> list[str]:
    if mode == "baseline":
        prediction, prune = False, False
    elif mode == "shadow":
        prediction, prune = True, False
    elif mode == "prune":
        prediction, prune = True, True
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return [
        f"--enable-ub-overflow-prediction={_bool(prediction)}",
        f"--prune-predicted-ub-overflow={_bool(prune)}",
    ]


def parse_model_results(stderr: str) -> list[dict[str, Any]]:
    results = []
    for line in stderr.splitlines():
        if not line.startswith(_MODEL_RESULT_PREFIX):
            continue
        fields: dict[str, Any] = {}
        for item in line[len(_MODEL_RESULT_PREFIX):].split():
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            fields[key] = _parse_scalar(value)
        results.append(fields)
    return results


def parse_fallback_actions(stderr: str) -> list[dict[str, str]]:
    actions = []
    for line in stderr.splitlines():
        match = _FALLBACK_RE.match(line)
        if match:
            actions.append(match.groupdict())
    return actions


def parse_native_ub_result(stderr: str, stdout: str = "") -> dict[str, Optional[int]]:
    matches = list(_NATIVE_UB_OVERFLOW_RE.finditer(stderr))
    if matches:
        fields = matches[-1].groupdict()
        return {name: int(value) for name, value in fields.items()}
    match = _UB_SIZE_RE.search(stdout)
    if match:
        return {"required_bits": int(match.group("required_bits")), "capacity_bits": None}
    return {"required_bits": None, "capacity_bits": None}


def _status(
    mode: str,
    returncode: int,
    timed_out: bool,
    stderr: str,
    model_results: list[dict[str, Any]],
    fallback_actions: list[dict[str, str]],
) -> tuple[str, bool]:
    has_blocker = any(result.get("status") == "blocker" for result in model_results)
    if timed_out:
        return "timeout", False
    if returncode == 0:
        if fallback_actions:
            return "success_after_fallback", True
        if has_blocker:
            return "model_blocker_then_success", True
        return "success_plan_memory", True

    last_model = model_results[-1] if model_results else {}
    if mode == "prune" and last_model.get("status") == "overflow" and last_model.get("precision") == "exact":
        earlier_reached_plan_memory = any(
            result.get("status") != "overflow" or result.get("precision") != "exact" for result in model_results[:-1])
        return "predicted_ub_overflow_final", earlier_reached_plan_memory
    if "ub overflow" in stderr.lower():
        return "native_ub_overflow_final", True
    return "non_ub_compile_error", False


class AdapterCompileExecutor:

    def __init__(self, options: CompileOnlyOptions):
        self.options = options

    def command_for(self, task: CompileTask) -> list[str]:
        return [
            str(self.options.compiler),
            str(task.adapter.path),
            "-o",
            os.devnull,
            *self.options.fixed_bisheng_options,
            *mode_to_bisheng_options(task.mode),
            *config_to_bisheng_options(task.normalized_config),
        ]

    def run(self, task: CompileTask) -> CompileOnlyCandidateResult:
        command = self.command_for(task)
        env = os.environ.copy()
        for name in (
                "BISHENGIR_STOP_BEFORE_LOCAL_PLAN_MEMORY",
                "BISHENGIR_DUMP_BEFORE_PLAN_MEMORY",
                "BISHENGIR_DUMP_BEFORE_CVPIPELINING",
                "BISHENGIR_UB_MODEL_VALIDATION",
                "BISHENGIR_DUMP_PLAN_MEMORY_ATTEMPTS",
                "BISHENGIR_PLAN_MEMORY_FORCE_SEED",
        ):
            env.pop(name, None)
        env["BISHENGIR_STOP_AFTER_LOCAL_PLAN_MEMORY"] = "1"
        env["BISHENGIR_UB_MODEL_EMIT_RESULT"] = "1"

        started = time.perf_counter_ns()
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                env=env,
                timeout=self.options.timeout,
                check=False,
                start_new_session=True,
            )
            stderr = completed.stderr or ""
            stdout = completed.stdout or ""
            returncode = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as error:
            stderr = error.stderr or ""
            stdout = error.stdout or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            returncode = 124
            timed_out = True
        wall_ns = time.perf_counter_ns() - started

        attempts = parse_model_results(stderr)
        fallbacks = parse_fallback_actions(stderr)
        native_ub = parse_native_ub_result(stderr, stdout)
        status, reached_plan_memory = _status(task.mode, returncode, timed_out, stderr, attempts, fallbacks)
        final = attempts[-1] if attempts else {}
        required_bits = final.get("required_bits")
        if required_bits is None:
            required_bits = native_ub["required_bits"]
        capacity_bits = final.get("capacity_bits")
        if capacity_bits is None:
            capacity_bits = native_ub["capacity_bits"]
        diagnostic_category = final.get("diagnostic_category")
        if not diagnostic_category and native_ub["required_bits"] is not None:
            diagnostic_category = "native_ub_overflow"

        return CompileOnlyCandidateResult(
            run_id=task.run_id,
            candidate_id=task.candidate_id,
            adapter_path=str(task.adapter.path),
            adapter_digest=task.adapter_digest,
            kernel_type=task.adapter.kernel_type,
            mode=task.mode,
            repeat=task.repeat,
            order_index=task.order_index,
            config_id=task.config_id,
            normalized_config=task.normalized_config,
            bisheng_arguments=command[1:],
            compiler_returncode=returncode,
            timed_out=timed_out,
            status=status,
            reached_plan_memory=reached_plan_memory,
            candidate_wall_ns=wall_ns,
            model_serialize_ns=sum(int(value.get("serialize_ns") or 0) for value in attempts),
            model_ns=sum(int(value.get("model_ns") or 0) for value in attempts),
            model_status=final.get("status"),
            precision=final.get("precision"),
            overflow=final.get("overflow"),
            ub_peak_bits=final.get("ub_peak_bits"),
            required_bits=required_bits,
            capacity_bits=capacity_bits,
            selected_seed=final.get("selected_seed"),
            decision_path=final.get("decision_path"),
            non_overflow_upper_bound_proven=final.get("non_overflow_upper_bound_proven"),
            conservative_upper_bound_bits=final.get("conservative_upper_bound_bits"),
            pipeline_fingerprint=final.get("pipeline_fingerprint"),
            attempt_count=max(len(attempts),
                              len(fallbacks) + 1),
            attempt_results=attempts,
            fallback_count=len(fallbacks),
            fallback_actions=fallbacks,
            diagnostic_category=diagnostic_category,
            stderr_digest=_digest(stderr.encode("utf-8")),
        )


def _compiler_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "digest": _digest(path.read_bytes()),
    }


def _make_tasks(options: CompileOnlyOptions) -> tuple[str, list[CompileTask]]:
    adapter_rows = []
    configs_by_type: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for kernel_type in VALID_KERNEL_TYPES:
        configs = build_all_configs(kernel_type)
        if options.limit_configs is not None:
            configs = configs[:options.limit_configs]
        configs_by_type[kernel_type] = []
        for config in configs:
            normalized = normalize_config(config, kernel_type)
            configs_by_type[kernel_type].append((_digest(normalized), normalized))

    adapter_digests = {}
    for adapter in options.adapters:
        digest = _digest(adapter.path.read_bytes())
        adapter_digests[str(adapter.path)] = digest
        adapter_rows.append({
            "path": str(adapter.path.resolve()),
            "digest": digest,
            "kernel_type": adapter.kernel_type,
        })

    run_identity = {
        "contract_version": 1,
        "executor_digest": _digest(Path(__file__).read_bytes()),
        "adapters": adapter_rows,
        "compiler": _compiler_identity(options.compiler),
        "fixed_bisheng_options": list(options.fixed_bisheng_options),
        "modes": list(options.modes),
        "repeat": options.repeat,
        "jobs": options.jobs,
        "timeout": options.timeout,
        "order_seed": options.order_seed,
        "search_values": ALL_SEARCH_VALUES,
        "limit_configs": options.limit_configs,
    }
    run_id = _digest(run_identity)

    tasks = []
    order_index = 0
    base = []
    for adapter in options.adapters:
        for config_id, normalized in configs_by_type[adapter.kernel_type]:
            base.append((adapter, config_id, normalized))

    for repeat in range(options.repeat):
        shuffled = list(base)
        random.Random(options.order_seed + repeat).shuffle(shuffled)
        modes = list(options.modes)
        if modes:
            shift = repeat % len(modes)
            modes = modes[shift:] + modes[:shift]
        for adapter, config_id, normalized in shuffled:
            for mode in modes:
                identity = {
                    "run_id": run_id,
                    "adapter": str(adapter.path.resolve()),
                    "mode": mode,
                    "repeat": repeat,
                    "config_id": config_id,
                }
                tasks.append(
                    CompileTask(
                        run_id=run_id,
                        candidate_id=_digest(identity),
                        adapter=adapter,
                        adapter_digest=adapter_digests[str(adapter.path)],
                        mode=mode,
                        repeat=repeat,
                        order_index=order_index,
                        config_id=config_id,
                        normalized_config=normalized,
                    ))
                order_index += 1
    return run_id, tasks


def _load_completed(path: Path, run_id: str) -> tuple[set[str], list[dict[str, Any]]]:
    completed: set[str] = set()
    rows = []
    if not path.is_file():
        return completed, rows
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    valid_chunks = []
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            valid_chunks.append(line)
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            # Only a partially written final line is recoverable.
            if line_number == len(lines):
                path.write_text("".join(valid_chunks), encoding="utf-8")
                break
            raise
        valid_chunks.append(line)
        if row.get("run_id") == run_id:
            completed.add(row["candidate_id"])
            rows.append(row)
    else:
        if raw and not raw.endswith("\n"):
            path.write_text(raw + "\n", encoding="utf-8")
    return completed, rows


def _percentile(values: Sequence[int], quantile: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(0, index)]


def _summarize(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {mode: [] for mode in VALID_MODES}
    for row in rows:
        grouped.setdefault(row["mode"], []).append(row)

    summaries = {}
    for mode, mode_rows in grouped.items():
        if not mode_rows:
            continue
        timings = [int(row["candidate_wall_ns"]) for row in mode_rows]
        statuses: dict[str, int] = {}
        for row in mode_rows:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        decision_paths: dict[str, int] = {}
        for row in mode_rows:
            for attempt in row.get("attempt_results", ()):
                path = attempt.get("decision_path")
                if path:
                    decision_paths[path] = decision_paths.get(path, 0) + 1
        summaries[mode] = {
            "candidates":
            len(mode_rows),
            "candidate_wall_sum_ns":
            sum(timings),
            "candidate_wall_min_ns":
            min(timings),
            "candidate_wall_median_ns":
            _percentile(timings, 0.5),
            "candidate_wall_p95_ns":
            _percentile(timings, 0.95),
            "candidate_wall_max_ns":
            max(timings),
            "model_ns_sum":
            sum(int(row.get("model_ns") or 0) for row in mode_rows),
            "fallback_count":
            sum(int(row.get("fallback_count") or 0) for row in mode_rows),
            "reached_plan_memory":
            sum(bool(row.get("reached_plan_memory")) for row in mode_rows),
            "statuses":
            statuses,
            "model_attempt_decision_paths":
            decision_paths,
            "non_overflow_fast_path_attempts":
            decision_paths.get("non_overflow_upper_bound", 0),
            "candidates_with_non_overflow_fast_path":
            sum(
                any(
                    attempt.get("decision_path") == "non_overflow_upper_bound"
                    for attempt in row.get("attempt_results", ()))
                for row in mode_rows),
        }
        repeat_totals: dict[int, int] = {}
        for row in mode_rows:
            repeat = int(row["repeat"])
            repeat_totals[repeat] = repeat_totals.get(repeat, 0) + int(row["candidate_wall_ns"])
        repeat_timings = list(repeat_totals.values())
        summaries[mode]["repeat_wall_sum_ns"] = {str(key): value for key, value in sorted(repeat_totals.items())}
        summaries[mode]["repeat_wall_median_ns"] = _percentile(repeat_timings, 0.5)
        summaries[mode]["repeat_wall_p95_ns"] = _percentile(repeat_timings, 0.95)
    return summaries


def _compare_modes(summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    baseline = summaries.get("baseline", {}).get("repeat_wall_sum_ns", {})
    comparisons = {}
    for mode, name in (("shadow", "shadow_overhead"), ("prune", "prune_net_change")):
        current = summaries.get(mode, {}).get("repeat_wall_sum_ns", {})
        if not baseline or not current:
            continue
        repeats = sorted(set(baseline) & set(current), key=int)
        deltas = {repeat: int(current[repeat]) - int(baseline[repeat]) for repeat in repeats}
        values = list(deltas.values())
        comparisons[name] = {
            "paired_repeat_delta_ns": deltas,
            "paired_repeat_delta_median_ns": _percentile(values, 0.5),
            "paired_repeats": len(repeats),
        }
    return comparisons


def _paired_candidate_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["adapter_digest"], row["config_id"], int(row["repeat"])


def _is_successful_without_overflow(row: dict[str, Any]) -> bool:
    if int(row.get("compiler_returncode") or 0) != 0 or row.get("timed_out"):
        return False
    if int(row.get("fallback_count") or 0) != 0 or row.get("overflow") is True:
        return False
    return not any(result.get("overflow") is True for result in row.get("attempt_results", ()))


def _experiment_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_mode: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], {})[_paired_candidate_key(row)] = row

    metrics: dict[str, dict[str, Any]] = {}
    baseline = by_mode.get("baseline", {})
    shadow = by_mode.get("shadow", {})
    no_overflow_keys = [
        key for key in baseline.keys() & shadow.keys()
        if _is_successful_without_overflow(baseline[key]) and _is_successful_without_overflow(shadow[key])
    ]
    if no_overflow_keys:
        baseline_ns = sum(int(baseline[key]["candidate_wall_ns"]) for key in no_overflow_keys)
        shadow_ns = sum(int(shadow[key]["candidate_wall_ns"]) for key in no_overflow_keys)
        delta_ns = shadow_ns - baseline_ns
        metrics["no_overflow_model_overhead"] = {
            "definition": "paired shadow minus baseline for successful candidate runs without overflow or fallback",
            "paired_candidate_runs": len(no_overflow_keys),
            "unique_input_configurations": len({key[:2]
                                                for key in no_overflow_keys}),
            "baseline_wall_sum_ns": baseline_ns,
            "shadow_wall_sum_ns": shadow_ns,
            "total_overhead_ns": delta_ns,
            "average_overhead_per_candidate_ns": delta_ns / len(no_overflow_keys),
            "overhead_percent": (100.0 * delta_ns / baseline_ns) if baseline_ns else None,
        }

    prune = by_mode.get("prune", {})
    all_keys = baseline.keys() & prune.keys()
    if all_keys:
        baseline_ns = sum(int(baseline[key]["candidate_wall_ns"]) for key in all_keys)
        prune_ns = sum(int(prune[key]["candidate_wall_ns"]) for key in all_keys)
        saved_ns = baseline_ns - prune_ns
        metrics["overall_prune_speedup"] = {
            "definition": "paired baseline minus prune across all candidate runs",
            "paired_candidate_runs": len(all_keys),
            "unique_input_configurations": len({key[:2]
                                                for key in all_keys}),
            "baseline_wall_sum_ns": baseline_ns,
            "prune_wall_sum_ns": prune_ns,
            "time_saved_ns": saved_ns,
            "time_saved_percent": (100.0 * saved_ns / baseline_ns) if baseline_ns else None,
            "speedup_ratio": (baseline_ns / prune_ns) if prune_ns else None,
        }
    return metrics


def run_adapter_compile_only(options: CompileOnlyOptions) -> CompileOnlySummary:
    options.validate()
    options.report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = options.report_dir / "results.jsonl"
    summary_path = options.report_dir / "summary.json"
    run_id, tasks = _make_tasks(options)

    completed, rows = _load_completed(checkpoint, run_id) if options.resume else (set(), [])
    pending = [task for task in tasks if task.candidate_id not in completed]
    if not options.resume and checkpoint.exists():
        checkpoint.unlink()

    executor = AdapterCompileExecutor(options)
    started = time.perf_counter_ns()
    with checkpoint.open("a", encoding="utf-8") as stream:
        executed = 0

        def record(result: CompileOnlyCandidateResult) -> None:
            nonlocal executed
            row = asdict(result)
            rows.append(row)
            stream.write(_stable_json(row) + "\n")
            stream.flush()
            executed += 1
            if options.progress_interval and (executed % options.progress_interval == 0 or executed == len(pending)):
                print(
                    f"Adapter compile-only: completed {executed}/{len(pending)} "
                    f"(total {len(completed) + executed}/{len(tasks)})",
                    file=sys.stderr,
                    flush=True,
                )

        if options.jobs == 1:
            for task in pending:
                record(executor.run(task))
        else:
            with ThreadPoolExecutor(max_workers=options.jobs) as pool:
                futures = {pool.submit(executor.run, task): task for task in pending}
                for future in as_completed(futures):
                    record(future.result())

    sweep_wall_ns = time.perf_counter_ns() - started
    mode_summaries = _summarize(rows)
    summary = CompileOnlySummary(
        run_id=run_id,
        total_candidates=len(tasks),
        executed_candidates=len(pending),
        resumed_candidates=len(tasks) - len(pending),
        sweep_wall_ns=sweep_wall_ns,
        report_path=str(summary_path),
        mode_summaries=mode_summaries,
        comparisons=_compare_modes(mode_summaries),
        metrics=_experiment_metrics(rows),
    )
    summary_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_manifest(path: Path, repo_root: Optional[Path] = None) -> tuple[list[AdapterSpec], tuple[str, ...]]:
    repo_root = (repo_root or _repo_root()).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    adapter_root = Path(payload.get("adapter_root", "."))
    if not adapter_root.is_absolute():
        adapter_root = repo_root / adapter_root
    adapters = [
        AdapterSpec(path=(adapter_root / row["path"]).resolve(), kernel_type=row["kernel_type"])
        for row in payload["adapters"]
    ]
    fixed = tuple(payload.get("fixed_bisheng_options", DEFAULT_FIXED_BISHENG_OPTIONS))
    return adapters, fixed


def find_compiler(explicit: Optional[str] = None) -> Path:
    candidates = []
    if explicit:
        compiler = Path(explicit).expanduser()
        if not compiler.is_file():
            raise FileNotFoundError(f"bishengir-compile not found: {compiler}")
        return compiler.resolve()
    if os.getenv("BISHENGIR_COMPILE_PATH"):
        candidates.append(Path(os.environ["BISHENGIR_COMPILE_PATH"]))
    repo_root = _repo_root()
    candidates.extend((
        repo_root / "third_party/ascend/AscendNPU-IR/build/bin/bishengir-compile",
        repo_root / "third_party/ascend/AscendNPU-IR/build/install/bin/bishengir-compile",
        repo_root.parent / "AscendNPU-IR/build/bin/bishengir-compile",
        repo_root.parent / "AscendNPU-IR/build/install/bin/bishengir-compile",
    ))
    found = shutil.which("bishengir-compile")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Unable to find bishengir-compile; pass --compiler or set BISHENGIR_COMPILE_PATH.")


def _compile_only_marker():
    pass


def run_with_autotuner(options: CompileOnlyOptions) -> CompileOnlySummary:
    from .autotuner import AutoTilingTuner

    tuner = AutoTilingTuner(
        _compile_only_marker,
        [],
        [],
        [],
        None,
        None,
        compile_only=True,
        compile_only_options=options,
    )
    return tuner.run()


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(modes) - set(VALID_MODES))
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported modes: {', '.join(invalid)}")
    return modes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Ascend adapter autotune through local PlanMemory only.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compiler")
    parser.add_argument("--search-space", choices=("all", ), default="all")
    parser.add_argument("--modes", type=_parse_modes, default=VALID_MODES)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--limit-configs", type=int)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    adapters, fixed = load_manifest(args.manifest)
    options = CompileOnlyOptions(
        adapters=adapters,
        compiler=find_compiler(args.compiler),
        report_dir=args.report_dir.resolve(),
        modes=args.modes,
        repeat=args.repeat,
        jobs=args.jobs,
        timeout=args.timeout,
        resume=args.resume,
        order_seed=args.order_seed,
        fixed_bisheng_options=fixed,
        limit_configs=args.limit_configs,
        progress_interval=args.progress_interval,
    )
    options.validate()
    if args.dry_run:
        _, tasks = _make_tasks(options)
        counts = {kernel_type: len(build_all_configs(kernel_type)) for kernel_type in VALID_KERNEL_TYPES}
        print(json.dumps({"config_counts": counts, "total_candidates": len(tasks)}, indent=2, sort_keys=True))
        return 0

    summary = run_with_autotuner(options) if __package__ else run_adapter_compile_only(options)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
