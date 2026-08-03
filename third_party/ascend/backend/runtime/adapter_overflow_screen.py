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
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

try:
    from .adapter_compile_only import (
        DEFAULT_FIXED_BISHENG_OPTIONS,
        DEFAULT_TUNABLE_VALUES,
        AdapterCompileExecutor,
        AdapterSpec,
        CompileOnlyCandidateResult,
        CompileOnlyOptions,
        CompileTask,
        _compiler_identity,
        _digest,
        _load_completed,
        find_compiler,
    )
except ImportError:
    from adapter_compile_only import (
        DEFAULT_FIXED_BISHENG_OPTIONS,
        DEFAULT_TUNABLE_VALUES,
        AdapterCompileExecutor,
        AdapterSpec,
        CompileOnlyCandidateResult,
        CompileOnlyOptions,
        CompileTask,
        _compiler_identity,
        _digest,
        _load_completed,
        find_compiler,
    )

_MIX_MODE_RE = re.compile(r'\bmix_mode\s*=\s*"(?P<mode>aiv|aic|mix)"')


@dataclass(frozen=True)
class StressConfig:
    name: str
    values: dict[str, Any]


@dataclass
class ScreenSummary:
    run_id: str
    adapters: int
    stress_configs: int
    screen_candidates: int
    predicted_overflow_adapters: int
    predicted_overflow_candidates: int
    confirmed_overflow_adapters: int
    confirmed_overflow_candidates: int
    mismatched_candidates: int
    screen_statuses: dict[str, int]
    verify_statuses: dict[str, int]
    confirmed_by_adapter: dict[str, dict[str, Any]]
    report_dir: str
    enriched_manifest: str
    confirmed_cases: str


def _normalized_config(**overrides: Any) -> dict[str, Any]:
    values = dict(DEFAULT_TUNABLE_VALUES)
    values.update(overrides)
    values["multibuffer"] = values["num_stages"] != 1
    return values


STRESS_CONFIGS = (
    StressConfig("production_default", _normalized_config()),
    StressConfig(
        "no_mb_balanced_tile_8",
        _normalized_config(
            tile_mix_vector_loop=8,
            tile_mix_cube_loop=8,
            enable_hivm_auto_cv_balance=True,
        ),
    ),
    StressConfig("auto_mb_default", _normalized_config(num_stages=2)),
    StressConfig(
        "auto_mb_no_limit",
        _normalized_config(
            num_stages=2,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        ),
    ),
    StressConfig(
        "auto_mb_local_only",
        _normalized_config(
            num_stages=2,
            limit_auto_multi_buffer_only_for_local_buffer=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        ),
    ),
    StressConfig(
        "auto_mb_vector_tile_8",
        _normalized_config(
            num_stages=2,
            tile_mix_vector_loop=8,
        ),
    ),
    StressConfig(
        "auto_mb_cube_tile_8_unit_sync",
        _normalized_config(
            num_stages=2,
            unit_flag=True,
            tile_mix_cube_loop=8,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        ),
    ),
    StressConfig(
        "auto_mb_balanced_tile_8",
        _normalized_config(
            num_stages=2,
            tile_mix_vector_loop=8,
            tile_mix_cube_loop=8,
            enable_hivm_auto_cv_balance=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        ),
    ),
)


def infer_kernel_type(path: Path) -> str:
    modes = set(_MIX_MODE_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
    if "mix" in modes:
        return "mixcv"
    if "aic" in modes:
        return "cube"
    if modes == {"aiv"}:
        return "vector"
    # Screening all tunables is conservative when metadata is absent or mixed.
    return "mixcv"


def discover_adapters(root: Path, limit: Optional[int] = None) -> list[AdapterSpec]:
    paths = sorted(root.expanduser().resolve().glob("*.ttadapter"))
    if limit is not None:
        paths = paths[:limit]
    return [AdapterSpec(path, infer_kernel_type(path)) for path in paths]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _make_run_id(
    adapters: Sequence[AdapterSpec],
    compiler: Path,
    fixed_options: Sequence[str],
    timeout: float,
    order_seed: int,
) -> str:
    identity = {
        "contract_version": 1,
        "screen_digest": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executor_digest": hashlib.sha256(
            (Path(__file__).with_name("adapter_compile_only.py")).read_bytes()).hexdigest(),
        "compiler": _compiler_identity(compiler),
        "adapters": [
            {
                "path": str(adapter.path),
                "digest": _digest(adapter.path.read_bytes()),
                "kernel_type": adapter.kernel_type,
            }
            for adapter in adapters
        ],
        "stress_configs": [asdict(config) for config in STRESS_CONFIGS],
        "fixed_options": list(fixed_options),
        "timeout": timeout,
        "order_seed": order_seed,
    }
    return _digest(identity)


def _make_tasks(
    run_id: str,
    adapters: Sequence[AdapterSpec],
    mode: str,
    configs: Iterable[StressConfig],
    order_seed: int,
) -> list[CompileTask]:
    tasks = []
    for adapter in adapters:
        adapter_digest = _digest(adapter.path.read_bytes())
        for config in configs:
            config_id = _digest(config.values)
            candidate_id = _digest({
                "run_id": run_id,
                "mode": mode,
                "adapter": str(adapter.path),
                "adapter_digest": adapter_digest,
                "config_name": config.name,
                "config_id": config_id,
            })
            tasks.append(CompileTask(
                run_id=run_id,
                candidate_id=candidate_id,
                adapter=adapter,
                adapter_digest=adapter_digest,
                mode=mode,
                repeat=0,
                order_index=0,
                config_id=config_id,
                normalized_config=dict(config.values),
            ))
    random.Random(order_seed).shuffle(tasks)
    return [CompileTask(**{**asdict(task), "adapter": task.adapter, "order_index": index})
            for index, task in enumerate(tasks)]


def _run_tasks(
    tasks: Sequence[CompileTask],
    executor: AdapterCompileExecutor,
    checkpoint: Path,
    resume: bool,
    jobs: int,
    progress_interval: int,
    label: str,
) -> list[dict[str, Any]]:
    run_id = tasks[0].run_id if tasks else "empty"
    completed, rows = _load_completed(checkpoint, run_id) if resume else (set(), [])
    pending = [task for task in tasks if task.candidate_id not in completed]
    if not resume and checkpoint.exists():
        checkpoint.unlink()

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a", encoding="utf-8") as stream:
        executed = 0

        def record(result: CompileOnlyCandidateResult) -> None:
            nonlocal executed
            row = asdict(result)
            rows.append(row)
            stream.write(_stable_json(row) + "\n")
            stream.flush()
            executed += 1
            if progress_interval and (executed % progress_interval == 0 or executed == len(pending)):
                print(
                    f"Adapter overflow {label}: completed {executed}/{len(pending)} "
                    f"(total {len(completed) + executed}/{len(tasks)})",
                    file=sys.stderr,
                    flush=True,
                )

        if jobs == 1:
            for task in pending:
                record(executor.run(task))
        else:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = [pool.submit(executor.run, task) for task in pending]
                for future in as_completed(futures):
                    record(future.result())
    return rows


def _status_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _config_names() -> dict[str, str]:
    return {_digest(config.values): config.name for config in STRESS_CONFIGS}


def _positive_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["adapter_digest"], row["config_id"]


def _write_artifacts(
    report_dir: Path,
    adapter_root: Path,
    fixed_options: Sequence[str],
    screen_rows: Sequence[dict[str, Any]],
    verify_rows: Sequence[dict[str, Any]],
) -> tuple[Path, Path, dict[str, dict[str, Any]], int]:
    predicted = {
        _positive_key(row): row
        for row in screen_rows
        if row["status"] == "predicted_ub_overflow_final"
    }
    verified = {
        _positive_key(row): row
        for row in verify_rows
        if row["status"] == "native_ub_overflow_final"
    }
    confirmed_keys = predicted.keys() & verified.keys()
    names = _config_names()
    by_adapter: dict[str, dict[str, Any]] = {}
    cases = []
    for key in sorted(confirmed_keys):
        screen = predicted[key]
        adapter_name = Path(screen["adapter_path"]).name
        entry = by_adapter.setdefault(adapter_name, {
            "kernel_type": screen["kernel_type"],
            "confirmed_configs": 0,
            "screened_configs": len(STRESS_CONFIGS),
            "confirmed_rate": 0.0,
            "config_names": [],
        })
        entry["confirmed_configs"] += 1
        entry["config_names"].append(names[screen["config_id"]])
        cases.append({
            "adapter": adapter_name,
            "kernel_type": screen["kernel_type"],
            "config_name": names[screen["config_id"]],
            "config_id": screen["config_id"],
            "normalized_config": screen["normalized_config"],
            "required_bits": screen["required_bits"],
            "capacity_bits": screen["capacity_bits"],
        })
    for entry in by_adapter.values():
        entry["config_names"].sort()
        entry["confirmed_rate"] = entry["confirmed_configs"] / entry["screened_configs"]

    manifest_path = report_dir / "overflow_enriched_manifest.json"
    manifest = {
        "adapter_root": str(adapter_root.resolve()),
        "adapters": [
            {"path": name, "kernel_type": values["kernel_type"]}
            for name, values in sorted(by_adapter.items())
        ],
        "fixed_bisheng_options": list(fixed_options),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cases_path = report_dir / "confirmed_overflow_cases.json"
    cases_path.write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, cases_path, by_adapter, len(predicted.keys() - verified.keys())


def run_screen(
    adapter_root: Path,
    compiler: Path,
    report_dir: Path,
    jobs: int = 1,
    timeout: float = 300.0,
    resume: bool = False,
    order_seed: int = 0,
    progress_interval: int = 100,
    limit_adapters: Optional[int] = None,
    observer: Optional[Callable[[str], None]] = None,
) -> ScreenSummary:
    adapters = discover_adapters(adapter_root, limit_adapters)
    if not adapters:
        raise ValueError(f"No .ttadapter files found under {adapter_root}")
    report_dir = report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    compiler = compiler.expanduser().resolve()
    run_id = _make_run_id(adapters, compiler, DEFAULT_FIXED_BISHENG_OPTIONS, timeout, order_seed)
    options = CompileOnlyOptions(
        adapters=adapters,
        compiler=compiler,
        report_dir=report_dir,
        modes=("prune", "baseline"),
        repeat=1,
        jobs=jobs,
        timeout=timeout,
        resume=resume,
        order_seed=order_seed,
        fixed_bisheng_options=DEFAULT_FIXED_BISHENG_OPTIONS,
        progress_interval=progress_interval,
    )
    options.validate()
    executor = AdapterCompileExecutor(options)

    screen_tasks = _make_tasks(run_id, adapters, "prune", STRESS_CONFIGS, order_seed)
    screen_rows = _run_tasks(
        screen_tasks,
        executor,
        report_dir / "screen_results.jsonl",
        resume,
        jobs,
        progress_interval,
        "screen",
    )
    positives = {
        _positive_key(row)
        for row in screen_rows
        if row["status"] == "predicted_ub_overflow_final"
    }
    if observer:
        observer(f"screen complete: {len(positives)} predicted overflow candidates")

    verify_tasks = _make_tasks(run_id, adapters, "baseline", STRESS_CONFIGS, order_seed)
    verify_tasks = [
        task for task in verify_tasks
        if (task.adapter_digest, task.config_id) in positives
    ]
    verify_rows = _run_tasks(
        verify_tasks,
        executor,
        report_dir / "verify_results.jsonl",
        resume,
        jobs,
        progress_interval,
        "verify",
    ) if verify_tasks else []

    manifest_path, cases_path, by_adapter, mismatches = _write_artifacts(
        report_dir,
        adapter_root,
        DEFAULT_FIXED_BISHENG_OPTIONS,
        screen_rows,
        verify_rows,
    )
    summary = ScreenSummary(
        run_id=run_id,
        adapters=len(adapters),
        stress_configs=len(STRESS_CONFIGS),
        screen_candidates=len(screen_tasks),
        predicted_overflow_adapters=len({
            Path(row["adapter_path"]).name for row in screen_rows
            if row["status"] == "predicted_ub_overflow_final"
        }),
        predicted_overflow_candidates=len(positives),
        confirmed_overflow_adapters=len(by_adapter),
        confirmed_overflow_candidates=sum(value["confirmed_configs"] for value in by_adapter.values()),
        mismatched_candidates=mismatches,
        screen_statuses=_status_counts(screen_rows),
        verify_statuses=_status_counts(verify_rows),
        confirmed_by_adapter=by_adapter,
        report_dir=str(report_dir),
        enriched_manifest=str(manifest_path),
        confirmed_cases=str(cases_path),
    )
    (report_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Screen an adapter corpus for model-predicted and PlanMemory-confirmed UB overflow.")
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--compiler")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--limit-adapters", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    adapters = discover_adapters(args.adapter_root, args.limit_adapters)
    if args.dry_run:
        print(json.dumps({
            "adapters": len(adapters),
            "stress_configs": len(STRESS_CONFIGS),
            "screen_candidates": len(adapters) * len(STRESS_CONFIGS),
            "kernel_types": {
                kernel_type: sum(adapter.kernel_type == kernel_type for adapter in adapters)
                for kernel_type in ("vector", "cube", "mixcv")
            },
        }, indent=2, sort_keys=True))
        return 0

    started = time.perf_counter_ns()
    summary = run_screen(
        adapter_root=args.adapter_root,
        compiler=find_compiler(args.compiler),
        report_dir=args.report_dir,
        jobs=args.jobs,
        timeout=args.timeout,
        resume=args.resume,
        order_seed=args.order_seed,
        progress_interval=args.progress_interval,
        limit_adapters=args.limit_adapters,
        observer=lambda message: print(message, file=sys.stderr, flush=True),
    )
    payload = asdict(summary)
    payload["workflow_wall_ns"] = time.perf_counter_ns() - started
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
