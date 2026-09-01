"""Microbenchmarks for cold Prepare+Evaluate and cached Evaluate paths."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Mapping, Optional

from .cost_model import prepare_cost_model
from .model_types import BaselineCertificate
from .stages.evaluate import evaluate_all_configurations
from .stages.validate_context import DEFAULT_PROFILE, CompilerProfile


def _distribution(values_ns: list[int]) -> dict[str, Any]:
    ordered = sorted(values_ns)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "runs": len(ordered),
        "min_us": ordered[0] / 1_000.0,
        "median_us": statistics.median(ordered) / 1_000.0,
        "p95_us": ordered[p95_index] / 1_000.0,
        "max_us": ordered[-1] / 1_000.0,
    }


def benchmark_cost_model(
    text: str,
    *,
    baseline: Optional[BaselineCertificate],
    warmup_runs: int,
    benchmark_runs: int,
    vf_merge_level: int = 0,
    profile: CompilerProfile = DEFAULT_PROFILE,
    observed_provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if benchmark_runs <= 0:
        return {}

    def run_full() -> None:
        prepared = prepare_cost_model(
            text,
            vf_merge_level=vf_merge_level,
            profile=profile,
            observed_provenance=observed_provenance,
        )
        evaluate_all_configurations(prepared, baseline)

    prepared_once = prepare_cost_model(
        text,
        vf_merge_level=vf_merge_level,
        profile=profile,
        observed_provenance=observed_provenance,
    )

    def run_evaluate() -> None:
        evaluate_all_configurations(prepared_once, baseline)

    for _ in range(warmup_runs):
        run_full()
        run_evaluate()
    full_times = []
    evaluate_times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter_ns()
        run_full()
        full_times.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        run_evaluate()
        evaluate_times.append(time.perf_counter_ns() - start)
    return {
        "prepare_and_evaluate_12": _distribution(full_times),
        "evaluate_12_from_prepared": _distribution(evaluate_times),
    }
