#!/usr/bin/env python3
"""Command-line entry point for the PlanComputeBlock UB cost model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

from .benchmark import benchmark_cost_model
from .cost_model import UbCostModel
from .model_types import BaselineCertificate, UnsupportedModelError
from .stages.validate_context import DEFAULT_PROFILE, CompilerProfile


def _load_json(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UnsupportedModelError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise UnsupportedModelError(f"{description} {path} must contain one JSON object")
    return value


def _optional_bool(value: Optional[str]) -> Optional[bool]:
    return None if value is None else value == "true"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict the 3x4 DynamicCV/MultiBuffer UB Delta table from PlanComputeBlock MLIR.")
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--intra-cache-num", type=int, choices=range(1, 4), required=True)
    parser.add_argument("--multibuffer-num", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--vf-merge-level", type=int, default=0)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--provenance-manifest", type=Path)
    parser.add_argument("--baseline-certificate", type=Path)
    parser.add_argument("--baseline-ub-bytes", type=int)
    parser.add_argument("--baseline-no-reuse-bytes", type=int)
    parser.add_argument("--baseline-used-reuse", choices=("true", "false"))
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--benchmark-runs", type=int, default=200)
    parser.add_argument("--format", choices=("human", "json"), default="human")
    return parser


def _format_human(payload: Mapping[str, Any]) -> str:
    prepared = payload["prepared"]
    lines = [
        f"status: {payload['status']}",
        f"model: {prepared['model_version']}",
        f"target: {prepared['target']}",
        f"normalized_ir_sha256: {prepared['normalized_ir_sha256']}",
        ("requested: "
         f"d={payload['parameters']['intra_cache_num']} "
         f"m={payload['parameters']['multibuffer_num']} "
         f"v={payload['parameters']['vf_merge_level']}"),
    ]
    summary = prepared["dynamic_cv_summary"]
    lines.extend((
        ("DynamicCV 分析: "
         f"跨块依赖={summary['broad_dependency_count']} "
         f"边界调整={summary['dependency_cut_count']} "
         f"候选缓冲={summary['supported_family_count']}"),
        "DynamicCV 跨块边界调整:",
    ))
    for cut in prepared["dynamic_cuts"]:
        family_values = ",".join(cut["family_values"]) or "-"
        lines.append(
            f"  {cut['seed_value']} b{cut['seed_producer_block']}->b{cut['activate_block']}: "
            f"{cut['origin_frontier_bytes']}B->{cut['selected_frontier_bytes']}B; "
            f"候选缓冲={family_values}")
    lines.extend((
        "",
        "d  m  Delta_d_B  Delta_m_B  Delta_dm_B  Delta_total_B  predicted_UB_B  verdict",
    ))
    for result in payload["results"]:
        values = (
            result["dynamic_from_d1_bytes"],
            result["ordinary_from_m1_bytes"],
            result["interaction_bytes"],
            result["total_from_11_bytes"],
            result["predicted_ub_bytes"],
        )
        rendered = ["-" if value is None else str(value) for value in values]
        lines.append(f"{result['intra_cache_num']}  {result['multibuffer_num']}  "
                     f"{rendered[0]:>9}  {rendered[1]:>9}  {rendered[2]:>10}  "
                     f"{rendered[3]:>13}  {rendered[4]:>14}  {result['verdict']}")
    if prepared["blockers"]:
        lines.extend(("", "不支持原因:"))
        lines.extend(f"  - {blocker}" for blocker in prepared["blockers"])
    lines.extend(("", "timing (us):"))
    for name, value in payload["timing"]["first_run_us"].items():
        lines.append(f"  {name}: {value:.3f}")
    for name, stats in payload["timing"]["benchmark"].items():
        lines.append(f"  {name}: median={stats['median_us']:.3f}, " f"p95={stats['p95_us']:.3f}, n={stats['runs']}")
    lines.extend(("", f"requested_reason: {payload['requested_result']['reason']}"))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    read_start = time.perf_counter_ns()
    try:
        text = args.ir.read_text(encoding="utf-8")
    except OSError as error:
        print(f"error: cannot read {args.ir}: {error}", file=sys.stderr)
        return 2
    input_read_us = (time.perf_counter_ns() - read_start) / 1_000.0

    try:
        profile = (CompilerProfile.from_mapping(_load_json(args.profile_json, "profile"))
                   if args.profile_json else DEFAULT_PROFILE)
        provenance = (_load_json(args.provenance_manifest, "provenance manifest") if args.provenance_manifest else None)
        if args.baseline_certificate and args.baseline_ub_bytes is not None:
            raise UnsupportedModelError("use either --baseline-certificate or scalar --baseline-* options")
        if args.baseline_certificate:
            baseline = BaselineCertificate.from_mapping(_load_json(args.baseline_certificate, "baseline certificate"))
        elif args.baseline_ub_bytes is not None:
            baseline = BaselineCertificate(
                ub_peak_bytes=args.baseline_ub_bytes,
                no_reuse_required_bytes=args.baseline_no_reuse_bytes,
                used_reuse=_optional_bool(args.baseline_used_reuse),
            )
        else:
            baseline = None
        capacity = profile.ub_capacity_bytes

        compute_start = time.perf_counter_ns()
        model = UbCostModel()
        prepared = model.prepare(
            text,
            vf_merge_level=args.vf_merge_level,
            profile=profile,
            observed_provenance=provenance,
        )
        evaluate_start = time.perf_counter_ns()
        results = model.evaluate_all(prepared, baseline)
        evaluate_all_us = (time.perf_counter_ns() - evaluate_start) / 1_000.0
        compute_total_us = (time.perf_counter_ns() - compute_start) / 1_000.0
        benchmark = benchmark_cost_model(
            text,
            baseline=baseline,
            warmup_runs=max(0, args.warmup_runs),
            benchmark_runs=max(0, args.benchmark_runs),
            vf_merge_level=args.vf_merge_level,
            profile=profile,
            observed_provenance=provenance,
        )
    except (TypeError, ValueError, UnsupportedModelError) as error:
        print(f"unsupported: {error}", file=sys.stderr)
        return 2

    requested_index = (args.intra_cache_num - 1) * 4 + args.multibuffer_num - 1
    payload = {
        "status": ("unknown" if prepared.blockers else "calibrated_predicted" if baseline else "delta_predicted"),
        "input": {
            "path": str(args.ir.resolve()),
            "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
        "parameters": {
            "intra_cache_num": args.intra_cache_num,
            "multibuffer_num": args.multibuffer_num,
            "vf_merge_level": args.vf_merge_level,
            "ub_capacity_bytes": capacity,
        },
        "compiler_profile": asdict(profile),
        "baseline": asdict(baseline) if baseline else None,
        "prepared": asdict(prepared),
        "requested_result": asdict(results[requested_index]),
        "results": [asdict(result) for result in results],
        "timing": {
            "first_run_us": {
                "input_read": input_read_us,
                **prepared.timing_us,
                "evaluate_all": evaluate_all_us,
                "compute_total": compute_total_us,
                "end_to_end": input_read_us + compute_total_us,
            },
            "benchmark": benchmark,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True) if args.format == "json" else _format_human(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
