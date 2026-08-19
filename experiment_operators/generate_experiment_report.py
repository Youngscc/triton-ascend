#!/usr/bin/env python3
"""Generate a self-contained interactive report from latest complete sweeps."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from summarize_latest import find_latest_runs, is_supported, pipeline_axis, row_depth

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / ".codex-remote/results"
DEFAULT_TEMPLATE = Path(__file__).with_name("experiment_report_template.html")
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "latest-summary/experiment-report.html"
OPERATOR_LABELS = {
    "fused_attention": "Fused attention",
    "unified_attention": "Unified attention",
    "hstu_attention": "HSTU attention",
}
ASCEND_NPU_IR_PATH = "third_party/ascend/AscendNPU-IR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def compact_row(row: dict) -> dict:
    ub_kib = row.get("required_ub_kib")
    if ub_kib is None and row.get("required_ub_bits") not in (None, 0):
        ub_kib = float(row["required_ub_bits"]) / 8192
    return {
        "depth": row_depth(row),
        # Keep the report at three visible axes: on A5, the first axis is the
        # ordered sequence "off, 1, 2, 3, 4" rather than adding a fourth
        # boolean filter that would obscure the direct baseline comparison.
        "intra_cache_num":
        (row.get("intra_cache_num") if row.get("enable_dynamic_cv_pipeline") is not False else "off"),
        "enable_dynamic_cv_pipeline": row.get("enable_dynamic_cv_pipeline"),
        "multibuffer_num": row.get("multibuffer_num"),
        "vf_merge_level": row.get("vf_merge_level"),
        "status": row.get("status", "missing"),
        "correctness_status": row.get("correctness_status", "missing"),
        "latency_ms": row.get("latency_ms"),
        "required_ub_kib": ub_kib,
        "required_ub_bits": row.get("required_ub_bits"),
        "binary_hash": row.get("binary_hash"),
        "diagnostic": row.get("diagnostic", ""),
    }


def git_value(*args: str) -> str | None:
    command = ["git"]
    top_git = ROOT / ".codex-remote/top-git"
    if not (ROOT / ".git").exists() and (top_git / "HEAD").is_file():
        command.extend([f"--git-dir={top_git}", f"--work-tree={ROOT}"])
    result = subprocess.run([*command, *args], cwd=ROOT, text=True, capture_output=True, check=False)
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def source_commits(manifest: dict) -> tuple[str | None, str | None]:
    triton_ascend_commit = manifest.get("git_commit")
    ascend_npu_ir_commit = manifest.get("ascend_npu_ir_commit")
    if not ascend_npu_ir_commit and triton_ascend_commit:
        # Older manifests did not always retain the gitlink. Resolve it from
        # that run's top-level commit, never from the report generator's HEAD.
        ascend_npu_ir_commit = git_value("rev-parse", f"{triton_ascend_commit}:{ASCEND_NPU_IR_PATH}")
    return triton_ascend_commit, ascend_npu_ir_commit


def report_data(latest: dict[str, dict]) -> dict:
    operators = []
    for operator, run in sorted(latest.items()):
        rows = [compact_row(row) for row in run["rows"]]
        triton_ascend_commit, ascend_npu_ir_commit = source_commits(run["manifest"])
        operators.append({
            "id":
            operator,
            "label":
            OPERATOR_LABELS.get(operator, operator),
            "run_id":
            run["run_id"],
            "schema":
            run["manifest"].get("experiment_schema", "legacy-cv-split-v0"),
            "triton_ascend_commit":
            triton_ascend_commit,
            "ascend_npu_ir_commit":
            ascend_npu_ir_commit,
            "pipeline_axis":
            pipeline_axis(run),
            "result_dir":
            run["result_dir"].name,
            "row_count":
            len(rows),
            "expected_row_count":
            run["manifest"].get("requested_configuration_count"),
            "measured_count":
            sum(is_supported(row) for row in run["rows"]),
            "distinct_binary_hashes":
            len({row["binary_hash"]
                 for row in rows
                 if row.get("binary_hash")}),
            "distinct_ttir_hashes":
            len({row.get("ttir_hash")
                 for row in run["rows"]
                 if row.get("ttir_hash")}),
            "rows":
            rows,
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operators": operators,
    }


def safe_json(data: dict) -> str:
    return (json.dumps(data, separators=(",", ":"),
                       ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e"))


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    template_path = args.template.resolve()
    output_path = args.output.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"results directory does not exist: {results_dir}")
    if not template_path.is_file():
        raise SystemExit(f"report template does not exist: {template_path}")

    latest = find_latest_runs(results_dir)
    if not latest:
        raise SystemExit(f"no complete full-sweep results found under {results_dir}")
    data = report_data(latest)
    template = template_path.read_text(encoding="utf-8")
    placeholder = "__EXPERIMENT_REPORT_DATA__"
    if template.count(placeholder) != 1:
        raise SystemExit(f"template must contain exactly one {placeholder} placeholder")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.replace(placeholder, safe_json(data)), encoding="utf-8")
    print(f"operators={len(data['operators'])}")
    print(f"rows={sum(operator['row_count'] for operator in data['operators'])}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
