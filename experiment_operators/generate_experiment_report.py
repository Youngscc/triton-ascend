#!/usr/bin/env python3
"""Generate a self-contained interactive report from latest complete sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from summarize_latest import find_latest_runs, is_supported, parse_run_time, pipeline_axis, row_depth

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
SIMPLE_RUN_DIR_RE = re.compile(r"^(?P<run_id>\d{8}T\d{6}(?:Z|[+-]\d{4}))-(?P<operator>.+)$")


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


def optional_int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


def optional_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def csv_bool(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean in simple results: {value!r}")


def normalize_simple_row(raw: dict[str, str], pipeline_axis_name: str) -> dict:
    pipeline_value = optional_int(raw.get(pipeline_axis_name))
    dynamic_cv = csv_bool(raw.get("enable_dynamic_cv_pipeline"))
    result = raw.get("结果", "")
    diagnostic = raw.get("原因", "")
    if result == "成功":
        status = "measured"
        correctness_status = "passed"
    elif result == "不支持":
        status = "unsupported"
        correctness_status = "missing"
    else:
        status = "incorrect" if "正确性" in diagnostic else "compile_failed"
        correctness_status = "failed" if status == "incorrect" else "missing"
    ub_kib = optional_float(raw.get("UB使用_KiB"))
    return {
        "depth": pipeline_value if pipeline_axis_name == "depth" else None,
        "intra_cache_num": pipeline_value if pipeline_axis_name == "intra_cache_num" else None,
        "enable_dynamic_cv_pipeline": dynamic_cv,
        "multibuffer_num": optional_int(raw.get("multibuffer_num")),
        "vf_merge_level": optional_int(raw.get("vf_merge_level")),
        "status": status,
        "correctness_status": correctness_status,
        "latency_ms": optional_float(raw.get("运行延迟_ms")),
        "required_ub_kib": ub_kib,
        "required_ub_bits": round(ub_kib * 8192) if ub_kib is not None else None,
        "binary_hash": None,
        "diagnostic": diagnostic,
    }


def simple_rows_are_complete(rows: list[dict], pipeline_axis_name: str) -> bool:
    if not rows:
        return False
    multibuffer_values = {row["multibuffer_num"] for row in rows}
    vf_values = {row["vf_merge_level"] for row in rows}
    if multibuffer_values != {1, 2, 3, 4} or vf_values not in ({0, 1}, {0, 1, 2}):
        return False
    if pipeline_axis_name == "intra_cache_num":
        expected = {(False, None, mb, vf) for mb in multibuffer_values for vf in vf_values}
        expected.update((True, intra, mb, vf) for intra in (1, 2, 3, 4) for mb in multibuffer_values
                        for vf in vf_values)
    else:
        expected = {(False, depth, mb, vf) for depth in (1, 2, 3, 4) for mb in multibuffer_values
                    for vf in vf_values}
    actual = {(row["enable_dynamic_cv_pipeline"], row[pipeline_axis_name], row["multibuffer_num"],
               row["vf_merge_level"]) for row in rows}
    return actual == expected and len(rows) == len(expected)


def load_simple_run(result_dir: Path) -> dict | None:
    csv_path = result_dir / "results.csv"
    if not csv_path.is_file():
        return None
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        if "intra_cache_num" in fieldnames:
            pipeline_axis_name = "intra_cache_num"
        elif "depth" in fieldnames:
            pipeline_axis_name = "depth"
        else:
            return None
        rows = [normalize_simple_row(row, pipeline_axis_name) for row in reader]
    if not simple_rows_are_complete(rows, pipeline_axis_name):
        return None

    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    match = SIMPLE_RUN_DIR_RE.fullmatch(result_dir.name)
    run_id = manifest.get("run_id") or (match.group("run_id") if match else None)
    operator = manifest.get("operator") or (match.group("operator") if match else None)
    if not run_id or not operator:
        return None
    manifest.update({
        "run_id": run_id,
        "operator": operator,
        "experiment_schema": manifest.get("experiment_schema", "simple-results-csv-v1"),
        "requested_configuration_count": len(rows),
        "executed_configuration_count": len(rows),
        "limited_smoke_run": False,
        "axes": {
            pipeline_axis_name: [1, 2, 3, 4],
            "multibuffer_num": [1, 2, 3, 4],
            "vf_merge_level": sorted({row["vf_merge_level"] for row in rows}),
        },
    })
    return {
        "operator": operator,
        "run_id": run_id,
        "run_time": parse_run_time(run_id),
        "result_dir": result_dir.resolve(),
        "manifest": manifest,
        "rows": rows,
        "source_format": "results.csv",
    }


def find_latest_report_runs(results_dir: Path) -> dict[str, dict]:
    latest = find_latest_runs(results_dir)
    for run in latest.values():
        run["source_format"] = "measurements.jsonl"
    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue
        candidate = load_simple_run(result_dir)
        if candidate is None:
            continue
        previous = latest.get(candidate["operator"])
        if previous is None or (candidate["run_time"], result_dir.name) > (
                previous["run_time"], previous["result_dir"].name):
            latest[candidate["operator"]] = candidate
    return latest


def report_data(latest: dict[str, dict]) -> dict:
    operators = []
    for operator, run in sorted(latest.items()):
        rows = [compact_row(row) for row in run["rows"]]
        triton_ascend_commit, ascend_npu_ir_commit = source_commits(run["manifest"])
        detailed = run.get("source_format") != "results.csv"
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
            (len({row["binary_hash"]
                  for row in rows
                  if row.get("binary_hash")}) if detailed else None),
            "distinct_ttir_hashes":
            (len({row.get("ttir_hash")
                  for row in run["rows"]
                  if row.get("ttir_hash")}) if detailed else None),
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

    latest = find_latest_report_runs(results_dir)
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
