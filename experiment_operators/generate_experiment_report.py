#!/usr/bin/env python3
"""Generate one self-contained HTML report from each operator's latest run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / ".codex-remote/results"
DEFAULT_TEMPLATE = Path(__file__).with_name("experiment_report_template.html")
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "latest-summary/experiment-report.html"
DEFAULT_COMBINED_CSV = DEFAULT_RESULTS_DIR / "latest-summary/combined-results.csv"
ASCEND_NPU_IR_PATH = "third_party/ascend/AscendNPU-IR"
RUN_DIR_RE = re.compile(r"^(?P<run_id>\d{8}T\d{6}(?:Z|[+-]\d{4}))-(?P<operator>.+)$")
OPERATOR_LABELS = {
    "fused_attention": "Fused attention",
    "flash_attention_npu_v8": "Flash attention NPU V8",
    "unified_attention": "Unified attention",
    "hstu_attention": "HSTU attention",
}
OFF = "off"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--combined-csv", type=Path)
    return parser.parse_args()


def parse_run_time(run_id: str) -> datetime:
    normalized = run_id[:-1] + "+0000" if run_id.endswith("Z") else run_id
    return datetime.strptime(normalized, "%Y%m%dT%H%M%S%z")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def is_complete(manifest: dict, rows: list[dict]) -> bool:
    requested = manifest.get("requested_configuration_count")
    executed = manifest.get("executed_configuration_count")
    return (isinstance(requested, int) and requested > 0 and executed == requested and len(rows) == requested
            and not manifest.get("limited_smoke_run", False))


def pipeline_axis(run: dict) -> str:
    axes = run["manifest"].get("axes", {})
    if not isinstance(axes, dict):
        raise ValueError("result axes must be an object")
    if "buf_slot_num_of_veccore" in axes:
        return "buf_slot_num_of_veccore"
    if "depth" in axes:
        return "depth"
    raise ValueError(f"result has no supported pipeline axis: {sorted(axes)}")


def row_depth(row: dict):
    return row.get("depth", row.get("set_workspace_multibuffer", row.get("cv_pipeline_depth")))


def is_supported(row: dict) -> bool:
    return (row.get("status") == "measured" and row.get("correctness_status") == "passed"
            and row.get("latency_ms") is not None and row.get("required_ub_bits") not in (None, 0))


def run_record(result_dir: Path, manifest: dict, rows: list[dict], source: str):
    operator = manifest.get("operator")
    run_id = manifest.get("run_id")
    axes = manifest.get("axes", {})
    if (not isinstance(axes, dict) or not operator or not run_id or not is_complete(manifest, rows)
            or not ({"depth", "buf_slot_num_of_veccore"} & axes.keys())):
        return None
    try:
        run_time = parse_run_time(run_id)
    except ValueError as error:
        raise ValueError(f"invalid run_id {run_id!r} in {result_dir}") from error
    return {
        "operator": operator,
        "run_id": run_id,
        "run_time": run_time,
        "result_dir": result_dir.resolve(),
        "manifest": manifest,
        "rows": rows,
        "source_format": source,
    }


def find_latest_jsonl_runs(results_dir: Path) -> dict[str, dict]:
    latest = {}
    for result_dir in sorted(results_dir.iterdir()):
        manifest_path = result_dir / "manifest.json"
        measurements_path = result_dir / "measurements.jsonl"
        if not result_dir.is_dir() or not manifest_path.is_file() or not measurements_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = run_record(result_dir, manifest, load_jsonl(measurements_path), "measurements.jsonl")
        if candidate is None:
            continue
        previous = latest.get(candidate["operator"])
        if previous is None or (candidate["run_time"], result_dir.name) > (previous["run_time"],
                                                                           previous["result_dir"].name):
            latest[candidate["operator"]] = candidate
    return latest


def optional_int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


def optional_axis_value(value: str | None) -> int | str | None:
    normalized = (value or "").strip().lower()
    if not normalized:
        return None
    return OFF if normalized == OFF else int(normalized)


def axis_sort_key(value) -> tuple[int, int]:
    if value == OFF:
        return (0, 0)
    return (1, int(value))


def optional_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def csv_bool(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean in results.csv: {value!r}")


def normalize_csv_row(raw: dict[str, str], axis: str) -> dict:
    requested_axis = optional_axis_value(raw.get(axis))
    value = requested_axis if isinstance(requested_axis, int) else None
    multibuffer = optional_axis_value(raw.get("multibuffer_num"))
    auto_multibuffer = (csv_bool(raw.get("enable_auto_multi_buffer"))
                        if "enable_auto_multi_buffer" in raw else multibuffer != OFF)
    result = raw.get("结果", "")
    diagnostic = raw.get("原因", "")
    if result == "成功":
        status, correctness = "measured", "passed"
    elif result == "不支持":
        status, correctness = "unsupported", "missing"
    else:
        status = "incorrect" if "正确性" in diagnostic else "compile_failed"
        correctness = "failed" if status == "incorrect" else "missing"
    ub_kib = optional_float(raw.get("UB使用_KiB"))
    return {
        "depth": value if axis == "depth" else None,
        "buf_slot_num_of_veccore": value if axis == "buf_slot_num_of_veccore" else None,
        "enable_dynamic_cv_pipeline": csv_bool(raw.get("enable_dynamic_cv_pipeline")),
        "enable_auto_multi_buffer": auto_multibuffer,
        "multibuffer_num": multibuffer,
        "resolved_local_multibuffer_num": (multibuffer if isinstance(multibuffer, int) else None),
        "vf_merge_level": optional_int(raw.get("vf_merge_level")),
        "status": status,
        "correctness_status": correctness,
        "latency_ms": optional_float(raw.get("运行延迟_ms")),
        "benchmark_method": raw.get("测量方式") or None,
        "required_ub_kib": ub_kib,
        "required_ub_bits": round(ub_kib * 8192) if ub_kib is not None else None,
        "binary_hash": None,
        "diagnostic": diagnostic,
    }


def load_csv_run(result_dir: Path) -> dict | None:
    csv_path = result_dir / "results.csv"
    measurements_path = result_dir / "measurements.jsonl"
    if not csv_path.is_file() or measurements_path.is_file():
        return None
    manifest_path = result_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {})
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        axis = ("buf_slot_num_of_veccore"
                if "buf_slot_num_of_veccore" in fields else "depth" if "depth" in fields else None)
        if axis is None:
            return None
        rows = [normalize_csv_row(row, axis) for row in reader]
    match = RUN_DIR_RE.fullmatch(result_dir.name)
    run_id = manifest.get("run_id") or (match.group("run_id") if match else None)
    operator = manifest.get("operator") or (match.group("operator") if match else None)
    if not run_id or not operator:
        return None
    manifest.update({
        "run_id": run_id,
        "operator": operator,
        "experiment_schema": manifest.get("experiment_schema", "legacy-results-csv-v1"),
        "requested_configuration_count": manifest.get("requested_configuration_count", len(rows)),
        "executed_configuration_count": manifest.get("executed_configuration_count", len(rows)),
        "axes": manifest.get("axes") or {
            axis:
            sorted(
                {
                    OFF if axis == "buf_slot_num_of_veccore" and not row["enable_dynamic_cv_pipeline"] else row[axis]
                    for row in rows
                }, key=axis_sort_key),
            "multibuffer_num":
            sorted({row["multibuffer_num"]
                    for row in rows}, key=axis_sort_key),
            "vf_merge_level":
            sorted({row["vf_merge_level"]
                    for row in rows}),
        },
    })
    unique_configs = {(
        row["enable_dynamic_cv_pipeline"],
        row[axis],
        row["multibuffer_num"],
        row["vf_merge_level"],
    )
                      for row in rows}
    if len(unique_configs) != len(rows):
        return None
    return run_record(result_dir, manifest, rows, "results.csv")


def find_latest_report_runs(results_dir: Path) -> dict[str, dict]:
    latest = find_latest_jsonl_runs(results_dir)
    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue
        candidate = load_csv_run(result_dir)
        if candidate is None:
            continue
        previous = latest.get(candidate["operator"])
        if previous is None or (candidate["run_time"], result_dir.name) > (previous["run_time"],
                                                                           previous["result_dir"].name):
            latest[candidate["operator"]] = candidate
    return latest


def compact_row(row: dict) -> dict:
    ub_kib = row.get("required_ub_kib")
    if ub_kib is None and row.get("required_ub_bits") not in (None, 0):
        ub_kib = float(row["required_ub_bits"]) / 8192
    auto_multibuffer = row.get("enable_auto_multi_buffer")
    multibuffer_num = row.get("multibuffer_num", row.get("resolved_local_multibuffer_num"))
    if auto_multibuffer is False:
        multibuffer_num = OFF
    return {
        "depth":
        row_depth(row),
        "buf_slot_num_of_veccore":
        (row.get("buf_slot_num_of_veccore") if row.get("enable_dynamic_cv_pipeline") is not False else "off"),
        "enable_dynamic_cv_pipeline":
        row.get("enable_dynamic_cv_pipeline"),
        "enable_auto_multi_buffer":
        auto_multibuffer,
        "multibuffer_num":
        multibuffer_num,
        "vf_merge_level":
        row.get("vf_merge_level"),
        "status":
        row.get("status", "missing"),
        "correctness_status":
        row.get("correctness_status", "missing"),
        "latency_ms":
        row.get("latency_ms"),
        "benchmark_method":
        row.get("benchmark_method"),
        "required_ub_kib":
        ub_kib,
        "required_ub_bits":
        row.get("required_ub_bits"),
        "binary_hash":
        row.get("binary_hash"),
        "diagnostic":
        row.get("diagnostic", ""),
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
    top_commit = manifest.get("git_commit")
    submodule_commit = manifest.get("ascend_npu_ir_commit")
    if not submodule_commit and top_commit:
        submodule_commit = git_value("rev-parse", f"{top_commit}:{ASCEND_NPU_IR_PATH}")
    return top_commit, submodule_commit


def report_data(latest: dict[str, dict]) -> dict:
    operators = []
    for operator, run in sorted(latest.items()):
        rows = [compact_row(row) for row in run["rows"]]
        top_commit, submodule_commit = source_commits(run["manifest"])
        detailed = run["source_format"] == "measurements.jsonl"
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
            top_commit,
            "ascend_npu_ir_commit":
            submodule_commit,
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
            "distinct_binary_hashes": (len({row["binary_hash"]
                                            for row in rows
                                            if row.get("binary_hash")}) if detailed else None),
            "distinct_ttir_hashes": (len({row.get("ttir_hash")
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


def write_combined_results_csv(latest: dict[str, dict], output_path: Path) -> int:
    provenance_fields = ["算子", "run_id", "结果目录"]
    source_records = []
    combined_fields = []
    for operator, run in sorted(latest.items()):
        csv_path = run["result_dir"] / "results.csv"
        if not csv_path.is_file():
            raise ValueError(f"selected report run has no results.csv: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            source_fields = list(reader.fieldnames or ())
            if not source_fields:
                raise ValueError(f"results.csv has no header: {csv_path}")
            source_rows = list(reader)
        if len(source_rows) != len(run["rows"]):
            raise ValueError(f"results.csv row count does not match selected run: {csv_path} "
                             f"({len(source_rows)} != {len(run['rows'])})")
        for field in source_fields:
            if field not in provenance_fields and field not in combined_fields:
                combined_fields.append(field)
        source_records.append((operator, run, source_rows))

    fieldnames = [*provenance_fields, *combined_fields]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    row_count = 0
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for operator, run, source_rows in source_records:
            for source_row in source_rows:
                writer.writerow({
                    **source_row,
                    "算子": operator,
                    "run_id": run["run_id"],
                    "结果目录": run["result_dir"].name,
                })
                row_count += 1
    temporary.replace(output_path)
    return row_count


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    template_path = args.template.resolve()
    output_path = args.output.resolve()
    combined_csv_path = (args.combined_csv.resolve()
                         if args.combined_csv else output_path.with_name(DEFAULT_COMBINED_CSV.name))
    if not results_dir.is_dir():
        raise SystemExit(f"results directory does not exist: {results_dir}")
    if not template_path.is_file():
        raise SystemExit(f"report template does not exist: {template_path}")
    latest = find_latest_report_runs(results_dir)
    if not latest:
        raise SystemExit(f"no complete experiment results found under {results_dir}")
    data = report_data(latest)
    template = template_path.read_text(encoding="utf-8")
    placeholder = "__EXPERIMENT_REPORT_DATA__"
    if template.count(placeholder) != 1:
        raise SystemExit(f"template must contain exactly one {placeholder} placeholder")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.replace(placeholder, safe_json(data)), encoding="utf-8")
    combined_row_count = write_combined_results_csv(latest, combined_csv_path)
    print(f"operators={len(data['operators'])}")
    print(f"rows={sum(operator['row_count'] for operator in data['operators'])}")
    print(f"output={output_path}")
    print(f"combined_csv_rows={combined_row_count}")
    print(f"combined_csv={combined_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
