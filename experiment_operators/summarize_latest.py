#!/usr/bin/env python3
"""Summarize supported configurations from each operator's latest sweep.

This script does not rank configurations or choose a winner.  It selects the
latest auditable result directory per operator, emits every measured row, and
reports coverage and failure statistics for the complete selected sweeps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / ".codex-remote/results"
SUPPORTED_FIELDS = [
    "operator",
    "depth",
    "intra_cache_num",
    "multibuffer_num",
    "vf_merge_level",
    "latency_ms",
    "required_ub_kib",
    "required_ub_bytes",
    "required_ub_bits",
    "compile_time_ms",
    "warmup",
    "active",
    "run_id",
    "experiment_schema",
    "ttir_hash",
    "binary_hash",
    "cache_key",
    "result_dir",
]
EFFECT_FIELDS = [
    "operator",
    "variable",
    "value",
    "reference_value",
    "controlled_slice",
    "matched_samples",
    "latency_ms_median",
    "latency_delta_pct",
    "required_ub_kib_median",
    "ub_delta_pct",
    "latency_ms_mean",
    "required_ub_kib_mean",
    "reference_latency_ms_median",
    "reference_ub_kib_median",
    "run_id",
]
EFFECT_SPECS = [
    {
        "variable": "depth",
        "reference": 1,
        "controlled_slice": "multibuffer_num=1; matched across vf_merge_level",
        "x_label": "CV depth",
    },
    {
        "variable": "intra_cache_num",
        "reference": 1,
        "controlled_slice": "multibuffer_num=1; matched across vf_merge_level",
        "x_label": "DynamicCV intra cache count",
    },
    {
        "variable": "multibuffer_num",
        "reference": 1,
        "controlled_slice": "pipeline axis=4; matched across vf_merge_level",
        "x_label": "Ordinary multibuffer count",
    },
    {
        "variable": "vf_merge_level",
        "reference": 0,
        "controlled_slice": "matched across identical (pipeline axis, multibuffer_num)",
        "x_label": "VF merge level",
    },
]
OPERATOR_LABELS = {
    "fused_attention": "Fused attention",
    "unified_attention": "Unified attention",
    "hstu_attention": "HSTU attention",
}
OPERATOR_COLORS = {
    "fused_attention": "#2563eb",
    "unified_attention": "#dc2626",
    "hstu_attention": "#059669",
}
FALLBACK_OPERATOR_COLORS = (
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#4338ca",
    "#0f766e",
    "#a21caf",
)


def operator_label(operator: str) -> str:
    return OPERATOR_LABELS.get(operator, operator.replace("_", " ").title())


def operator_color(operator: str) -> str:
    if operator in OPERATOR_COLORS:
        return OPERATOR_COLORS[operator]
    digest = hashlib.sha256(operator.encode("utf-8")).digest()
    return FALLBACK_OPERATOR_COLORS[digest[0] % len(FALLBACK_OPERATOR_COLORS)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <results-dir>/latest-summary",
    )
    return parser.parse_args()


def parse_run_time(run_id: str) -> datetime:
    normalized = run_id[:-1] + "+0000" if run_id.endswith("Z") else run_id
    return datetime.strptime(normalized, "%Y%m%dT%H%M%S%z")


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def is_complete_sweep(manifest: dict, rows: list[dict]) -> bool:
    requested = manifest.get("requested_configuration_count")
    executed = manifest.get("executed_configuration_count")
    return (
        isinstance(requested, int)
        and requested > 0
        and executed == requested
        and len(rows) == requested
        and not manifest.get("limited_smoke_run", False)
    )


def find_latest_runs(results_dir: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue
        manifest_path = result_dir / "manifest.json"
        measurements_path = result_dir / "measurements.jsonl"
        if not manifest_path.is_file() or not measurements_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        operator = manifest.get("operator")
        run_id = manifest.get("run_id")
        if not operator or not run_id:
            continue
        rows = load_rows(measurements_path)
        if not is_complete_sweep(manifest, rows):
            continue
        try:
            run_time = parse_run_time(run_id)
        except ValueError as error:
            raise ValueError(f"invalid run_id {run_id!r} in {manifest_path}") from error
        candidate = {
            "operator": operator,
            "run_id": run_id,
            "run_time": run_time,
            "result_dir": result_dir.resolve(),
            "manifest": manifest,
            "rows": rows,
        }
        previous = latest.get(operator)
        if previous is None or (run_time, result_dir.name) > (
            previous["run_time"],
            previous["result_dir"].name,
        ):
            latest[operator] = candidate
    return latest


def is_supported(row: dict) -> bool:
    return (
        row.get("status") == "measured"
        and row.get("correctness_status") == "passed"
        and row.get("latency_ms") is not None
        and row.get("required_ub_bits") not in (None, 0)
    )


def coverage(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def experiment_schema(run: dict) -> str:
    return run["manifest"].get("experiment_schema", "legacy-cv-split-v0")


def row_depth(row: dict):
    return row.get(
        "depth",
        row.get("set_workspace_multibuffer", row.get("cv_pipeline_depth")),
    )


def pipeline_axis(run: dict) -> str:
    return (
        "intra_cache_num"
        if "intra_cache_num" in run["manifest"].get("axes", {})
        else "depth"
    )


def row_pipeline_value(row: dict):
    intra = row.get("intra_cache_num")
    return intra if intra is not None else row_depth(row)


def summarize_run(run: dict) -> dict:
    rows = run["rows"]
    supported = [row for row in rows if is_supported(row)]
    statuses = Counter(row.get("status", "missing") for row in rows)
    correctness_passed = sum(row.get("correctness_status") == "passed" for row in rows)
    latency_present = sum(row.get("latency_ms") is not None for row in rows)
    ub_present = sum(row.get("required_ub_bits") not in (None, 0) for row in rows)
    timed_out = sum(bool(row.get("timed_out")) for row in rows)
    latencies = [float(row["latency_ms"]) for row in supported]
    ub_bits = [int(row["required_ub_bits"]) for row in supported]
    result = {
        "operator": run["operator"],
        "run_id": run["run_id"],
        "result_dir": str(run["result_dir"]),
        "experiment_schema": experiment_schema(run),
        "row_count": len(rows),
        "expected_row_count": run["manifest"].get("requested_configuration_count"),
        "status_counts": dict(sorted(statuses.items())),
        "supported_count": len(supported),
        "correctness_passed_count": correctness_passed,
        "correctness_coverage": coverage(correctness_passed, len(rows)),
        "latency_present_count": latency_present,
        "latency_coverage": coverage(latency_present, len(rows)),
        "ub_present_count": ub_present,
        "ub_coverage": coverage(ub_present, len(rows)),
        "timed_out_count": timed_out,
        "distinct_cache_keys": len({row.get("cache_key") for row in rows if row.get("cache_key")}),
        "distinct_ttir_hashes": len({row.get("ttir_hash") for row in rows if row.get("ttir_hash")}),
        "distinct_binary_hashes": len({row.get("binary_hash") for row in rows if row.get("binary_hash")}),
        "supported_latency_ms": None,
        "supported_ub_bits": None,
    }
    if latencies:
        result["supported_latency_ms"] = {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
        }
    if ub_bits:
        result["supported_ub_bits"] = {
            "min": min(ub_bits),
            "max": max(ub_bits),
            "distinct": sorted(set(ub_bits)),
        }
    return result


def supported_rows(latest: dict[str, dict]) -> list[dict]:
    output = []
    for operator, run in sorted(latest.items()):
        for row in sorted(
            (row for row in run["rows"] if is_supported(row)),
            key=lambda item: (
                row_pipeline_value(item),
                item.get("multibuffer_num", -1),
                item["vf_merge_level"],
            ),
        ):
            output.append(
                {
                    field: (
                        operator
                        if field == "operator"
                        else run["run_id"]
                        if field == "run_id"
                        else str(run["result_dir"])
                        if field == "result_dir"
                        else experiment_schema(run)
                        if field == "experiment_schema"
                        else row_depth(row)
                        if field == "depth"
                        else row.get(field)
                    )
                    for field in SUPPORTED_FIELDS
                }
            )
    return output


def row_ub_kib(row: dict) -> float | None:
    value = row.get("required_ub_kib")
    if value is not None:
        return float(value)
    bits = row.get("required_ub_bits")
    return float(bits) / 8192 if bits not in (None, 0) else None


def effect_value(row: dict, variable: str):
    if variable == "depth":
        return row_depth(row)
    return row.get(variable)


def effect_control_key(row: dict, variable: str):
    pipeline_value = row_pipeline_value(row)
    multibuffer_num = row.get("multibuffer_num")
    merge = row.get("vf_merge_level")
    if variable in ("depth", "intra_cache_num"):
        return merge if multibuffer_num == 1 else None
    if variable == "multibuffer_num":
        return merge if pipeline_value == 4 else None
    if variable == "vf_merge_level":
        if pipeline_value is None or multibuffer_num is None:
            return None
        return pipeline_value, multibuffer_num
    raise ValueError(f"unknown effect variable: {variable}")


def rounded_median(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 6) if values else None


def rounded_mean(values: list[float]) -> float | None:
    return round(float(statistics.fmean(values)), 6) if values else None


def percent_delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in (None, 0):
        return None
    return round((value / reference - 1.0) * 100.0, 3)


def effect_rows(latest: dict[str, dict]) -> list[dict]:
    output = []
    for operator, run in sorted(latest.items()):
        supported = [row for row in run["rows"] if is_supported(row)]
        for spec in EFFECT_SPECS:
            variable = spec["variable"]
            if variable in ("depth", "intra_cache_num") and variable != pipeline_axis(run):
                continue
            grouped: dict[int, dict[object, dict]] = {}
            for row in supported:
                value = effect_value(row, variable)
                control_key = effect_control_key(row, variable)
                if value is None or control_key is None:
                    continue
                grouped.setdefault(int(value), {})[control_key] = row

            reference_value = spec["reference"]
            reference_group = grouped.get(reference_value, {})
            for value, candidate_group in sorted(grouped.items()):
                matched_keys = sorted(set(candidate_group) & set(reference_group))
                candidate_rows = [candidate_group[key] for key in matched_keys]
                reference_rows = [reference_group[key] for key in matched_keys]
                candidate_latency = [float(row["latency_ms"]) for row in candidate_rows]
                reference_latency = [float(row["latency_ms"]) for row in reference_rows]
                candidate_ub = [row_ub_kib(row) for row in candidate_rows]
                reference_ub = [row_ub_kib(row) for row in reference_rows]
                candidate_ub = [value for value in candidate_ub if value is not None]
                reference_ub = [value for value in reference_ub if value is not None]

                latency_median = rounded_median(candidate_latency)
                reference_latency_median = rounded_median(reference_latency)
                ub_median = rounded_median(candidate_ub)
                reference_ub_median = rounded_median(reference_ub)
                output.append(
                    {
                        "operator": operator,
                        "variable": variable,
                        "value": value,
                        "reference_value": reference_value,
                        "controlled_slice": spec["controlled_slice"],
                        "matched_samples": len(matched_keys),
                        "latency_ms_median": latency_median,
                        "latency_delta_pct": percent_delta(
                            latency_median, reference_latency_median
                        ),
                        "required_ub_kib_median": ub_median,
                        "ub_delta_pct": percent_delta(ub_median, reference_ub_median),
                        "latency_ms_mean": rounded_mean(candidate_latency),
                        "required_ub_kib_mean": rounded_mean(candidate_ub),
                        "reference_latency_ms_median": reference_latency_median,
                        "reference_ub_kib_median": reference_ub_median,
                        "run_id": run["run_id"],
                    }
                )
    return output


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_supported_markdown(path: Path, rows: list[dict]) -> None:
    columns = [
        "operator",
        "depth",
        "intra_cache_num",
        "multibuffer_num",
        "vf_merge_level",
        "latency_ms",
        "required_ub_kib",
        "run_id",
    ]
    labels = [
        "operator",
        "CV depth",
        "DynamicCV intra cache",
        "ordinary buffers",
        "VF merge",
        "latency_ms",
        "UB_KiB",
        "run_id",
    ]
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(labels)) + " |"]
    for row in rows:
        values = [row.get(column) for column in columns]
        lines.append("| " + " | ".join("" if value is None else str(value) for value in values) + " |")
    if not rows:
        lines.append("| _no supported configurations_ |  |  |  |  |  |  |  |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_percent(value) -> str:
    if value is None:
        return ""
    return f"{float(value):+.3f}%"


def format_number(value, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_effects_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Controlled effects from the latest complete sweeps",
        "",
        "![Controlled latency and UB effects](effects.svg)",
        "",
        "Negative latency delta means faster; positive UB delta means more UB. "
        "Every delta uses matched configurations and the lowest value of that "
        "variable as its reference. No configuration is ranked or selected.",
    ]
    active_variables = {row["variable"] for row in rows}
    for spec in EFFECT_SPECS:
        variable = spec["variable"]
        if variable not in active_variables:
            continue
        selected = [row for row in rows if row["variable"] == variable]
        lines.extend(
            [
                "",
                f"## {spec['x_label']}",
                "",
                f"Controlled slice: `{spec['controlled_slice']}`.",
                "",
                "| operator | value | matched | latency median (ms) | latency Δ | UB median (KiB) | UB Δ |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in selected:
            lines.append(
                "| "
                + " | ".join(
                    [
                        operator_label(row["operator"]),
                        str(row["value"]),
                        str(row["matched_samples"]),
                        format_number(row["latency_ms_median"]),
                        format_percent(row["latency_delta_pct"]),
                        format_number(row["required_ub_kib_median"], 3),
                        format_percent(row["ub_delta_pct"]),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_effects_svg(path: Path, rows: list[dict]) -> None:
    width = 1200
    operators = sorted({row["operator"] for row in rows})
    legend_columns = 3
    legend_rows = max(1, (len(operators) + legend_columns - 1) // legend_columns)
    legend_row_height = 21
    top = 84 + legend_rows * legend_row_height
    active_variables = {row["variable"] for row in rows}
    active_specs = [
        spec for spec in EFFECT_SPECS if spec["variable"] in active_variables
    ]
    height = top + len(active_specs) * 245 + 30
    outer_x = 45
    row_height = 245
    column_gap = 40
    panel_width = (width - 2 * outer_x - column_gap) / 2
    panel_height = 215
    plot_left = 70
    plot_right = 18
    plot_top = 42
    plot_bottom = 42
    metrics = [
        ("latency_delta_pct", "Latency Δ vs reference (%)"),
        ("ub_delta_pct", "UB Δ vs reference (%)"),
    ]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" fill="#111827">',
        '<text x="45" y="32" font-size="22" font-weight="700">Controlled effects of the three compiler variables</text>',
        '<text x="45" y="55" font-size="12" fill="#4b5563">Matched comparisons; negative latency is faster, positive UB uses more memory</text>',
    ]
    legend_start_x = 520
    legend_width = 215
    for index, operator in enumerate(operators):
        legend_x = legend_start_x + (index % legend_columns) * legend_width
        legend_y = 50 + (index // legend_columns) * legend_row_height
        color = operator_color(operator)
        label = html.escape(operator_label(operator))
        svg.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{legend_x + 12}" cy="{legend_y}" r="4" fill="{color}"/>',
                f'<text x="{legend_x + 31}" y="{legend_y + 4}" font-size="12">{label}</text>',
            ]
        )

    for row_index, spec in enumerate(active_specs):
        variable = spec["variable"]
        variable_rows = [row for row in rows if row["variable"] == variable]
        x_values = sorted({int(row["value"]) for row in variable_rows})
        for column_index, (metric, metric_title) in enumerate(metrics):
            panel_x = outer_x + column_index * (panel_width + column_gap)
            panel_y = top + row_index * row_height
            chart_x = panel_x + plot_left
            chart_y = panel_y + plot_top
            chart_width = panel_width - plot_left - plot_right
            chart_height = panel_height - plot_top - plot_bottom
            values = [
                float(row[metric])
                for row in variable_rows
                if row.get(metric) is not None
            ]
            low = min([0.0, *values]) if values else -1.0
            high = max([0.0, *values]) if values else 1.0
            span = high - low
            if span < 1.0:
                low -= (1.0 - span) / 2
                high += (1.0 - span) / 2
                span = high - low
            padding = span * 0.12
            low -= padding
            high += padding

            def x_pos(value: int) -> float:
                if len(x_values) <= 1:
                    return chart_x + chart_width / 2
                index = x_values.index(value)
                return chart_x + index * chart_width / (len(x_values) - 1)

            def y_pos(value: float) -> float:
                return chart_y + (high - value) * chart_height / (high - low)

            title = f"{spec['x_label']} · {metric_title}"
            svg.extend(
                [
                    f'<text x="{panel_x}" y="{panel_y + 18}" font-size="14" font-weight="600">{html.escape(title)}</text>',
                    f'<rect x="{chart_x:.2f}" y="{chart_y:.2f}" width="{chart_width:.2f}" height="{chart_height:.2f}" fill="none" stroke="#d1d5db"/>',
                ]
            )
            for tick_index in range(5):
                tick = low + (high - low) * tick_index / 4
                y = y_pos(tick)
                svg.extend(
                    [
                        f'<line x1="{chart_x:.2f}" y1="{y:.2f}" x2="{chart_x + chart_width:.2f}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                        f'<text x="{chart_x - 8:.2f}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#4b5563">{tick:.1f}</text>',
                    ]
                )
            if low <= 0 <= high:
                zero_y = y_pos(0)
                svg.append(
                    f'<line x1="{chart_x:.2f}" y1="{zero_y:.2f}" x2="{chart_x + chart_width:.2f}" y2="{zero_y:.2f}" stroke="#6b7280" stroke-dasharray="4 3"/>'
                )
            for value in x_values:
                x = x_pos(value)
                svg.extend(
                    [
                        f'<line x1="{x:.2f}" y1="{chart_y + chart_height:.2f}" x2="{x:.2f}" y2="{chart_y + chart_height + 5:.2f}" stroke="#6b7280"/>',
                        f'<text x="{x:.2f}" y="{chart_y + chart_height + 20:.2f}" text-anchor="middle" font-size="11">{value}</text>',
                    ]
                )
            for operator in operators:
                series = sorted(
                    (
                        row
                        for row in variable_rows
                        if row["operator"] == operator and row.get(metric) is not None
                    ),
                    key=lambda row: row["value"],
                )
                if not series:
                    continue
                color = operator_color(operator)
                points = [
                    (x_pos(int(row["value"])), y_pos(float(row[metric])), row)
                    for row in series
                ]
                svg.append(
                    '<polyline fill="none" stroke="{}" stroke-width="2.5" points="{}"/>'.format(
                        color, " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
                    )
                )
                for x, y, row in points:
                    tooltip = html.escape(
                        f"{operator_label(operator)}: {variable}={row['value']}, {metric}={row[metric]:+.3f}%"
                    )
                    svg.append(
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"><title>{tooltip}</title></circle>'
                    )
            svg.append(
                f'<text x="{chart_x + chart_width / 2:.2f}" y="{panel_y + panel_height - 2:.2f}" text-anchor="middle" font-size="11" fill="#4b5563">{html.escape(spec["x_label"])}</text>'
            )
    svg.extend(["</g>", "</svg>"])
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"results directory does not exist: {results_dir}")
    output_dir = (args.output_dir or results_dir / "latest-summary").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    latest = find_latest_runs(results_dir)
    if not latest:
        raise SystemExit(f"no complete full-sweep result artifacts found under {results_dir}")
    rows = supported_rows(latest)
    effects = effect_rows(latest)
    per_operator = [summarize_run(run) for _, run in sorted(latest.items())]
    summary = {
        "selected_operator_count": len(latest),
        "supported_row_count": len(rows),
        "operators": per_operator,
        "controlled_effect_row_count": len(effects),
        "controlled_effect_method": {
            spec["variable"]: spec["controlled_slice"] for spec in EFFECT_SPECS
        },
        "notes": [
            "Supported means measured, correctness passed, latency present, and nonzero UB present.",
            "A3 varies static CV `depth` with DynamicCV disabled; A5 enables "
            "DynamicCV and varies `intra_cache_num` while fixing inter/load cache counts to 1.",
            "Effect deltas use matched controlled slices rather than confounded marginal averages.",
            "No configuration is ranked and no winner is selected.",
        ],
    }

    write_csv(output_dir / "supported.csv", rows, SUPPORTED_FIELDS)
    write_supported_markdown(output_dir / "supported.md", rows)
    write_csv(output_dir / "effects.csv", effects, EFFECT_FIELDS)
    write_effects_markdown(output_dir / "effects.md", effects)
    write_effects_svg(output_dir / "effects.svg", effects)
    (output_dir / "effects.json").write_text(
        json.dumps(effects, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for run in latest.values():
        print(f"latest {run['operator']}: {run['result_dir']}")
    print(f"supported_rows={len(rows)}")
    print(f"controlled_effect_rows={len(effects)}")
    print(f"output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
