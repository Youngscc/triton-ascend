#!/usr/bin/env python3
"""Host-only VFMerge sensitivity screen for a directory of TTAdapter inputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import difflib
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


BEFORE_MARKER = "// -----// IR Dump Before MergeVecScope (hfusion-merge-vf) //----- //"
AFTER_MARKER = "// -----// IR Dump After MergeVecScope (hfusion-merge-vf) //----- //"
UB_PATTERN = re.compile(r"Allocated UB size = (\d+) bits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--target", default="Ascend910B1")
    return parser.parse_args()


def extract_module(text: str) -> str | None:
    match = re.search(r"(?m)^module(?:\s|$)", text)
    if not match:
        return None
    start = match.start()
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(brace, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def normalize_ir(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip() + "\n"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_pass_pairs(stderr: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    cursor = 0
    while True:
        before_at = stderr.find(BEFORE_MARKER, cursor)
        if before_at < 0:
            break
        after_at = stderr.find(AFTER_MARKER, before_at + len(BEFORE_MARKER))
        if after_at < 0:
            break
        next_before = stderr.find(BEFORE_MARKER, after_at + len(AFTER_MARKER))
        before = extract_module(stderr[before_at + len(BEFORE_MARKER) : after_at])
        after_region = stderr[
            after_at + len(AFTER_MARKER) : next_before if next_before >= 0 else None
        ]
        after = extract_module(after_region)
        if before is not None and after is not None:
            pairs.append((normalize_ir(before), normalize_ir(after)))
        cursor = after_at + len(AFTER_MARKER)
    return pairs


def compiler_command(
    compiler: Path, adapter: Path, level: int, output: Path, target: str
) -> list[str]:
    command = [
        str(compiler),
        str(adapter),
        "--enable-hfusion-compile=true",
        "--enable-hivm-compile=true",
        "--enable-triton-kernel-compile=true",
        f"--target={target}",
        f"--enable-vf-merge-level={level}",
        "--enable-print-memory-allocated-size=true",
        "--enable-auto-multi-buffer=true",
        "--limit-auto-multi-buffer-buffer=no-limit",
        "--set-local-multibuffer=1",
        "--set-workspace-multibuffer=2",
        "--enable-tuning-mode=true",
        "--mlir-disable-threading",
    ]
    if level:
        command.extend(
            [
                "--mlir-print-ir-before=hfusion-merge-vf",
                "--mlir-print-ir-after=hfusion-merge-vf",
            ]
        )
    command.extend(["-o", str(output)])
    return command


def run_one(
    compiler: Path,
    adapter: Path,
    level: int,
    output_dir: Path,
    timeout: float,
    target: str,
) -> dict[str, object]:
    started = time.monotonic()
    output = output_dir / "objects" / f"{adapter.stem}.vf{level}.o"
    command = compiler_command(compiler, adapter, level, output, target)
    environment = os.environ.copy()
    environment["BISHENGIR_PLAN_MEMORY_FORCE_SEED"] = "0"
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=environment,
            timeout=timeout,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        returncode = 124
        timed_out = True

    ub_values = [int(value) for value in UB_PATTERN.findall(stdout + "\n" + stderr)]
    positive_ub_values = [value for value in ub_values if value > 0]
    pairs = parse_pass_pairs(stderr) if level else []
    affected = [before != after for before, after in pairs]
    changed_lines = [
        sum(1 for _ in difflib.unified_diff(before.splitlines(), after.splitlines()))
        if changed
        else 0
        for (before, after), changed in zip(pairs, affected)
    ]
    vector_functions_before = [before.count("hivm.vector_function") for before, _ in pairs]
    vector_functions_after = [after.count("hivm.vector_function") for _, after in pairs]
    diagnostic_lines = [
        line
        for line in stderr.splitlines()
        if "Cannot find hivmc" not in line
        and "Failed to run `hivmc --version`" not in line
        and "Failed to detect hivmc version" not in line
        and not line.startswith("// -----// IR Dump")
    ]
    # IR dumps dominate stderr. Keep only actual diagnostic-looking lines.
    diagnostics = [
        line
        for line in diagnostic_lines
        if ("error:" in line.lower() or line.startswith("[ERROR]"))
        and "External hivmc run fails" not in line
        and line != "[ERROR] Failed to run BiShengIR pipeline"
    ]
    return {
        "adapter": adapter.name,
        "vf_merge_level": level,
        "elapsed_s": round(time.monotonic() - started, 6),
        "returncode": returncode,
        "timed_out": timed_out,
        "pass_observed": bool(pairs) if level else False,
        "pass_attempts": len(pairs),
        "affected": any(affected) if pairs else False,
        "affected_attempts": sum(affected),
        "changed_diff_lines_max": max(changed_lines, default=0),
        "vector_functions_before_max": max(vector_functions_before, default=0),
        "vector_functions_after_max": max(vector_functions_after, default=0),
        "before_sha256": pairs[-1] and digest(pairs[-1][0]) if pairs else "",
        "after_sha256": pairs[-1] and digest(pairs[-1][1]) if pairs else "",
        "ub_reported_bits": max(ub_values) if ub_values else None,
        "ub_bits": max(positive_ub_values) if positive_ub_values else None,
        "ub_bytes": max(positive_ub_values) // 8 if positive_ub_values else None,
        "ub_kib": round(max(positive_ub_values) / 8192, 6) if positive_ub_values else None,
        "ub_line_observed": bool(ub_values),
        "diagnostic": " | ".join(diagnostics[-3:])[:1200],
        "command": command,
    }


def main() -> int:
    args = parse_args()
    compiler = args.compiler.resolve()
    adapter_dir = args.adapter_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "objects").mkdir(exist_ok=True)
    adapters = sorted(adapter_dir.glob("*.ttadapter"))
    tasks = [(adapter, level) for adapter in adapters for level in (0, 1, 2)]
    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_one, compiler, adapter, level, output_dir, args.timeout, args.target
            ): (adapter, level)
            for adapter, level in tasks
        }
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            adapter, level = futures[future]
            try:
                row = future.result()
            except Exception as error:  # preserve every requested observation
                row = {
                    "adapter": adapter.name,
                    "vf_merge_level": level,
                    "elapsed_s": None,
                    "returncode": None,
                    "timed_out": False,
                    "pass_observed": False,
                    "pass_attempts": 0,
                    "affected": False,
                    "affected_attempts": 0,
                    "changed_diff_lines_max": 0,
                    "vector_functions_before_max": 0,
                    "vector_functions_after_max": 0,
                    "before_sha256": "",
                    "after_sha256": "",
                    "ub_reported_bits": None,
                    "ub_bits": None,
                    "ub_bytes": None,
                    "ub_kib": None,
                    "ub_line_observed": False,
                    "diagnostic": f"runner error: {error}",
                    "command": [],
                }
            rows.append(row)
            completed_count += 1
            if completed_count % 25 == 0 or completed_count == len(tasks):
                print(f"completed {completed_count}/{len(tasks)}", flush=True)

    rows.sort(key=lambda row: (str(row["adapter"]), int(row["vf_merge_level"])))
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")

    csv_fields = [key for key in rows[0] if key != "command"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in rows)

    per_adapter = []
    for adapter in adapters:
        values = {int(row["vf_merge_level"]): row for row in rows if row["adapter"] == adapter.name}
        per_adapter.append(
            {
                "adapter": adapter.name,
                "level1_affected": values[1]["affected"],
                "level2_affected": values[2]["affected"],
                "any_affected": values[1]["affected"] or values[2]["affected"],
                "level0_ub_bits": values[0]["ub_bits"],
                "level1_ub_bits": values[1]["ub_bits"],
                "level2_ub_bits": values[2]["ub_bits"],
                "level0_ub_kib": values[0]["ub_kib"],
                "level1_ub_kib": values[1]["ub_kib"],
                "level2_ub_kib": values[2]["ub_kib"],
                "level1_pass_observed": values[1]["pass_observed"],
                "level2_pass_observed": values[2]["pass_observed"],
                "level1_vector_functions_before": values[1][
                    "vector_functions_before_max"
                ],
                "level2_vector_functions_before": values[2][
                    "vector_functions_before_max"
                ],
                "all_levels_ub_line_observed": all(
                    bool(values[level]["ub_line_observed"])
                    for level in (0, 1, 2)
                ),
                "diagnostic": " | ".join(
                    str(values[level]["diagnostic"])
                    for level in (0, 1, 2)
                    if values[level]["diagnostic"]
                )[:1200],
            }
        )
    with (output_dir / "operators.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_adapter[0]))
        writer.writeheader()
        writer.writerows(per_adapter)

    summary = {
        "compiler": str(compiler),
        "adapter_dir": str(adapter_dir),
        "target": args.target,
        "adapter_count": len(adapters),
        "compile_count": len(rows),
        "affected_any_count": sum(bool(row["any_affected"]) for row in per_adapter),
        "affected_level1_count": sum(bool(row["level1_affected"]) for row in per_adapter),
        "affected_level2_count": sum(bool(row["level2_affected"]) for row in per_adapter),
        "all_levels_ub_line_observed_count": sum(
            bool(row["all_levels_ub_line_observed"]) for row in per_adapter
        ),
        "all_levels_nonzero_ub_count": sum(
            all(row[f"level{level}_ub_bits"] is not None for level in (0, 1, 2))
            for row in per_adapter
        ),
        "criterion": "normalized IR before/after the actual hfusion-merge-vf pass",
        "ub_role": "reported independently; never used to select affected operators",
        "fixed_controls": {
            "workspace_multibuffer": 2,
            "ordinary_local_multibuffer": 1,
            "ordinary_auto_multibuffer": True,
            "mix_multibuffer_strategy": "no-limit",
            "plan_memory_seed": 0,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
