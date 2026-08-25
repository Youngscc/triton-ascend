#!/usr/bin/env python3
"""Compare UB usage for matching parameter sets in two experiment CSV files."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys

DYNAMIC_CV_AXES = ("buf_slot_num_of_veccore", "intra_cache_num")
PARAMETER_FIELDS = (
    "enable_dynamic_cv_pipeline",
    "enable_auto_multi_buffer",
    "multibuffer_num",
    "vf_merge_level",
)
IDENTITY_FIELDS = ("算子", "operator")
UB_FIELDS = {
    "UB使用_KiB": Decimal(8192),
    "required_ub_kib": Decimal(8192),
    "required_ub_bytes": Decimal(8),
    "required_ub_bits": Decimal(1),
}


class ComparisonError(ValueError):
    pass


def normalize_parameter(value: str) -> str:
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered in {"true", "yes"}:
        return "true"
    if lowered in {"false", "no"}:
        return "false"
    if lowered == "off" or not normalized:
        return lowered
    try:
        number = Decimal(normalized)
    except InvalidOperation:
        return normalized
    return format(number.normalize(), "f")


def select_axis(fields: set[str]) -> tuple[str, str]:
    dynamic_axes = [field for field in DYNAMIC_CV_AXES if field in fields]
    if "depth" in fields:
        if dynamic_axes:
            raise ComparisonError("a CSV cannot contain both depth and a DynamicCV first-axis column")
        return "depth", "depth"
    if len(dynamic_axes) == 1:
        return "dynamic_cv_slots", dynamic_axes[0]
    if len(dynamic_axes) > 1:
        raise ComparisonError("a CSV contains multiple DynamicCV first-axis columns")
    raise ComparisonError("a CSV must contain depth, intra_cache_num, or buf_slot_num_of_veccore")


def select_columns(first_fields: list[str],
                   second_fields: list[str]) -> tuple[list[str], list[str], list[str], str, str]:
    first = set(first_fields)
    second = set(second_fields)
    canonical_axis, first_axis = select_axis(first)
    second_canonical_axis, second_axis = select_axis(second)
    if canonical_axis != second_canonical_axis:
        raise ComparisonError("the two CSV files use different static and DynamicCV first axes")

    missing = [field for field in PARAMETER_FIELDS if field not in first or field not in second]
    if missing:
        raise ComparisonError(f"missing parameter columns: {', '.join(missing)}")

    identity = [field for field in IDENTITY_FIELDS if field in first and field in second]
    key_fields = [
        *identity, "enable_dynamic_cv_pipeline", canonical_axis, "enable_auto_multi_buffer", "multibuffer_num",
        "vf_merge_level"
    ]
    first_key_fields = [
        *identity, "enable_dynamic_cv_pipeline", first_axis, "enable_auto_multi_buffer", "multibuffer_num",
        "vf_merge_level"
    ]
    second_key_fields = [
        *identity, "enable_dynamic_cv_pipeline", second_axis, "enable_auto_multi_buffer", "multibuffer_num",
        "vf_merge_level"
    ]
    first_ub = next((field for field in UB_FIELDS if field in first), "")
    second_ub = next((field for field in UB_FIELDS if field in second), "")
    if not first_ub or not second_ub:
        raise ComparisonError("both CSV files must contain a UB usage column")
    return key_fields, first_key_fields, second_key_fields, first_ub, second_ub


def parse_ub(value: str, field: str, path: Path, row_number: int) -> Decimal | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return Decimal(normalized) * UB_FIELDS[field]
    except InvalidOperation as error:
        raise ComparisonError(f"{path}: row {row_number} has invalid {field} value {value!r}") from error


def load_rows(path: Path, key_fields: list[str], ub_field: str) -> dict[tuple[str, ...], dict]:
    rows: dict[tuple[str, ...], dict] = {}
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError as error:
        raise ComparisonError(f"cannot open {path}: {error}") from error

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ComparisonError(f"{path}: missing CSV header")
        for row_number, row in enumerate(reader, 2):
            key = tuple(normalize_parameter(row.get(field, "")) for field in key_fields)
            if key in rows:
                formatted = format_key(key_fields, key)
                raise ComparisonError(f"{path}: duplicate parameter set at row {row_number}: {formatted}")
            rows[key] = {
                "row_number": row_number,
                "ub_bits": parse_ub(row.get(ub_field, ""), ub_field, path, row_number),
                "ub_value": row.get(ub_field, "").strip(),
                "ub_field": ub_field,
            }
    return rows


def read_header(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle).fieldnames or ())
    except OSError as error:
        raise ComparisonError(f"cannot open {path}: {error}") from error


def format_key(fields: list[str], key: tuple[str, ...]) -> str:
    return " ".join(f"{field}={value or '<empty>'}" for field, value in zip(fields, key))


def compare(first_path: Path, second_path: Path) -> dict:
    key_fields, first_key_fields, second_key_fields, first_ub, second_ub = select_columns(
        read_header(first_path), read_header(second_path))
    first_rows = load_rows(first_path, first_key_fields, first_ub)
    second_rows = load_rows(second_path, second_key_fields, second_ub)
    matching_keys = sorted(first_rows.keys() & second_rows.keys())

    different = []
    same = 0
    missing_ub = 0
    for key in matching_keys:
        first = first_rows[key]
        second = second_rows[key]
        if first["ub_bits"] is None or second["ub_bits"] is None:
            missing_ub += 1
        elif first["ub_bits"] == second["ub_bits"]:
            same += 1
        else:
            different.append((key, first, second))

    return {
        "key_fields": key_fields,
        "matching": len(matching_keys),
        "same": same,
        "different": different,
        "missing_ub": missing_ub,
        "only_first": len(first_rows.keys() - second_rows.keys()),
        "only_second": len(second_rows.keys() - first_rows.keys()),
    }


def print_result(result: dict) -> None:
    for key, first, second in result["different"]:
        print("UB_DIFFERENT " + format_key(result["key_fields"], key) +
              f" first={first['ub_value'] or '<missing>'}({first['ub_field']})" +
              f" second={second['ub_value'] or '<missing>'}({second['ub_field']})")
    print("SUMMARY "
          f"matching={result['matching']} same_ub={result['same']} "
          f"different_ub={len(result['different'])} missing_ub={result['missing_ub']} "
          f"only_in_first={result['only_first']} only_in_second={result['only_second']}")
    if result["different"]:
        conclusion = "UB_DIFFERENT"
    elif result["same"]:
        conclusion = "UB_SAME"
    else:
        conclusion = "NO_COMPARABLE_UB"
    print(f"RESULT {conclusion}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare UB usage only for matching parameter sets in two experiment CSV files.")
    parser.add_argument("first_csv", type=Path)
    parser.add_argument("second_csv", type=Path)
    args = parser.parse_args(argv)

    try:
        result = compare(args.first_csv, args.second_csv)
    except ComparisonError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2
    print_result(result)
    if result["different"]:
        return 1
    return 0 if result["same"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
