#!/usr/bin/env python3
"""Compare main and main-dev PlanComputeBlock snapshots."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

OPERATORS = ("fused_attention", "flash_attention_npu_v8", "hstu_attention", "unified_attention")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text: str) -> str:
    text = re.sub(r"/tmp/triton-ascend-[^\"\s)]+", "<WORKTREE>", text)
    text = re.sub(r"/private/tmp/triton-ascend-[^\"\s)]+", "<WORKTREE>", text)
    text = re.sub(r"loc\(\"[^\"]+\":\d+:\d+\)", "loc(<SOURCE>)", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip() + "\n"


def operation_counts(text: str) -> Counter[str]:
    quoted = re.findall(r'^\s*(?:%[^=]+\s*=\s*)?"([A-Za-z0-9_.]+)"\(', text, re.MULTILINE)
    bare = re.findall(r"^\s*(?:%[^=]+\s*=\s*)?([a-z][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)\b", text, re.MULTILINE)
    return Counter(quoted + bare)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows = []
    for operator in OPERATORS:
        main_path = args.root / "main" / operator / "after-plan-compute-block.mlir"
        dev_path = args.root / "main-dev" / operator / "after-plan-compute-block.mlir"
        left = main_path.read_text()
        right = dev_path.read_text()
        left_norm, right_norm = normalize(left), normalize(right)
        counts_left, counts_right = operation_counts(left_norm), operation_counts(right_norm)
        diff = "".join(
            difflib.unified_diff(
                left_norm.splitlines(keepends=True),
                right_norm.splitlines(keepends=True),
                fromfile=f"main/{operator}",
                tofile=f"main-dev/{operator}",
            ))
        diff_path = args.root / "diffs" / f"{operator}.diff"
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(diff)
        rows.append({
            "operator": operator,
            "raw_equal": left == right,
            "normalized_equal": left_norm == right_norm,
            "main_sha256": sha256(left),
            "main_dev_sha256": sha256(right),
            "main_lines": len(left.splitlines()),
            "main_dev_lines": len(right.splitlines()),
            "main_operation_count": sum(counts_left.values()),
            "main_dev_operation_count": sum(counts_right.values()),
            "operation_count_delta": dict(sorted((counts_right - counts_left).items())),
            "operation_count_removed": dict(sorted((counts_left - counts_right).items())),
            "diff_lines": len(diff.splitlines()),
            "diff_path": str(diff_path.relative_to(args.root)),
        })

    (args.root / "comparison.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    header = "operator,raw_equal,normalized_equal,main_lines,main_dev_lines,main_ops,main_dev_ops,diff_lines"
    lines = [header]
    for row in rows:
        lines.append(f"{row['operator']},{row['raw_equal']},{row['normalized_equal']},"
                     f"{row['main_lines']},{row['main_dev_lines']},"
                     f"{row['main_operation_count']},{row['main_dev_operation_count']},{row['diff_lines']}")
    (args.root / "comparison.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
