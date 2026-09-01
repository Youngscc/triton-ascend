"""Build the compact SSA, def-use, loop, and compute-block graph."""

from __future__ import annotations

import re
from typing import Optional

from ..model_types import FunctionArgument, IRGraph, LifeInterval, LoopRegion, Operation, UnsupportedModelError

SSA_TOKEN_PATTERN = r"%[A-Za-z0-9_.$-]+(?:#[0-9]+)?"
SSA_TOKEN_RE = re.compile(SSA_TOKEN_PATTERN)
DEFINITION_RE = re.compile(r"^\s*(%[A-Za-z0-9_.$-]+)(?::([0-9]+))?\s*=\s*(.*)$")
ATTRIBUTE_INT_RE = re.compile(r"([A-Za-z0-9_.]+)\s*=\s*(-?[0-9]+)\s*:\s*i[0-9]+")
CORE_TYPE_RE = re.compile(r'ssbuffer\.core_type\s*=\s*"([A-Z, ]+)"')
BLOCK_ID_RE = re.compile(r"ssbuffer\.block_id\s*=\s*([0-9]+)")
STATIC_MEMREF_RE = re.compile(r"memref<((?:[0-9]+x)+(?:bf16|f[0-9]+|i[0-9]+))(?=[,>])")
MEMREF_ELEMENT_RE = re.compile(r"memref<[^,>]*x(bf16|f[0-9]+|i[0-9]+)(?=[,>])")
STATIC_SHAPED_TYPE_RE = re.compile(r"(?:tensor|memref|vector)<((?:[0-9]+x)+(?:bf16|f[0-9]+|i[0-9]+))(?=[,>])")

TRANSPARENT_SOURCE_OPS = {
    "arith.bitcast",
    "arith.extf",
    "arith.truncf",
    "bufferization.to_tensor",
    "tensor.cast",
    "tensor.collapse_shape",
    "tensor.expand_shape",
    "tensor.extract_slice",
}

TRANSPARENT_MEMREF_OPS = {
    "memref.cast",
    "memref.subview",
}


def _strip_strings(text: str) -> str:
    result: list[str] = []
    escaped = False
    in_string = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            result.append(" ")
        else:
            if char == '"':
                in_string = True
                result.append(" ")
            else:
                result.append(char)
    return "".join(result)


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    chunks: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0, "<": 0}
    closers = {")": "(", "]": "[", "}": "{", ">": "<"}
    in_string = False
    escaped = False
    for index, char in enumerate(text):
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
        elif char in depths:
            depths[char] += 1
        elif char in closers:
            opener = closers[char]
            depths[opener] = max(0, depths[opener] - 1)
        elif char == delimiter and not any(depths.values()):
            chunks.append(text[start:index].strip())
            start = index + 1
    chunks.append(text[start:].strip())
    return [chunk for chunk in chunks if chunk]


def _matching_paren(text: str, open_index: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_index, len(text)):
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
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    raise UnsupportedModelError("cannot find the end of the func.func argument list")


def _first_ssa_operand(operation: Operation) -> Optional[str]:
    remainder = operation.text[len(operation.name):]
    match = SSA_TOKEN_RE.search(remainder)
    return match.group(0) if match else None


def _operation_name(text: str) -> str:
    match = re.match(r"([A-Za-z0-9_.]+)", text.strip())
    return match.group(1) if match else "unknown"


def _static_memref_shape_and_dtype(text: str) -> tuple[tuple[int, ...], str]:
    match = STATIC_MEMREF_RE.search(text)
    if not match:
        raise UnsupportedModelError(f"cannot project a static memref type from: {text[:160]}")
    pieces = match.group(1).split("x")
    return tuple(int(piece) for piece in pieces[:-1]), pieces[-1]


def _argument_memref_dtype(argument: FunctionArgument) -> str:
    match = MEMREF_ELEMENT_RE.search(argument.type_text)
    if not match:
        raise UnsupportedModelError(f"cannot read the memref element type of argument {argument.name}")
    return match.group(1)


def build_ir_graph(text: str) -> IRGraph:
    lines = text.splitlines()
    definitions: dict[str, Operation] = {}
    operations: list[Operation] = []
    uses: dict[str, list[int]] = {}
    materialize_lines: list[tuple[int, str, tuple[str, ...]]] = []
    copy_lines: list[tuple[int, str, tuple[str, ...]]] = []
    brace_depth = 0
    # (body brace depth, kind, loop id).  The loop id is its one-based source
    # line, which is stable for one normalized PlanComputeBlock input.
    loop_stack: list[tuple[int, str, int]] = []
    loop_builders: dict[int, dict[str, object]] = {}
    block_ids: set[int] = set()
    vector_blocks: set[int] = set()
    cube_blocks: set[int] = set()

    for line_number, line in enumerate(lines, start=1):
        while loop_stack and brace_depth < loop_stack[-1][0]:
            _threshold, _kind, finished_id = loop_stack.pop()
            loop_builders[finished_id]["line_end"] = line_number - 1
        loop_kinds = tuple(kind for _threshold, kind, _loop_id in loop_stack)
        loop_depth = len(loop_kinds)
        loop_id = loop_stack[-1][2] if loop_stack else None

        definition = DEFINITION_RE.match(line)
        if definition:
            result, count_text, operation_text = definition.groups()
            result_count = int(count_text or "1")
            core_match = CORE_TYPE_RE.search(operation_text)
            block_match = BLOCK_ID_RE.search(operation_text)
            core_type = core_match.group(1) if core_match else None
            block_id = int(block_match.group(1)) if block_match else None
            operation = Operation(
                result=result,
                result_count=result_count,
                name=_operation_name(operation_text),
                operands=tuple(SSA_TOKEN_RE.findall(operation_text)),
                text=operation_text,
                line_number=line_number,
                loop_depth=loop_depth,
                loop_kinds=loop_kinds,
                core_type=core_type,
                block_id=block_id,
                loop_id=loop_id,
            )
            definitions[result] = operation
            operations.append(operation)
            for result_index in range(result_count):
                definitions[f"{result}#{result_index}"] = operation
            if block_id is not None:
                block_ids.add(block_id)
                if core_type == "VECTOR":
                    vector_blocks.add(block_id)
                elif core_type == "CUBE":
                    cube_blocks.add(block_id)
                if loop_id is not None:
                    loop_builders[loop_id]["block_ids"].add(block_id)
                    if core_type:
                        loop_builders[loop_id]["core_types"].add(core_type)

        if "bufferization.materialize_in_destination" in line:
            materialize_lines.append((line_number, line, loop_kinds))
        if re.search(r"\bmemref\.copy\b", line):
            copy_lines.append((line_number, line, loop_kinds))

        use_text = definition.group(3) if definition else line
        for operand in SSA_TOKEN_RE.findall(use_text):
            uses.setdefault(operand, []).append(line_number)

        cleaned = _strip_strings(line)
        brace_delta = cleaned.count("{") - cleaned.count("}")
        loop_match = re.search(r"\b(scf\.(?:for|while|parallel|forall))\b", cleaned)
        if loop_match and brace_delta > 0:
            kind = loop_match.group(1)
            bindings: list[tuple[str, str]] = []
            if kind == "scf.for":
                iter_match = re.search(r"iter_args\((.*?)\)\s*->", line)
            elif kind == "scf.while":
                iter_match = re.search(r"scf\.while\s*\((.*?)\)\s*:", line)
            else:
                iter_match = None
            if iter_match:
                for binding in _split_top_level(iter_match.group(1)):
                    match = re.match(
                        rf"\s*({SSA_TOKEN_PATTERN})\s*=\s*({SSA_TOKEN_PATTERN})",
                        binding,
                    )
                    if match:
                        bindings.append((match.group(1), match.group(2)))
            loop_builders[line_number] = {
                "kind": kind,
                "parent_loop_id": loop_id,
                "line_start": line_number,
                "line_end": len(lines),
                "iter_args": tuple(item[0] for item in bindings),
                "initial_values": tuple(item[1] for item in bindings),
                "yielded_values": (),
                "core_types": set(),
                "block_ids": set(),
            }
            loop_stack.append((brace_depth + brace_delta, kind, line_number))

        # Only the scf.yield directly in the loop body closes its iter_args.
        # Branch yields live at a deeper brace depth and must not overwrite it.
        if (loop_stack and line.lstrip().startswith("scf.yield")
                and brace_depth == loop_stack[-1][0]):
            yield_text = line.split("scf.yield", 1)[1]
            if yield_text.lstrip().startswith("{") and "}" in yield_text:
                yield_text = yield_text.split("}", 1)[1]
            yield_text = yield_text.split(":", 1)[0]
            loop_builders[loop_stack[-1][2]]["yielded_values"] = tuple(
                SSA_TOKEN_RE.findall(yield_text))
        brace_depth += brace_delta
        while loop_stack and brace_depth < loop_stack[-1][0]:
            _threshold, _kind, finished_id = loop_stack.pop()
            loop_builders[finished_id]["line_end"] = line_number

    while loop_stack:
        _threshold, _kind, finished_id = loop_stack.pop()
        loop_builders[finished_id]["line_end"] = len(lines)

    function_lines = [line for line in lines if re.search(r"\bfunc\.func\s+@", line)]
    if len(function_lines) != 1:
        raise UnsupportedModelError(f"profile expects exactly one func.func, found {len(function_lines)}")
    function_line = function_lines[0]
    function_match = re.search(r"\bfunc\.func\s+@([A-Za-z0-9_.$-]+)", function_line)
    if not function_match:
        raise UnsupportedModelError("cannot parse func.func symbol name")
    function_name = function_match.group(1)
    open_index = function_line.index("(", function_line.index("func.func"))
    close_index = _matching_paren(function_line, open_index)
    arguments: dict[str, FunctionArgument] = {}
    for argument_text in _split_top_level(function_line[open_index + 1:close_index]):
        match = re.match(r"\s*(%[A-Za-z0-9_.$-]+)\s*:\s*(.*)", argument_text)
        if not match:
            continue
        name, type_text = match.groups()
        kind_match = re.search(r"tt\.tensor_kind\s*=\s*([01])", type_text)
        tensor_kind = int(kind_match.group(1)) if kind_match else None
        arguments[name] = FunctionArgument(name, type_text, tensor_kind)

    output_arguments = frozenset(argument.name for argument in arguments.values() if argument.tensor_kind == 1)
    input_arguments = frozenset(argument.name for argument in arguments.values() if argument.tensor_kind == 0)
    if not output_arguments:
        raise UnsupportedModelError("func.func has no tt.tensor_kind=1 output argument")

    module_line = next((line for line in lines if line.lstrip().startswith("module attributes")), "")
    module_attributes = {name: int(value) for name, value in ATTRIBUTE_INT_RE.findall(module_line)}
    target_match = re.search(r'hacc\.target\s*=\s*#hacc\.target<"([^"]+)">', module_line)
    target = target_match.group(1) if target_match else ""

    loops = tuple(
        LoopRegion(
            loop_id=loop_id,
            kind=str(data["kind"]),
            parent_loop_id=data["parent_loop_id"],
            line_start=int(data["line_start"]),
            line_end=int(data["line_end"]),
            iter_args=tuple(data["iter_args"]),
            initial_values=tuple(data["initial_values"]),
            yielded_values=tuple(data["yielded_values"]),
            core_types=tuple(sorted(data["core_types"])),
            block_ids=tuple(sorted(data["block_ids"])),
        )
        for loop_id, data in sorted(loop_builders.items())
    )

    return IRGraph(
        lines=tuple(lines),
        definitions=definitions,
        uses={name: tuple(line_numbers)
              for name, line_numbers in uses.items()},
        arguments=arguments,
        output_arguments=output_arguments,
        input_arguments=input_arguments,
        materialize_lines=tuple(materialize_lines),
        copy_lines=tuple(copy_lines),
        module_attributes=module_attributes,
        has_fallback="ssbuffer.fallback" in module_line,
        target=target,
        function_name=function_name,
        block_count=len(block_ids),
        vector_block_count=len(vector_blocks),
        cube_block_count=len(cube_blocks),
        loops=loops,
        operations=tuple(operations),
    )


def _trace_memref_view(
    value: str,
    parsed: IRGraph,
    expected_roots: frozenset[str],
) -> Optional[tuple[str, str, tuple[int, ...], str, str, Optional[int], tuple[str, ...]]]:
    seen: set[str] = set()
    provenance: list[str] = []
    current = value
    while current not in seen:
        seen.add(current)
        operation = parsed.definitions.get(current)
        if operation is None:
            return None
        provenance.append(f"{operation.result}:{operation.name}@{operation.line_number}")
        operand = _first_ssa_operand(operation)
        if operand is None:
            return None
        if operation.name in TRANSPARENT_MEMREF_OPS:
            current = operand
            continue
        if operation.name != "memref.reinterpret_cast":
            return None
        if operand not in expected_roots:
            current = operand
            continue
        sizes_match = re.search(r"sizes:\s*\[([^\]]+)\]", operation.text)
        if not sizes_match:
            return None
        size_tokens = [token.strip() for token in sizes_match.group(1).split(",")]
        if not size_tokens or any(not token.isdigit() for token in size_tokens):
            raise UnsupportedModelError(
                f"dynamic GM view at line {operation.line_number} is outside the supported profile")
        dtype = _argument_memref_dtype(parsed.arguments[operand])
        return (
            operand,
            operation.result,
            tuple(int(token) for token in size_tokens),
            dtype,
            operation.core_type or "",
            operation.block_id,
            tuple(provenance + [f"{operand}:func_argument"]),
        )
    return None


def _trace_to_alloc(value: str, parsed: IRGraph) -> Optional[tuple[Operation, tuple[str, ...]]]:
    seen: set[str] = set()
    provenance: list[str] = []
    current = value
    while current not in seen:
        seen.add(current)
        operation = parsed.definitions.get(current)
        if operation is None:
            return None
        provenance.append(f"{operation.result}:{operation.name}@{operation.line_number}")
        if operation.name == "memref.alloc":
            return operation, tuple(provenance)
        if operation.name not in TRANSPARENT_MEMREF_OPS:
            return None
        operand = _first_ssa_operand(operation)
        if operand is None:
            return None
        current = operand
    return None


def _resolve_gm_load_hint(allocation: str, parsed: IRGraph) -> Optional[int]:
    """Read the compile hint consumed by MarkGMLoadPass when it is present."""

    hint_pattern = re.compile(r"(?:gm_load[^=]*|gm_load_multi_buffer[^=]*)=\s*(-?[0-9]+)")
    for line_number in parsed.uses.get(allocation, ()):
        line = parsed.lines[line_number - 1]
        if "annotation.mark" not in line or "gm_load" not in line:
            continue
        match = hint_pattern.search(line)
        if not match:
            raise UnsupportedModelError(f"cannot parse gm_load hint for {allocation} at line {line_number}")
        return int(match.group(1))
    return None


def _value_lifetime(value: str, definition_line: int, parsed: IRGraph) -> LifeInterval:
    return LifeInterval(definition_line, max(parsed.uses.get(value, (definition_line, ))))
