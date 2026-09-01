"""Find, size, and assign ownership to UB-relevant buffer families."""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional, Sequence

from .build_ir_graph import (
    BLOCK_ID_RE,
    CORE_TYPE_RE,
    SSA_TOKEN_PATTERN,
    SSA_TOKEN_RE,
    TRANSPARENT_SOURCE_OPS,
    _first_ssa_operand,
    _resolve_gm_load_hint,
    _static_memref_shape_and_dtype,
    _trace_memref_view,
    _trace_to_alloc,
    _value_lifetime,
)
from .validate_context import CompilerProfile
from .buffer_geometry import align_up as _align_up
from .buffer_geometry import dtype_bits as _dtype_bits
from .buffer_geometry import project_physical_shape as _project_physical_shape
from ..model_types import (
    BufferAnalysis,
    BufferContribution,
    DynamicCVAnalysis,
    IRGraph,
    LifeInterval,
    RawLoadPath,
    RawStorePath,
    UnsupportedModelError,
)


def _make_contribution(
    *,
    origin: str,
    path_kind: str,
    classification: str,
    logical_shape: tuple[int, ...],
    dtype: str,
    core_type: str,
    block_id: Optional[int],
    owner_loop_kinds: tuple[str, ...],
    lifetime: LifeInterval,
    multiplicity_kind: str,
    fixed_count: int,
    provenance: Sequence[str],
    profile: CompilerProfile,
    reason: str,
) -> BufferContribution:
    physical_shape = _project_physical_shape(logical_shape, core_type, profile)
    element_bits = _dtype_bits(dtype)
    raw_bits = math.prod(physical_shape) * element_bits
    raw_bytes = (raw_bits + 7) // 8
    aligned_bytes = _align_up(raw_bytes, profile.ub_alignment_bytes)
    return BufferContribution(
        origin=origin,
        path_kind=path_kind,
        classification=classification,
        logical_shape=logical_shape,
        physical_shape=physical_shape,
        dtype=dtype,
        element_bits=element_bits,
        raw_bytes=raw_bytes,
        aligned_bytes=aligned_bytes,
        delta_step_bytes=aligned_bytes if classification == "ordinary" else 0,
        block_id=block_id,
        owner_loop_kinds=owner_loop_kinds,
        lifetime=lifetime,
        multiplicity_kind=multiplicity_kind,
        fixed_count=fixed_count,
        provenance=tuple(provenance),
        reason=reason,
    )


def _extract_boundary_paths(parsed: IRGraph) -> tuple[list[RawStorePath], list[RawLoadPath], list[str]]:
    stores: list[RawStorePath] = []
    loads: list[RawLoadPath] = []
    blockers: list[str] = []

    for line_number, line, loop_kinds in parsed.materialize_lines:
        match = re.search(
            rf"materialize_in_destination\s+({SSA_TOKEN_PATTERN})\s+in\s+writable\s+({SSA_TOKEN_PATTERN})",
            line,
        )
        if not match:
            blockers.append(f"cannot parse materialize_in_destination at line {line_number}")
            continue
        source, destination = match.groups()
        view = _trace_memref_view(destination, parsed, parsed.output_arguments)
        if view is None:
            blockers.append(f"cannot trace output GM view for {destination} at line {line_number}")
            continue
        output_argument, destination_view, shape, dtype, core_type, block_id, provenance = view
        line_core_match = CORE_TYPE_RE.search(line)
        line_block_match = BLOCK_ID_RE.search(line)
        effective_core = line_core_match.group(1) if line_core_match else core_type
        effective_block = int(line_block_match.group(1)) if line_block_match else block_id
        source_operation = parsed.definitions.get(source)
        source_line = source_operation.line_number if source_operation else line_number
        stores.append(
            RawStorePath(
                source=source,
                destination=destination,
                output_argument=output_argument,
                destination_view=destination_view,
                logical_shape=shape,
                dtype=dtype,
                core_type=effective_core,
                block_id=effective_block,
                line_number=line_number,
                loop_depth=len(loop_kinds),
                loop_kinds=loop_kinds,
                source_line=source_line,
                provenance=(f"{source}:store_source", *provenance, f"materialize@{line_number}"),
            ))

    for line_number, line, loop_kinds in parsed.copy_lines:
        operands = SSA_TOKEN_RE.findall(line[line.index("memref.copy") + len("memref.copy"):])
        if len(operands) < 2:
            continue
        source, destination = operands[:2]

        output_view = _trace_memref_view(destination, parsed, parsed.output_arguments)
        if output_view is not None:
            blockers.append(
                f"memref.copy GM output path at line {line_number} is outside the materialize-store profile")
            continue

        allocation_trace = _trace_to_alloc(destination, parsed)
        if allocation_trace is None:
            continue
        allocation, allocation_provenance = allocation_trace
        source_view = _trace_memref_view(source, parsed, parsed.input_arguments)
        if source_view is None:
            continue
        input_argument, _view_name, _view_shape, _view_dtype, _core, _block, source_provenance = source_view
        try:
            shape, dtype = _static_memref_shape_and_dtype(allocation.text)
        except UnsupportedModelError as error:
            blockers.append(str(error))
            continue
        loads.append(
            RawLoadPath(
                input_argument=input_argument,
                allocation=allocation.result,
                logical_shape=shape,
                dtype=dtype,
                core_type=allocation.core_type or "",
                block_id=allocation.block_id,
                line_number=line_number,
                loop_kinds=loop_kinds,
                lifetime=_value_lifetime(allocation.result, allocation.line_number, parsed),
                hint_count=_resolve_gm_load_hint(allocation.result, parsed),
                provenance=(*source_provenance, f"memref.copy@{line_number}", *allocation_provenance),
            ))

    return stores, loads, blockers


def _trace_store_source(path: RawStorePath, parsed: IRGraph) -> tuple[bool, tuple[str, ...]]:
    current = path.source
    seen: set[str] = set()
    provenance: list[str] = []
    while current not in seen:
        seen.add(current)
        operation = parsed.definitions.get(current)
        if operation is None:
            return False, tuple(provenance)
        provenance.append(f"{operation.result}:{operation.name}@{operation.line_number}")
        if "ssbuffer.add_from_matmul" in operation.text:
            return True, tuple(provenance)
        if operation.name not in TRANSPARENT_SOURCE_OPS:
            return False, tuple(provenance)
        operand = _first_ssa_operand(operation)
        if operand is None:
            return False, tuple(provenance)
        current = operand
    return False, tuple(provenance)


def _has_supported_loop_chain(loop_kinds: Sequence[str]) -> bool:
    return all(kind in {"scf.for", "scf.while"} for kind in loop_kinds)


def _classify_ownership(
    parsed: IRGraph,
    stores: Iterable[RawStorePath],
    loads: Iterable[RawLoadPath],
    profile: CompilerProfile,
) -> tuple[list[BufferContribution], list[BufferContribution], list[str]]:
    ordinary: list[BufferContribution] = []
    excluded: list[BufferContribution] = []
    blockers: list[str] = []
    seen_ordinary: set[tuple[str, tuple[int, ...], str]] = set()
    seen_excluded: set[tuple[str, str]] = set()

    for path in stores:
        if path.core_type != "VECTOR":
            blockers.append(f"output store {path.output_argument} at line {path.line_number} is not VECTOR")
            continue
        if not _has_supported_loop_chain(path.loop_kinds):
            blockers.append(f"output store {path.output_argument} has unsupported loop chain "
                            f"{','.join(path.loop_kinds)} at line {path.line_number}")
            continue
        is_fixpipe, source_provenance = _trace_store_source(path, parsed)
        lifetime = LifeInterval(min(path.source_line, path.line_number), path.line_number)
        if is_fixpipe:
            key = (path.output_argument, "fixpipe_like_store")
            if key not in seen_excluded:
                excluded.append(
                    _make_contribution(
                        origin=path.output_argument,
                        path_kind="gm_store",
                        classification="fixpipe_like",
                        logical_shape=path.logical_shape,
                        dtype=path.dtype,
                        core_type=path.core_type,
                        block_id=path.block_id,
                        owner_loop_kinds=path.loop_kinds,
                        lifetime=lifetime,
                        multiplicity_kind="single",
                        fixed_count=1,
                        provenance=(*source_provenance, *path.provenance),
                        profile=profile,
                        reason=("store source reaches ssbuffer.add_from_matmul through only view/cast ops; "
                                "the suffix lowers it through the tightly-coupled Fixpipe path, so ordinary "
                                "Store marking cannot claim it"),
                    ))
                seen_excluded.add(key)
            continue
        key = (path.output_argument, path.block_id, path.logical_shape, path.dtype)
        if key in seen_ordinary:
            continue
        ordinary.append(
            _make_contribution(
                origin=path.output_argument,
                path_kind="gm_store",
                classification="ordinary",
                logical_shape=path.logical_shape,
                dtype=path.dtype,
                core_type=path.core_type,
                block_id=path.block_id,
                owner_loop_kinds=path.loop_kinds,
                lifetime=lifetime,
                multiplicity_kind="ordinary",
                fixed_count=1,
                provenance=(*source_provenance, *path.provenance),
                profile=profile,
                reason=("Vector producer reaches a tensor_kind=1 GM destination inside a supported loop; "
                        "AIV sub-block lowering creates the local UB source of an ordinary hivm.Store, "
                        "which MarkMultiBuffer marks with the requested local count"),
            ))
        seen_ordinary.add(key)

    for path in loads:
        if not _has_supported_loop_chain(path.loop_kinds):
            blockers.append(f"GM load allocation {path.allocation} has unsupported loop chain "
                            f"{','.join(path.loop_kinds)} at line {path.line_number}")
            continue
        if path.core_type not in {"VECTOR", "CUBE"}:
            blockers.append(f"GM load allocation {path.allocation} has no supported core type")
            continue
        if path.core_type == "CUBE":
            classification = "non_ub_cube_gm_load"
            fixed_count = path.hint_count if path.hint_count is not None else profile.gm_load_cube_buffer_count
            multiplicity_kind = "fixed" if fixed_count > 1 else "single"
            reason = "GM load belongs to the Cube scope and is not a UB ordinary Load candidate"
        else:
            skip_function = parsed.function_name in profile.gm_load_skip_functions
            automatic_count = profile.gm_load_vector_buffer_count
            if skip_function:
                resolved_count = 1
            elif path.hint_count is not None:
                resolved_count = path.hint_count
            else:
                resolved_count = automatic_count

            if resolved_count > 1:
                classification = "fixed_gm_load"
                fixed_count = resolved_count
                multiplicity_kind = "fixed"
                reason = ("MarkGMLoadPass traces the source to an entry GM argument and the destination to this "
                          f"Vector alloc, then pre-marks it with multi_buffer={resolved_count}; ordinary "
                          "MarkMultiBuffer observes the existing mark and skips it")
            else:
                classification = "ordinary"
                fixed_count = 1
                multiplicity_kind = "ordinary"
                reason = ("GM-load pre-marking is disabled by the function rule or hint; the later Vector "
                          "hivm.Load can therefore claim the local UB allocation as an ordinary candidate")

        contribution = _make_contribution(
            origin=path.allocation,
            path_kind="gm_load",
            classification=classification,
            logical_shape=path.logical_shape,
            dtype=path.dtype,
            core_type=path.core_type,
            block_id=path.block_id,
            owner_loop_kinds=path.loop_kinds,
            lifetime=path.lifetime,
            multiplicity_kind=multiplicity_kind,
            fixed_count=max(1, fixed_count),
            provenance=path.provenance,
            profile=profile,
            reason=reason,
        )
        if classification == "ordinary":
            key = (path.allocation, path.block_id, path.logical_shape, path.dtype)
            if key not in seen_ordinary:
                ordinary.append(contribution)
                seen_ordinary.add(key)
        else:
            key = (path.allocation, classification)
            if key not in seen_excluded:
                excluded.append(contribution)
                seen_excluded.add(key)

    return ordinary, excluded, blockers


def analyze_buffer_families(
    graph: IRGraph,
    profile: CompilerProfile,
    dynamic: DynamicCVAnalysis,
) -> BufferAnalysis:
    """Resolve ordinary families, ownership, and cross-family support guards."""

    stores, loads, boundary_blockers = _extract_boundary_paths(graph)
    ordinary, excluded, ownership_blockers = _classify_ownership(graph, stores, loads, profile)

    dynamic_origins = {family.origin for family in dynamic.families}
    ordinary_origins = {family.origin for family in ordinary}
    overlap = sorted(dynamic_origins & ordinary_origins)
    interaction_blockers = (["DynamicCV and ordinary ownership overlap is unsupported: " +
                             ", ".join(overlap)] if overlap else [])
    return BufferAnalysis(
        dependency_summary=dynamic.dependency_summary,
        dependencies=dynamic.dependencies,
        dynamic_cuts=dynamic.cuts,
        dynamic_families=dynamic.families,
        ordinary_families=tuple(ordinary),
        excluded_families=tuple(excluded),
        blockers=tuple((*dynamic.blockers, *boundary_blockers, *ownership_blockers, *interaction_blockers)),
    )
