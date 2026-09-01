"""Compiler-aligned DynamicCV dependency-cut analysis.

The common PlanComputeBlock graph is prepared once.  This module mirrors the
SSA part of UBUsageOpt: build loop-local weighted def-use edges, map loop-carried
arguments back to their yielded values, move the minimum-frontier prefix to the
producer block, and expose the shaped VECTOR values crossing the rewritten
frontier.  It never uses an operator name or an observed UB total as a key.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

from .buffer_geometry import align_up, dtype_bits, project_physical_shape
from .build_ir_graph import STATIC_SHAPED_TYPE_RE
from .validate_context import CompilerProfile, DYNAMIC_FAMILY_RULE_ID
from ..model_types import (
    DynamicBufferFamily,
    DynamicCut,
    DynamicCVAnalysis,
    DynamicDependency,
    IRGraph,
    LoopRegion,
    Operation,
)

MAX_EDGE_SIZE = 1 << 30
DYNAMIC_CUT_RULE_ID = DYNAMIC_FAMILY_RULE_ID

_NON_ALLOCATING_OPS = {
    "bufferization.alloc_tensor",
    "memref.alloc",
    "tensor.empty",
}


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    value: str
    size_bytes: int
    from_loop_carried_argument: bool


@dataclass(frozen=True)
class _RawCut:
    loop_id: int
    seed_value: str
    seed_node: str
    target_block: int
    activate_node: str
    activate_block: int
    origin_bytes: int
    selected_bytes: int
    chain: tuple[str, ...]
    moved: tuple[str, ...]


@dataclass(frozen=True)
class _LoopCutAnalysis:
    """One loop's shared cut state, retained once regardless of cut count."""

    cuts: tuple[_RawCut, ...]
    assignments: dict[str, int]
    assignment_causes: dict[str, frozenset[int]]
    blockers: tuple[str, ...]


def _base_value(value: str) -> str:
    return value.split("#", 1)[0]


def _shape_and_dtype(operation: Operation) -> Optional[tuple[tuple[int, ...], str]]:
    matches = STATIC_SHAPED_TYPE_RE.findall(operation.text)
    if not matches:
        return None
    pieces = matches[-1].split("x")
    return tuple(int(piece) for piece in pieces[:-1]), pieces[-1]


def _logical_size_bytes(operation: Operation) -> int:
    shaped = _shape_and_dtype(operation)
    if shaped is None:
        return 0
    # UBUsageOpt deliberately gives memref values an infinite edge weight.
    # to_tensor/cast operations end in a tensor type, so checking the final
    # shaped spelling rather than any operand memref preserves that behavior.
    last_tensor = operation.text.rfind("tensor<")
    last_vector = operation.text.rfind("vector<")
    last_memref = operation.text.rfind("memref<")
    if last_memref > max(last_tensor, last_vector):
        return MAX_EDGE_SIZE
    shape, dtype = shaped
    return max(1, math.prod(shape) * dtype_bits(dtype) // 8)


def _operation_index(graph: IRGraph) -> dict[str, tuple[Operation, ...]]:
    values: dict[str, list[Operation]] = defaultdict(list)
    for operation in graph.operations:
        values[operation.result].append(operation)
    return {
        value: tuple(sorted(operations, key=lambda item: item.line_number))
        for value, operations in values.items()
    }


def _resolve_before(
    index: dict[str, tuple[Operation, ...]],
    value: str,
    consumer: Operation,
) -> Optional[Operation]:
    candidates = index.get(_base_value(value), ())
    preceding = [item for item in candidates if item.line_number < consumer.line_number]
    if not preceding:
        return None
    same_loop = [item for item in preceding if item.loop_id == consumer.loop_id]
    return (same_loop or preceding)[-1]


def _resolve_in_loop(
    local_index: dict[str, tuple[Operation, ...]],
    value: str,
    consumer: Optional[Operation] = None,
    *,
    allow_after: bool = False,
) -> Optional[Operation]:
    candidates = local_index.get(_base_value(value), ())
    if not candidates:
        return None
    if allow_after or consumer is None:
        return candidates[-1]
    preceding = [item for item in candidates if item.line_number < consumer.line_number]
    return preceding[-1] if preceding else None


def _build_broad_dependencies(
    graph: IRGraph,
    profile: CompilerProfile,
) -> tuple[dict[str, object], tuple[DynamicDependency, ...]]:
    index = _operation_index(graph)
    dependencies: list[DynamicDependency] = []
    for consumer in graph.operations:
        if consumer.block_id is None or consumer.core_type is None:
            continue
        for operand in consumer.operands:
            producer = _resolve_before(index, operand, consumer)
            if producer is None or producer.block_id is None or producer.core_type is None:
                continue
            if producer.block_id == consumer.block_id or _shape_and_dtype(producer) is None:
                continue
            if producer.name in _NON_ALLOCATING_OPS:
                continue
            same_core = producer.core_type == consumer.core_type
            dependencies.append(
                DynamicDependency(
                    value=operand,
                    producer_block=producer.block_id,
                    consumer_block=consumer.block_id,
                    producer_core=producer.core_type,
                    consumer_core=consumer.core_type,
                    dependency_kind="intra_core" if same_core else "inter_core",
                    multiplicity=(graph.module_attributes["ssbuffer.intra_buf_count"]
                                  if same_core else profile.inter_cache_num),
                    line_number=consumer.line_number,
                    loop_id=consumer.loop_id,
                    consumer_value=consumer.result,
                    logical_bytes=_logical_size_bytes(producer),
                ))

    dependencies = sorted(
        set(dependencies),
        key=lambda item: (
            item.producer_block,
            item.consumer_block,
            item.value,
            item.line_number,
        ),
    )
    summary: dict[str, object] = {
        "block_count": graph.block_count,
        "vector_block_count": graph.vector_block_count,
        "cube_block_count": graph.cube_block_count,
        "intra_cache_num": graph.module_attributes["ssbuffer.intra_buf_count"],
        "inter_cache_num": graph.module_attributes["ssbuffer.inter_core_buf_count"],
        "load_cache_num": graph.module_attributes["ssbuffer.load_store_buf_count"],
        "aiv_subblock_factor": profile.aiv_subblock_factor,
        "dependency_count": len(dependencies),
        "intra_dependency_count": sum(item.dependency_kind == "intra_core" for item in dependencies),
        "inter_dependency_count": sum(item.dependency_kind == "inter_core" for item in dependencies),
    }
    return summary, tuple(dependencies)


def _eligible_loops(graph: IRGraph) -> Iterable[LoopRegion]:
    # TODO(cost-model): `iter_args` is not a compiler eligibility condition.
    # Do not replace this prototype restriction with an unconditional
    # "innermost loop only" rule: an enclosing loop may own DynamicCV-relevant
    # dependencies. Mirror MarkMainLoop plus inner/outer cache ownership for
    # nested and sibling loops, then make loop-carried edges optional.
    for loop in graph.loops:
        if set(loop.core_types) != {"CUBE", "VECTOR"}:
            continue
        if not loop.iter_args:
            continue
        yield loop


def _analyze_loop_cut(
    graph: IRGraph,
    loop: LoopRegion,
) -> _LoopCutAnalysis:
    blockers: list[str] = []
    if loop.kind != "scf.for":
        return _LoopCutAnalysis(
            (), {}, {}, (f"mixed-core loop {loop.loop_id} uses unsupported {loop.kind}",))
    if len(loop.iter_args) != len(loop.yielded_values):
        return _LoopCutAnalysis(
            (),
            {},
            {},
            (f"loop {loop.loop_id} has {len(loop.iter_args)} iter_args but "
             f"{len(loop.yielded_values)} yielded values",),
        )

    operations = tuple(
        item for item in graph.operations
        if item.loop_id == loop.loop_id
        and item.block_id is not None
        and item.core_type in {"VECTOR", "CUBE"}
    )
    local_lists: dict[str, list[Operation]] = defaultdict(list)
    for operation in operations:
        local_lists[operation.result].append(operation)
    duplicate_values = tuple(
        sorted(value for value, items in local_lists.items() if len(items) != 1))
    if duplicate_values:
        # MLIR permits equal printed SSA spellings in disjoint nested regions.
        # A textual prototype cannot safely join those scopes, so refuse to
        # predict instead of silently binding an operand to the wrong value.
        return _LoopCutAnalysis(
            (),
            {},
            {},
            (f"loop {loop.loop_id} has ambiguous region-local SSA names: "
             f"{', '.join(duplicate_values)}",),
        )
    nodes = {value: items[0] for value, items in local_lists.items()}
    local_index = {
        value: tuple(sorted(items, key=lambda item: item.line_number))
        for value, items in local_lists.items()
    }
    carried = dict(zip(loop.iter_args, loop.yielded_values))

    edges: list[_Edge] = []
    incoming: dict[str, list[int]] = defaultdict(list)
    outgoing: dict[str, list[int]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for consumer in sorted(operations, key=lambda item: item.line_number):
        for operand in consumer.operands:
            from_carried = operand in carried
            resolved_value = carried.get(operand, operand)
            producer = _resolve_in_loop(
                local_index,
                resolved_value,
                consumer,
                allow_after=from_carried,
            )
            if producer is None or producer.result == consumer.result:
                continue
            if consumer.core_type == "CUBE" and producer.block_id == consumer.block_id:
                continue
            pair = (producer.result, consumer.result)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            size = _logical_size_bytes(producer)
            if from_carried and size < MAX_EDGE_SIZE:
                size *= 2
            edge_id = len(edges)
            edges.append(_Edge(producer.result, consumer.result, resolved_value, size, from_carried))
            outgoing[producer.result].append(edge_id)
            incoming[consumer.result].append(edge_id)

    @lru_cache(maxsize=None)
    def dependency_nodes(target: str, excluded_predecessor: str) -> tuple[str, ...]:
        found: list[str] = []
        visited = {target}
        queue = deque([target])
        while queue:
            current = queue.popleft()
            if current == excluded_predecessor:
                continue
            for edge_id in incoming.get(current, ()):
                source = edges[edge_id].source
                if source not in visited:
                    visited.add(source)
                    found.append(source)
                    queue.append(source)
        return tuple(found)

    def is_active_end(source: str, target: str) -> bool:
        source_op = nodes[source]
        target_op = nodes[target]
        if source_op.core_type != target_op.core_type:
            return False
        if source_op.block_id == target_op.block_id:
            return False
        for dependency in dependency_nodes(target, source):
            dependency_block = nodes[dependency].block_id
            if dependency_block not in {source_op.block_id, target_op.block_id}:
                # This is the current compiler's allowance for a root such as
                # an empty+fill seed that has no incoming graph edge.
                if incoming.get(dependency):
                    return False
        return True

    def unique_next(current: str) -> Optional[str]:
        edge_ids = outgoing.get(current, ())
        if len(edge_ids) != 1:
            return None
        next_node = edges[edge_ids[0]].target
        if nodes[next_node].block_id != nodes[current].block_id:
            return None
        if any(nodes[item].block_id != nodes[current].block_id
               for item in dependency_nodes(next_node, current)):
            return None
        return next_node

    cuts: list[_RawCut] = []
    assignments: dict[str, int] = {}
    assignment_causes: dict[str, set[int]] = defaultdict(set)

    def assign(node: str, block_id: int, cut_index: int) -> None:
        previous = assignments.get(node)
        if previous is not None and previous != block_id:
            blockers.append(
                f"loop {loop.loop_id} assigns {node} to both block {previous} and block {block_id}")
            return
        assignments[node] = block_id
        assignment_causes[node].add(cut_index)

    for source in sorted(nodes, key=lambda value: nodes[value].line_number):
        source_op = nodes[source]
        if source_op.core_type != "VECTOR":
            continue
        active_targets: list[tuple[str, str]] = []
        for edge_id in outgoing.get(source, ()):
            edge = edges[edge_id]
            if is_active_end(source, edge.target):
                active_targets.append((edge.target, edge.value))
        for activate, seed_value in active_targets:
            origin_size = sum(
                edges[edge_id].size_bytes
                for edge_id in incoming.get(activate, ())
                if nodes[edges[edge_id].source].block_id != nodes[edges[edge_id].target].block_id
            )
            chain = [activate]
            while (next_node := unique_next(chain[-1])) is not None:
                chain.append(next_node)

            best_count = 0
            selected_size = origin_size
            for index, node in enumerate(chain, start=1):
                cut_size = sum(edges[edge_id].size_bytes for edge_id in outgoing.get(node, ()))
                if cut_size < selected_size:
                    best_count = index
                    selected_size = cut_size
            if best_count == 0:
                continue

            moved = tuple(chain[:best_count])
            cut_index = len(cuts)
            cuts.append(_RawCut(
                loop_id=loop.loop_id,
                seed_value=seed_value,
                seed_node=source,
                target_block=source_op.block_id,
                activate_node=activate,
                activate_block=nodes[activate].block_id,
                origin_bytes=origin_size,
                selected_bytes=selected_size,
                chain=tuple(chain),
                moved=moved,
            ))
            for index, node in enumerate(moved):
                assign(node, source_op.block_id, cut_index)
                predecessor = source if index == 0 else chain[index - 1]
                for dependency in dependency_nodes(node, predecessor):
                    if nodes[dependency].block_id != source_op.block_id:
                        assign(dependency, source_op.block_id, cut_index)

    return _LoopCutAnalysis(
        cuts=tuple(cuts),
        assignments=assignments,
        assignment_causes={
            value: frozenset(indexes) for value, indexes in assignment_causes.items()
        },
        blockers=tuple(blockers),
    )


def analyze_dynamic_cv(graph: IRGraph, profile: CompilerProfile) -> DynamicCVAnalysis:
    summary, dependencies = _build_broad_dependencies(graph, profile)
    loop_analyses: dict[int, _LoopCutAnalysis] = {}
    blockers: list[str] = []

    # Keep per-loop results separate because MLIR permits the same printed SSA
    # name in sibling regions.
    for loop in _eligible_loops(graph):
        analysis = _analyze_loop_cut(graph, loop)
        blockers.extend(analysis.blockers)
        if analysis.cuts or analysis.assignments:
            loop_analyses[loop.loop_id] = analysis

    # Reconstruct the family frontier per loop from the recorded cuts.  The
    # compact representation intentionally stores only values caused by a cut;
    # pre-existing broad dependencies remain evidence, not cache families.
    operations_by_loop: dict[int, list[Operation]] = defaultdict(list)
    for operation in graph.operations:
        if operation.loop_id is not None:
            operations_by_loop[operation.loop_id].append(operation)

    family_records: dict[tuple[int, str], dict[str, object]] = {}
    cuts_by_loop = {
        loop_id: analysis.cuts for loop_id, analysis in loop_analyses.items()
    }
    loop_by_id = {loop.loop_id: loop for loop in graph.loops}

    # Rebuild direct users cheaply from loop-local Operations.
    for loop_id, analysis in loop_analyses.items():
        assignments = analysis.assignments
        operations = operations_by_loop[loop_id]
        local_values: dict[str, list[Operation]] = defaultdict(list)
        for operation in operations:
            local_values[operation.result].append(operation)
        local_index = {
            value: tuple(sorted(items, key=lambda item: item.line_number))
            for value, items in local_values.items()
        }
        carried_loop = loop_by_id[loop_id]
        carried = dict(zip(carried_loop.iter_args, carried_loop.yielded_values))
        for consumer in operations:
            for operand in consumer.operands:
                actual = carried.get(operand, operand)
                source_op = _resolve_in_loop(
                    local_index,
                    actual,
                    consumer,
                    allow_after=operand in carried,
                )
                if source_op is None:
                    continue
                source = source_op.result
                if source not in assignments:
                    continue
                source_block = assignments[source]
                target_block = assignments.get(consumer.result, consumer.block_id)
                if (source_block == target_block or source_op.core_type != "VECTOR"
                        or consumer.core_type != "VECTOR"):
                    continue
                shaped = _shape_and_dtype(source_op)
                if shaped is None or source_op.name in _NON_ALLOCATING_OPS:
                    continue
                record = family_records.setdefault((loop_id, source), {
                    "operation": source_op,
                    "producer_block": source_block,
                    "consumer_blocks": set(),
                    "consumer_values": set(),
                    "cut_indexes": set(),
                })
                record["consumer_blocks"].add(target_block)
                record["consumer_values"].add(consumer.result)
                record["cut_indexes"].update(analysis.assignment_causes.get(source, ()))

    families: list[DynamicBufferFamily] = []
    for (loop_id, value), record in sorted(
            family_records.items(), key=lambda item: item[1]["operation"].line_number):
        operation = record["operation"]
        shape, dtype = _shape_and_dtype(operation)  # type: ignore[arg-type]
        physical_shape = project_physical_shape(shape, "VECTOR", profile)
        element_bits = dtype_bits(dtype)
        raw_bytes = (math.prod(physical_shape) * element_bits + 7) // 8
        cut_indexes = sorted(record["cut_indexes"])
        loop_cuts = cuts_by_loop[loop_id]
        selected_cuts = [loop_cuts[index] for index in cut_indexes if index < len(loop_cuts)]
        seeds = tuple(sorted({cut.seed_value for cut in selected_cuts}))
        provenance = [
            f"dependency_cut_rule:{DYNAMIC_CUT_RULE_ID}",
            *(f"seed:{cut.seed_value}:b{cut.target_block}->b{cut.activate_block}"
              for cut in selected_cuts),
            *(f"frontier:{cut.origin_bytes}B->{cut.selected_bytes}B"
              for cut in selected_cuts),
            f"block_rewrite:{operation.block_id}->{record['producer_block']}",
            f"family:{value}:consumers={sorted(record['consumer_blocks'])}",
        ]
        families.append(DynamicBufferFamily(
            origin=value,
            pattern="ub_usage_min_cut",
            logical_shape=shape,
            physical_shape=physical_shape,
            dtype=dtype,
            element_bits=element_bits,
            raw_bytes=raw_bytes,
            aligned_bytes=align_up(raw_bytes, profile.ub_alignment_bytes),
            line_number=operation.line_number,
            loop_depth=operation.loop_depth,
            producer_block=int(record["producer_block"]),
            effective_counts=(1, 1, 1, 1),
            provenance=tuple(provenance),
            reason=("UBUsageOpt's loop-local weighted dependency graph moves the minimum-frontier "
                    "prefix to the seed producer block; this shaped VECTOR result is a post-rewrite "
                    "cross-block dependency consumed by AddMultiBufferInnerScope"),
            consumer_blocks=tuple(sorted(record["consumer_blocks"])),
            seed_dependencies=seeds,
            selection_rule=DYNAMIC_CUT_RULE_ID,
        ))

    public_cuts: list[DynamicCut] = []
    for loop_id, loop_cuts in sorted(cuts_by_loop.items()):
        family_by_cut: dict[int, list[str]] = defaultdict(list)
        for (family_loop, value), record in family_records.items():
            if family_loop != loop_id:
                continue
            for cut_index in record["cut_indexes"]:
                family_by_cut[cut_index].append(value)
        for index, cut in enumerate(loop_cuts):
            public_cuts.append(DynamicCut(
                loop_id=loop_id,
                seed_value=cut.seed_value,
                seed_producer_block=cut.target_block,
                activate_value=cut.activate_node,
                activate_block=cut.activate_block,
                origin_frontier_bytes=cut.origin_bytes,
                selected_frontier_bytes=cut.selected_bytes,
                moved_values=cut.moved,
                family_values=tuple(sorted(family_by_cut[index])),
                rule=DYNAMIC_CUT_RULE_ID,
                provenance=(
                    f"seed_node:{cut.seed_node}",
                    f"chain:{','.join(cut.chain)}",
                    f"selected_prefix:{len(cut.moved)}",
                ),
            ))

    # The compact model deliberately simulates UBUsageOpt only.  Later
    # ComputeBlockOpt passes (BroadcastUBOpt, MergeSameSourceAxis and
    # MergeSmallBlock in this compiler profile) can change the final
    # cross-block family set when the selected cut is not a one-to-one
    # frontier, multiple cuts share one seed, or the first consumer fans out
    # further.  These conditions are structural, cheap to detect, and must be
    # fail-open until those passes are modeled.  Returning a numerical Delta
    # here would otherwise under- or over-count AddMultiBufferInnerScope allocs.
    family_operations = {
        (loop_id, value): record["operation"]
        for (loop_id, value), record in family_records.items()
    }
    for cut in public_cuts:
        if not cut.family_values:
            continue
        family_frontier = sum(
            _logical_size_bytes(family_operations[(cut.loop_id, value)])
            for value in cut.family_values
        )
        if family_frontier != cut.selected_frontier_bytes:
            blockers.append(
                f"loop {cut.loop_id} cut {cut.seed_value} selects "
                f"{cut.selected_frontier_bytes} bytes but exposes "
                f"{family_frontier} family bytes; downstream compute-block "
                "rewrite is unsupported")

    family_cuts_by_seed: dict[tuple[int, str], list[DynamicCut]] = defaultdict(list)
    for cut in public_cuts:
        if cut.family_values:
            family_cuts_by_seed[(cut.loop_id, cut.seed_value)].append(cut)
    for (loop_id, seed), seed_cuts in family_cuts_by_seed.items():
        if len(seed_cuts) > 1:
            blockers.append(
                f"loop {loop_id} seed {seed} produces {len(seed_cuts)} "
                "independent family cuts; downstream same-source merging is unsupported")

    for (loop_id, _value), record in family_records.items():
        operations = operations_by_loop[loop_id]
        local_values: dict[str, list[Operation]] = defaultdict(list)
        for operation in operations:
            local_values[operation.result].append(operation)
        local_index = {
            value: tuple(sorted(items, key=lambda item: item.line_number))
            for value, items in local_values.items()
        }
        direct_user_counts: dict[str, set[str]] = defaultdict(set)
        for consumer in operations:
            for operand in consumer.operands:
                source_op = _resolve_in_loop(local_index, operand, consumer)
                if source_op is not None:
                    direct_user_counts[source_op.result].add(consumer.result)
        for consumer_value in record["consumer_values"]:
            fanout = len(direct_user_counts.get(consumer_value, ()))
            if fanout > 2:
                blockers.append(
                    f"loop {loop_id} cross-block consumer {consumer_value} "
                    f"has shaped dependency fanout {fanout}; downstream "
                    "same-source merging is unsupported")

    eligible_by_parent: dict[Optional[int], list[int]] = defaultdict(list)
    family_loop_ids = {loop_id for loop_id, _value in family_records}
    for loop in _eligible_loops(graph):
        if loop.loop_id in family_loop_ids:
            eligible_by_parent[loop.parent_loop_id].append(loop.loop_id)
    for parent_loop_id, child_loop_ids in eligible_by_parent.items():
        if len(child_loop_ids) > 1:
            blockers.append(
                f"parent loop {parent_loop_id} contains {len(child_loop_ids)} "
                "mixed-core child loops; cross-stage DynamicCV coupling is unsupported")

    summary.update({
        "candidate_loop_count": sum(1 for _ in _eligible_loops(graph)),
        "dependency_cut_count": len(public_cuts),
        "family_producing_cut_count": sum(bool(item.family_values) for item in public_cuts),
        "dependency_cut_rule": DYNAMIC_CUT_RULE_ID,
    })
    return DynamicCVAnalysis(
        dependency_summary=summary,
        dependencies=dependencies,
        cuts=tuple(public_cuts),
        families=tuple(families),
        blockers=tuple(blockers),
    )
