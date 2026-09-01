"""Immutable records exchanged by the cost-model modules."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

DEFAULT_UB_ALIGNMENT_BYTES = 32


class UnsupportedModelError(RuntimeError):
    """The input is outside the explicitly supported cost-model profile."""


@dataclass(frozen=True, order=True)
class LifeInterval:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid lifetime interval [{self.start}, {self.end}]")

    def overlaps(self, other: "LifeInterval") -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class MemoryEntry:
    """Compact counterpart of one pre-expansion PlanMemory StorageEntry."""

    entry_id: str
    size_bytes: int
    lifetimes: tuple[LifeInterval, ...]
    alignment_bytes: int = DEFAULT_UB_ALIGNMENT_BYTES
    multiplicity_kind: str = "single"
    fixed_count: int = 1
    alias_group: Optional[str] = None
    origin: Optional[str] = None
    memory_scope: str = "ub"

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError(f"negative size for {self.entry_id}")
        if self.alignment_bytes <= 0:
            raise ValueError(f"invalid alignment for {self.entry_id}")
        if not self.lifetimes:
            raise ValueError(f"missing lifetime for {self.entry_id}")
        if self.multiplicity_kind not in {"single", "fixed", "ordinary"}:
            raise ValueError(f"invalid multiplicity kind for {self.entry_id}")
        if self.fixed_count < 1:
            raise ValueError(f"invalid fixed count for {self.entry_id}")


@dataclass(frozen=True)
class BufferPlacement:
    entry_id: str
    copy_index: int
    offset_bytes: int
    size_bytes: int


@dataclass(frozen=True)
class PlanLiteResult:
    status: str
    no_reuse_required_bytes: int
    live_lower_bound_bytes: int
    predicted_peak_bytes: Optional[int]
    placements: tuple[BufferPlacement, ...]
    proof: Optional[str]
    reliable: bool
    prune_allowed: bool


@dataclass(frozen=True)
class Operation:
    result: str
    result_count: int
    name: str
    operands: tuple[str, ...]
    text: str
    line_number: int
    loop_depth: int
    loop_kinds: tuple[str, ...]
    core_type: Optional[str]
    block_id: Optional[int]
    loop_id: Optional[int] = None


@dataclass(frozen=True)
class LoopRegion:
    """One structured loop body used as an independent dependency scope."""

    loop_id: int
    kind: str
    parent_loop_id: Optional[int]
    line_start: int
    line_end: int
    iter_args: tuple[str, ...]
    initial_values: tuple[str, ...]
    yielded_values: tuple[str, ...]
    core_types: tuple[str, ...]
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class FunctionArgument:
    name: str
    type_text: str
    tensor_kind: Optional[int]


@dataclass(frozen=True)
class IRGraph:
    lines: tuple[str, ...]
    definitions: dict[str, Operation]
    uses: dict[str, tuple[int, ...]]
    arguments: dict[str, FunctionArgument]
    output_arguments: frozenset[str]
    input_arguments: frozenset[str]
    materialize_lines: tuple[tuple[int, str, tuple[str, ...]], ...]
    copy_lines: tuple[tuple[int, str, tuple[str, ...]], ...]
    module_attributes: dict[str, int]
    has_fallback: bool
    target: str
    function_name: str
    block_count: int
    vector_block_count: int
    cube_block_count: int
    loops: tuple[LoopRegion, ...] = ()
    operations: tuple[Operation, ...] = ()


@dataclass(frozen=True)
class RawStorePath:
    source: str
    destination: str
    output_argument: str
    destination_view: str
    logical_shape: tuple[int, ...]
    dtype: str
    core_type: str
    block_id: Optional[int]
    line_number: int
    loop_depth: int
    loop_kinds: tuple[str, ...]
    source_line: int
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class RawLoadPath:
    input_argument: str
    allocation: str
    logical_shape: tuple[int, ...]
    dtype: str
    core_type: str
    block_id: Optional[int]
    line_number: int
    loop_kinds: tuple[str, ...]
    lifetime: LifeInterval
    hint_count: Optional[int]
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class DynamicDependency:
    value: str
    producer_block: int
    consumer_block: int
    producer_core: str
    consumer_core: str
    dependency_kind: str
    multiplicity: int
    line_number: int
    loop_id: Optional[int] = None
    consumer_value: Optional[str] = None
    logical_bytes: Optional[int] = None


@dataclass(frozen=True)
class DynamicCut:
    """One compiler-style UBUsageOpt cut and its block-rewrite evidence."""

    loop_id: int
    seed_value: str
    seed_producer_block: int
    activate_value: str
    activate_block: int
    origin_frontier_bytes: int
    selected_frontier_bytes: int
    moved_values: tuple[str, ...]
    family_values: tuple[str, ...]
    rule: str
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class DynamicBufferFamily:
    """One DynamicCV intra-cache family projected to one AIV function."""

    origin: str
    pattern: str
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    dtype: str
    element_bits: int
    raw_bytes: int
    aligned_bytes: int
    line_number: int
    loop_depth: int
    producer_block: Optional[int]
    effective_counts: tuple[int, ...]
    provenance: tuple[str, ...]
    reason: str
    consumer_blocks: tuple[int, ...] = ()
    seed_dependencies: tuple[str, ...] = ()
    selection_rule: str = ""


@dataclass(frozen=True)
class DynamicCVAnalysis:
    dependency_summary: dict[str, Any]
    dependencies: tuple[DynamicDependency, ...]
    cuts: tuple[DynamicCut, ...]
    families: tuple[DynamicBufferFamily, ...]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class InteractionContribution:
    origin: str
    cause: str
    intra_cache_num: int
    multibuffer_num: int
    contribution_bytes: int
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class BufferContribution:
    origin: str
    path_kind: str
    classification: str
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    dtype: str
    element_bits: int
    raw_bytes: int
    aligned_bytes: int
    delta_step_bytes: int
    block_id: Optional[int]
    owner_loop_kinds: tuple[str, ...]
    lifetime: LifeInterval
    multiplicity_kind: str
    fixed_count: int
    provenance: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class BaselineCertificate:
    ub_peak_bytes: int
    no_reuse_required_bytes: Optional[int] = None
    used_reuse: Optional[bool] = None
    profile_id: Optional[str] = None
    profile_fingerprint: Optional[str] = None
    model_fingerprint: Optional[str] = None
    ir_sha256: Optional[str] = None
    normalized_ir_sha256: Optional[str] = None
    intra_cache_num: Optional[int] = None
    multibuffer_num: Optional[int] = None
    vf_merge_level: Optional[int] = None
    entries: tuple[MemoryEntry, ...] = ()
    entry_graph_complete: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BaselineCertificate":
        data = dict(value)
        entries: list[MemoryEntry] = []
        for entry_value in data.pop("entries", []):
            item = dict(entry_value)
            item["lifetimes"] = tuple(LifeInterval(**interval) for interval in item.get("lifetimes", []))
            entries.append(MemoryEntry(**item))
        data["entries"] = tuple(entries)
        return cls(**data)


@dataclass(frozen=True)
class CostEstimate:
    intra_cache_num: int
    multibuffer_num: int
    dynamic_from_d1_bytes: Optional[int]
    ordinary_from_m1_bytes: Optional[int]
    interaction_bytes: Optional[int]
    total_from_11_bytes: Optional[int]
    predicted_ub_bytes: Optional[int]
    no_reuse_required_bytes: Optional[int]
    verdict: str
    confidence: str
    exact_no_reuse_path: bool
    prune_allowed: bool
    reason: str
    evaluate_us: float = 0.0
    interaction_contributions: tuple[InteractionContribution, ...] = ()


@dataclass(frozen=True)
class AutotuneDecision:
    action: str
    result: CostEstimate
    reason: str


@dataclass(frozen=True)
class PreparedCostModel:
    model_version: str
    profile: str
    profile_fingerprint: str
    model_fingerprint: str
    profile_provenance_verified: bool
    profile_guard_notes: tuple[str, ...]
    target: str
    ub_capacity_bytes: int
    ub_alignment_bytes: int
    source_ir_sha256: str
    normalized_ir_sha256: str
    source_intra_cache_num: int
    vf_merge_level: int
    dynamic_reduction_rule: str
    dynamic_cuts: tuple[DynamicCut, ...]
    dynamic_buffers: tuple[DynamicBufferFamily, ...]
    ordinary_buffers: tuple[BufferContribution, ...]
    excluded_buffers: tuple[BufferContribution, ...]
    dynamic_delta_table_bytes: tuple[int, ...]
    ordinary_step_table_bytes: tuple[int, ...]
    coupled_adjustment_table_bytes: tuple[tuple[int, ...], ...]
    interactions: tuple[InteractionContribution, ...]
    projected_entries: tuple[MemoryEntry, ...]
    projected_entry_graph_complete: bool
    blockers: tuple[str, ...]
    dynamic_cv_summary: dict[str, Any]
    timing_us: dict[str, float]
    coverage: str = "guarded-a5-dynamiccv-multibuffer-v1"


@dataclass
class StageTimer:
    values_ns: dict[str, int] = field(default_factory=dict)

    def measure(self, name: str, callback: Any) -> Any:
        start = time.perf_counter_ns()
        result = callback()
        self.values_ns[name] = time.perf_counter_ns() - start
        return result

    def as_microseconds(self) -> dict[str, float]:
        return {name: value / 1_000.0 for name, value in self.values_ns.items()}


@dataclass(frozen=True)
class BufferAnalysis:
    dependency_summary: dict[str, Any]
    dependencies: tuple[DynamicDependency, ...]
    dynamic_cuts: tuple[DynamicCut, ...]
    dynamic_families: tuple[DynamicBufferFamily, ...]
    ordinary_families: tuple[BufferContribution, ...]
    excluded_families: tuple[BufferContribution, ...]
    blockers: tuple[str, ...]
