"""Optional lightweight lifetime expansion and address placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .model_types import BufferPlacement, LifeInterval, MemoryEntry, PlanLiteResult, UnsupportedModelError


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class _ExpandedEntry:
    entry_id: str
    copy_index: int
    size_bytes: int
    alignment_bytes: int
    lifetimes: tuple[LifeInterval, ...]
    order: int


def _merge_lifetimes(intervals: Iterable[LifeInterval]) -> tuple[LifeInterval, ...]:
    ordered = sorted(intervals)
    if not ordered:
        return ()
    merged: list[LifeInterval] = [ordered[0]]
    for interval in ordered[1:]:
        previous = merged[-1]
        if interval.start <= previous.end:
            merged[-1] = LifeInterval(previous.start, max(previous.end, interval.end))
        else:
            merged.append(interval)
    return tuple(merged)


def _resolved_multiplicity(entry: MemoryEntry, multibuffer_num: int) -> int:
    if entry.multiplicity_kind == "ordinary":
        return multibuffer_num
    if entry.multiplicity_kind == "fixed":
        return entry.fixed_count
    return 1


def _expand_memory_entries(
    entries: Sequence[MemoryEntry],
    multibuffer_num: int,
) -> list[_ExpandedEntry]:
    grouped: dict[str, list[tuple[int, MemoryEntry]]] = {}
    for order, entry in enumerate(entries):
        key = entry.alias_group or f"__entry__:{order}:{entry.entry_id}"
        grouped.setdefault(key, []).append((order, entry))

    expanded: list[_ExpandedEntry] = []
    for group in grouped.values():
        first_order = min(order for order, _entry in group)
        representative = min(group, key=lambda item: item[0])[1]
        if len({entry.size_bytes for _order, entry in group}) > 1:
            raise UnsupportedModelError("aliased PlanLite entries must have the same compiler storage size")
        if len({entry.alignment_bytes for _order, entry in group}) > 1:
            raise UnsupportedModelError("aliased PlanLite entries must have the same alignment")
        size_bytes = max(entry.size_bytes for _order, entry in group)
        alignment_bytes = max(entry.alignment_bytes for _order, entry in group)
        lifetimes = _merge_lifetimes(interval for _order, entry in group for interval in entry.lifetimes)
        count = max(_resolved_multiplicity(entry, multibuffer_num) for _order, entry in group)
        entry_id = "+".join(entry.entry_id for _order, entry in sorted(group))
        for copy_index in range(count):
            expanded.append(
                _ExpandedEntry(
                    entry_id=entry_id or representative.entry_id,
                    copy_index=copy_index,
                    size_bytes=_align_up(size_bytes, alignment_bytes),
                    alignment_bytes=alignment_bytes,
                    lifetimes=lifetimes,
                    order=first_order,
                ))
    return expanded


def _entries_conflict(first: _ExpandedEntry, second: _ExpandedEntry) -> bool:
    return any(left.overlaps(right) for left in first.lifetimes for right in second.lifetimes)


def _live_lower_bound(entries: Sequence[_ExpandedEntry]) -> int:
    points = sorted(
        {point
         for entry in entries for interval in entry.lifetimes for point in (interval.start, interval.end)})
    return max(
        (sum(entry.size_bytes for entry in entries if any(interval.start <= point <= interval.end
                                                          for interval in entry.lifetimes)) for point in points),
        default=0,
    )


def _place_entries(entries: Sequence[_ExpandedEntry]) -> tuple[int, tuple[BufferPlacement, ...]]:
    placed: list[tuple[_ExpandedEntry, int]] = []
    peak = 0
    for entry in entries:
        conflicting = sorted(
            ((other, offset) for other, offset in placed if _entries_conflict(entry, other)),
            key=lambda item: item[1],
        )
        offset = 0
        while True:
            offset = _align_up(offset, entry.alignment_bytes)
            collision = next(
                ((other, other_offset) for other, other_offset in conflicting
                 if offset < other_offset + other.size_bytes and other_offset < offset + entry.size_bytes),
                None,
            )
            if collision is None:
                break
            other, other_offset = collision
            offset = other_offset + other.size_bytes
        placed.append((entry, offset))
        peak = max(peak, offset + entry.size_bytes)
    placements = tuple(
        BufferPlacement(entry.entry_id, entry.copy_index, offset, entry.size_bytes) for entry, offset in placed)
    return peak, placements


def plan_lite(
    entries: Sequence[MemoryEntry],
    *,
    multibuffer_num: int,
    ub_capacity_bytes: int,
    entry_graph_complete: bool,
) -> PlanLiteResult:
    """Run the guarded subset of PlanMemory needed by the cost model."""

    if multibuffer_num not in range(1, 5):
        raise UnsupportedModelError("multibuffer_num must be one of 1, 2, 3, 4")
    scopes = {entry.memory_scope for entry in entries}
    if scopes - {"ub"}:
        raise UnsupportedModelError(f"PlanLite supports only UB entries, got {sorted(scopes)}")
    alignments = {entry.alignment_bytes for entry in entries}
    if len(alignments) > 1:
        raise UnsupportedModelError("PlanLite requires one compiler scope alignment for all UB entries")
    expanded = _expand_memory_entries(entries, multibuffer_num)
    no_reuse_required = sum(entry.size_bytes for entry in expanded)
    live_lower_bound = _live_lower_bound(expanded)

    if no_reuse_required <= ub_capacity_bytes:
        offset = 0
        placements: list[BufferPlacement] = []
        for entry in expanded:
            offset = _align_up(offset, entry.alignment_bytes)
            placements.append(BufferPlacement(entry.entry_id, entry.copy_index, offset, entry.size_bytes))
            offset += entry.size_bytes
        return PlanLiteResult(
            status="exact_no_reuse" if entry_graph_complete else "partial_no_reuse_estimate",
            no_reuse_required_bytes=no_reuse_required,
            live_lower_bound_bytes=live_lower_bound,
            predicted_peak_bytes=offset,
            placements=tuple(placements),
            proof="all expanded entries fit in PlanMemory's no-reuse branch" if entry_graph_complete else None,
            reliable=entry_graph_complete,
            prune_allowed=False,
        )

    if entry_graph_complete and live_lower_bound > ub_capacity_bytes:
        return PlanLiteResult(
            status="proven_overflow",
            no_reuse_required_bytes=no_reuse_required,
            live_lower_bound_bytes=live_lower_bound,
            predicted_peak_bytes=None,
            placements=(),
            proof=(f"simultaneously live aligned entries require {live_lower_bound} bytes, "
                   f"exceeding capacity {ub_capacity_bytes}"),
            reliable=True,
            prune_allowed=True,
        )

    orderings = [
        sorted(expanded, key=lambda entry: (entry.order, entry.copy_index)),
        sorted(expanded, key=lambda entry: (-entry.size_bytes, entry.order, entry.copy_index)),
        sorted(
            expanded,
            key=lambda entry: (
                min(interval.start for interval in entry.lifetimes),
                -entry.size_bytes,
                entry.order,
                entry.copy_index,
            ),
        ),
    ]
    attempts = [_place_entries(ordering) for ordering in orderings]
    peak, placements = min(attempts, key=lambda item: item[0])
    if entry_graph_complete and peak <= ub_capacity_bytes:
        return PlanLiteResult(
            status="feasible_placement",
            no_reuse_required_bytes=no_reuse_required,
            live_lower_bound_bytes=live_lower_bound,
            predicted_peak_bytes=peak,
            placements=placements,
            proof="a conflict-free placement was constructed from the complete lifetime graph",
            reliable=True,
            prune_allowed=False,
        )
    return PlanLiteResult(
        status="unknown" if entry_graph_complete else "partial_projection",
        no_reuse_required_bytes=no_reuse_required,
        live_lower_bound_bytes=live_lower_bound,
        predicted_peak_bytes=peak,
        placements=placements,
        proof=None,
        reliable=False,
        prune_allowed=False,
    )
