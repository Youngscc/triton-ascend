"""Reduce analyzed families and build the reusable d/m parameter model."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .validate_context import (
    DYNAMIC_REDUCTION_RULE_ID,
    DYNAMIC_SELECTOR_EFFECTIVE_COUNTS,
    MODEL_VERSION,
    CompilerProfile,
    ValidatedContext,
    model_profile_fingerprint,
)
from ..model_types import (
    BufferAnalysis,
    BufferContribution,
    DynamicBufferFamily,
    IRGraph,
    MemoryEntry,
    PreparedCostModel,
)


def _reduce_dynamic_families(families: Sequence[DynamicBufferFamily]) -> tuple[DynamicBufferFamily, ...]:
    """Apply the profile-bound bufferization/MergeInplaceSE entry rule."""

    return tuple(
        replace(
            family,
            effective_counts=DYNAMIC_SELECTOR_EFFECTIVE_COUNTS,
            provenance=(*family.provenance, f"entry_reducer:{DYNAMIC_REDUCTION_RULE_ID}"),
        ) for family in families)


def _project_memory_entries(
    ordinary: Sequence[BufferContribution],
    excluded: Sequence[BufferContribution],
    profile: CompilerProfile,
) -> tuple[MemoryEntry, ...]:
    entries = []
    for index, family in enumerate((*ordinary, *excluded)):
        if family.classification == "non_ub_cube_gm_load":
            continue
        entries.append(
            MemoryEntry(
                entry_id=f"projected:{index}:{family.path_kind}:{family.origin}",
                size_bytes=family.aligned_bytes,
                alignment_bytes=profile.ub_alignment_bytes,
                lifetimes=(family.lifetime, ),
                multiplicity_kind=family.multiplicity_kind,
                fixed_count=family.fixed_count,
                origin=family.origin,
            ))
    return tuple(entries)


def build_parametric_model(
    graph: IRGraph,
    context: ValidatedContext,
    analysis: BufferAnalysis,
) -> PreparedCostModel:
    """Reduce analyzed families into one model reusable by all 12 d/m pairs."""

    profile = context.profile
    dynamic_families = _reduce_dynamic_families(analysis.dynamic_families)
    dynamic_delta = tuple(
        sum((family.effective_counts[index] - family.effective_counts[0]) * family.aligned_bytes
            for family in dynamic_families) for index in range(3))
    ordinary_step = sum(family.delta_step_bytes for family in analysis.ordinary_families)
    ordinary_steps = (ordinary_step, ) * 3
    coupled = tuple((0, 0, 0, 0) for _ in range(3))
    entries = _project_memory_entries(analysis.ordinary_families, analysis.excluded_families, profile)
    dependency_summary = {
        **analysis.dependency_summary,
        "broad_dependency_count": len(analysis.dependencies),
        "supported_family_count": len(dynamic_families),
        "supported_family_bytes": sum(family.aligned_bytes for family in dynamic_families),
        "effective_selector_counts": list(DYNAMIC_SELECTOR_EFFECTIVE_COUNTS),
        "dynamic_delta_table_bytes": list(dynamic_delta),
        "reduction_rule": DYNAMIC_REDUCTION_RULE_ID,
        "scope": "loop-local-weighted-dependency-cut",
        "family_ledger": [
            {
                "origin": family.origin,
                "seed_dependencies": list(family.seed_dependencies),
                "producer_block": family.producer_block,
                "consumer_blocks": list(family.consumer_blocks),
                "aligned_bytes": family.aligned_bytes,
                "selection_rule": family.selection_rule,
            }
            for family in dynamic_families
        ],
    }
    return PreparedCostModel(
        model_version=MODEL_VERSION,
        profile=profile.profile_id,
        profile_fingerprint=profile.fingerprint,
        model_fingerprint=model_profile_fingerprint(profile),
        profile_provenance_verified=context.profile_provenance_verified,
        profile_guard_notes=context.profile_guard_notes,
        target=graph.target,
        ub_capacity_bytes=profile.ub_capacity_bytes,
        ub_alignment_bytes=profile.ub_alignment_bytes,
        source_ir_sha256=context.source_ir_sha256,
        normalized_ir_sha256=context.normalized_ir_sha256,
        source_intra_cache_num=context.source_intra_cache_num,
        vf_merge_level=context.vf_merge_level,
        dynamic_reduction_rule=DYNAMIC_REDUCTION_RULE_ID,
        dynamic_cuts=analysis.dynamic_cuts,
        dynamic_buffers=dynamic_families,
        ordinary_buffers=analysis.ordinary_families,
        excluded_buffers=analysis.excluded_families,
        dynamic_delta_table_bytes=dynamic_delta,
        ordinary_step_table_bytes=ordinary_steps,
        coupled_adjustment_table_bytes=coupled,
        interactions=(),
        projected_entries=entries,
        projected_entry_graph_complete=False,
        blockers=analysis.blockers,
        dynamic_cv_summary=dependency_summary,
        timing_us={},
    )
