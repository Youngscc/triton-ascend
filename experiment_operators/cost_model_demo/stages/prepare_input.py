"""Stage 1: parse PlanComputeBlock IR and validate the model context."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .build_ir_graph import build_ir_graph
from .validate_context import CompilerProfile, ValidatedContext, validate_context
from ..model_types import IRGraph


def prepare_input(
    source_text: str,
    *,
    vf_merge_level: int,
    profile: CompilerProfile,
    observed_provenance: Optional[Mapping[str, Any]],
) -> tuple[IRGraph, ValidatedContext]:
    """Build the structural index and reject an incompatible compiler context."""

    graph = build_ir_graph(source_text)
    context = validate_context(
        source_text,
        graph,
        vf_merge_level,
        profile,
        observed_provenance,
    )
    return graph, context
