"""Evaluate parameter combinations and calibrate them with a real baseline."""

from __future__ import annotations

import time
from typing import Optional, Sequence

from ..model_types import (
    BaselineCertificate,
    CostEstimate,
    InteractionContribution,
    MemoryEntry,
    PreparedCostModel,
    UnsupportedModelError,
)


def _validate_baseline(
    prepared: PreparedCostModel,
    baseline: BaselineCertificate,
) -> bool:
    expected = {
        "profile_id": prepared.profile,
        "profile_fingerprint": prepared.profile_fingerprint,
        "model_fingerprint": prepared.model_fingerprint,
        "normalized_ir_sha256": prepared.normalized_ir_sha256,
        "intra_cache_num": 1,
        "multibuffer_num": 1,
        "vf_merge_level": prepared.vf_merge_level,
    }
    for field_name, expected_value in expected.items():
        actual = getattr(baseline, field_name)
        if actual is not None and actual != expected_value:
            raise UnsupportedModelError(f"(d,m)=(1,1) baseline certificate {field_name} is {actual!r}, "
                                        f"expected {expected_value!r}")
    if baseline.ir_sha256 is not None and prepared.source_intra_cache_num == 1:
        if baseline.ir_sha256 != prepared.source_ir_sha256:
            raise UnsupportedModelError("(1,1) baseline raw IR identity does not match the d=1 model input")
    if baseline.ub_peak_bytes < 0:
        raise UnsupportedModelError("baseline ub_peak_bytes must be non-negative")
    if baseline.ub_peak_bytes > prepared.ub_capacity_bytes:
        raise UnsupportedModelError("successful baseline UB peak exceeds the profile capacity")
    if baseline.no_reuse_required_bytes is not None and baseline.no_reuse_required_bytes < 0:
        raise UnsupportedModelError("baseline no_reuse_required_bytes must be non-negative")
    if baseline.used_reuse is False:
        if baseline.no_reuse_required_bytes is None:
            raise UnsupportedModelError("used_reuse=false requires no_reuse_required_bytes")
        if baseline.no_reuse_required_bytes > prepared.ub_capacity_bytes:
            raise UnsupportedModelError("used_reuse=false contradicts a no-reuse total above UB capacity")
        if baseline.ub_peak_bytes != baseline.no_reuse_required_bytes:
            raise UnsupportedModelError("no-reuse baseline peak must equal its aligned required total")
    if baseline.used_reuse is True and baseline.no_reuse_required_bytes is not None:
        if baseline.no_reuse_required_bytes <= prepared.ub_capacity_bytes:
            raise UnsupportedModelError("used_reuse=true contradicts PlanMemory's no-reuse capacity branch")
    return (prepared.profile_provenance_verified
            and all(getattr(baseline, field_name) is not None for field_name in expected))


def make_baseline_certificate(
        prepared: PreparedCostModel,
        *,
        ub_peak_bytes: int,
        no_reuse_required_bytes: int,
        used_reuse: bool,
        entries: Sequence[MemoryEntry] = (),
        entry_graph_complete: bool = False,
) -> BaselineCertificate:
    """Bind one real (d,m,v)=(1,1,0) PlanMemory result to a prepared model."""

    certificate = BaselineCertificate(
        ub_peak_bytes=ub_peak_bytes,
        no_reuse_required_bytes=no_reuse_required_bytes,
        used_reuse=used_reuse,
        profile_id=prepared.profile,
        profile_fingerprint=prepared.profile_fingerprint,
        model_fingerprint=prepared.model_fingerprint,
        ir_sha256=prepared.source_ir_sha256 if prepared.source_intra_cache_num == 1 else None,
        normalized_ir_sha256=prepared.normalized_ir_sha256,
        intra_cache_num=1,
        multibuffer_num=1,
        vf_merge_level=prepared.vf_merge_level,
        entries=tuple(entries),
        entry_graph_complete=entry_graph_complete,
    )
    _validate_baseline(prepared, certificate)
    return certificate


def _unknown_estimate(
        intra_cache_num: int,
        multibuffer_num: int,
        *,
        reason: str,
        dynamic_delta: Optional[int] = None,
        ordinary_delta: Optional[int] = None,
        interaction_delta: Optional[int] = None,
        total_delta: Optional[int] = None,
        predicted_ub: Optional[int] = None,
        no_reuse_required: Optional[int] = None,
        confidence: str = "none",
        evaluate_us: float = 0.0,
        interaction_contributions: Sequence[InteractionContribution] = (),
) -> CostEstimate:
    return CostEstimate(
        intra_cache_num=intra_cache_num,
        multibuffer_num=multibuffer_num,
        dynamic_from_d1_bytes=dynamic_delta,
        ordinary_from_m1_bytes=ordinary_delta,
        interaction_bytes=interaction_delta,
        total_from_11_bytes=total_delta,
        predicted_ub_bytes=predicted_ub,
        no_reuse_required_bytes=no_reuse_required,
        verdict="unknown",
        confidence=confidence,
        exact_no_reuse_path=False,
        prune_allowed=False,
        reason=reason,
        evaluate_us=evaluate_us,
        interaction_contributions=tuple(interaction_contributions),
    )


def _delta_components(
    prepared: PreparedCostModel,
    intra_cache_num: int,
    multibuffer_num: int,
) -> tuple[int, int, int, int]:
    d_index = intra_cache_num - 1
    m_index = multibuffer_num - 1
    dynamic_delta = prepared.dynamic_delta_table_bytes[d_index]
    ordinary_delta = m_index * prepared.ordinary_step_table_bytes[0]
    ordinary_at_d = m_index * prepared.ordinary_step_table_bytes[d_index]
    coupled = prepared.coupled_adjustment_table_bytes[d_index][m_index]
    total_delta = dynamic_delta + ordinary_at_d + coupled
    interaction_delta = total_delta - dynamic_delta - ordinary_delta
    return dynamic_delta, ordinary_delta, interaction_delta, total_delta


def evaluate_configuration(
    prepared: PreparedCostModel,
    *,
    intra_cache_num: int,
    multibuffer_num: int,
    baseline: Optional[BaselineCertificate],
    ub_capacity_bytes: Optional[int] = None,
) -> CostEstimate:
    evaluate_start = time.perf_counter_ns()
    if intra_cache_num not in range(1, 4):
        raise UnsupportedModelError("intra_cache_num must be one of 1, 2, 3")
    if multibuffer_num not in range(1, 5):
        raise UnsupportedModelError("multibuffer_num must be one of 1, 2, 3, 4")
    capacity = prepared.ub_capacity_bytes if ub_capacity_bytes is None else ub_capacity_bytes
    if capacity != prepared.ub_capacity_bytes:
        raise UnsupportedModelError(
            f"UB capacity {capacity} does not match profile capacity {prepared.ub_capacity_bytes}")
    if prepared.blockers:
        return _unknown_estimate(
            intra_cache_num,
            multibuffer_num,
            reason="IR projection has blockers: " + "; ".join(prepared.blockers),
            evaluate_us=(time.perf_counter_ns() - evaluate_start) / 1_000.0,
        )

    dynamic_delta, ordinary_delta, interaction_delta, total_delta = _delta_components(
        prepared,
        intra_cache_num,
        multibuffer_num,
    )
    active_interactions = tuple(item for item in prepared.interactions
                                if item.intra_cache_num == intra_cache_num and item.multibuffer_num == multibuffer_num)
    explained_interaction = sum(item.contribution_bytes for item in active_interactions)
    if interaction_delta != explained_interaction:
        return _unknown_estimate(
            intra_cache_num,
            multibuffer_num,
            dynamic_delta=dynamic_delta,
            ordinary_delta=ordinary_delta,
            interaction_delta=None,
            total_delta=None,
            confidence="incomplete_interaction_ledger",
            reason=(f"the numeric interaction is {interaction_delta} bytes but origin-level provenance "
                    f"explains {explained_interaction} bytes; total Delta is not publishable"),
            interaction_contributions=active_interactions,
            evaluate_us=(time.perf_counter_ns() - evaluate_start) / 1_000.0,
        )
    if baseline is None:
        return _unknown_estimate(
            intra_cache_num,
            multibuffer_num,
            dynamic_delta=dynamic_delta,
            ordinary_delta=ordinary_delta,
            interaction_delta=interaction_delta,
            total_delta=total_delta,
            confidence="pure_delta",
            reason=("the structural model predicts Delta from (d,m)=(1,1); no matching real baseline "
                    "was supplied, so absolute UB and overflow remain unknown"),
            interaction_contributions=active_interactions,
            evaluate_us=(time.perf_counter_ns() - evaluate_start) / 1_000.0,
        )

    identity_verified = _validate_baseline(prepared, baseline)
    predicted_ub = baseline.ub_peak_bytes + total_delta
    no_reuse_required = (None if baseline.no_reuse_required_bytes is None else baseline.no_reuse_required_bytes +
                         total_delta)
    evaluate_us = (time.perf_counter_ns() - evaluate_start) / 1_000.0
    if (identity_verified and baseline.used_reuse is False and no_reuse_required is not None
            and no_reuse_required <= capacity):
        return CostEstimate(
            intra_cache_num=intra_cache_num,
            multibuffer_num=multibuffer_num,
            dynamic_from_d1_bytes=dynamic_delta,
            ordinary_from_m1_bytes=ordinary_delta,
            interaction_bytes=interaction_delta,
            total_from_11_bytes=total_delta,
            predicted_ub_bytes=predicted_ub,
            no_reuse_required_bytes=no_reuse_required,
            verdict="safe",
            confidence="exact_calibrated_no_reuse",
            exact_no_reuse_path=True,
            prune_allowed=False,
            reason=("the identity-matched (1,1) compiler certificate proves the no-reuse branch, and "
                    "the modeled parameterized total remains within UB capacity"),
            evaluate_us=evaluate_us,
            interaction_contributions=active_interactions,
        )

    reason = ("the calibrated UB value is available, but the baseline identity or no-reuse branch is "
              "not strong enough to prove safety or overflow; continue real compilation")
    if identity_verified and no_reuse_required is not None and no_reuse_required > capacity:
        reason = ("the parameterized no-reuse total exceeds capacity, but PlanMemory may reuse addresses; "
                  "without a complete lifetime graph this is not an overflow proof")
    return _unknown_estimate(
        intra_cache_num,
        multibuffer_num,
        dynamic_delta=dynamic_delta,
        ordinary_delta=ordinary_delta,
        interaction_delta=interaction_delta,
        total_delta=total_delta,
        predicted_ub=predicted_ub,
        no_reuse_required=no_reuse_required,
        confidence="calibrated" if identity_verified else "calibrated_unverified",
        reason=reason,
        evaluate_us=evaluate_us,
        interaction_contributions=active_interactions,
    )


def evaluate_all_configurations(
    prepared: PreparedCostModel,
    baseline: Optional[BaselineCertificate] = None,
    ub_capacity_bytes: Optional[int] = None,
) -> list[CostEstimate]:
    return [
        evaluate_configuration(
            prepared,
            intra_cache_num=intra_cache_num,
            multibuffer_num=multibuffer_num,
            baseline=baseline,
            ub_capacity_bytes=ub_capacity_bytes,
        ) for intra_cache_num in range(1, 4) for multibuffer_num in range(1, 5)
    ]
