"""Main orchestration and public API for the UB cost model."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from typing import Any, Mapping, Optional

from .model_types import (
    AutotuneDecision,
    BaselineCertificate,
    CostEstimate,
    PreparedCostModel,
    StageTimer,
    UnsupportedModelError,
)
from .stages.analyze_buffer_families import analyze_buffer_families
from .stages.analyze_dynamic_cv import analyze_dynamic_cv
from .stages.build_parametric_model import build_parametric_model
from .stages.evaluate import _unknown_estimate, evaluate_all_configurations, evaluate_configuration
from .stages.prepare_input import prepare_input
from .stages.validate_context import (
    COUNT_ATTRIBUTE_VALUE_RE,
    DEFAULT_PROFILE,
    CompilerProfile,
    model_profile_fingerprint,
    normalized_ir_sha256,
)


def prepare_cost_model(
    ir: Any,
    *,
    vf_merge_level: int = 0,
    profile: CompilerProfile = DEFAULT_PROFILE,
    observed_provenance: Optional[Mapping[str, Any]] = None,
) -> PreparedCostModel:
    """Run stages 1-4 once and return the reusable parameter model."""

    text = ir if isinstance(ir, str) else str(ir)
    timer = StageTimer()
    total_start = time.perf_counter_ns()

    graph, context = timer.measure(
        "stage1_prepare_input",
        lambda: prepare_input(
            text,
            vf_merge_level=vf_merge_level,
            profile=profile,
            observed_provenance=observed_provenance,
        ),
    )
    dynamic = timer.measure(
        "stage2_analyze_dynamic_cv",
        lambda: analyze_dynamic_cv(graph, context.profile),
    )
    analysis = timer.measure(
        "stage3_resolve_buffer_families",
        lambda: analyze_buffer_families(graph, context.profile, dynamic),
    )
    prepared = timer.measure(
        "stage4_build_parametric_model",
        lambda: build_parametric_model(graph, context, analysis),
    )

    timing_us = timer.as_microseconds()
    timing_us["prepare_total"] = (time.perf_counter_ns() - total_start) / 1_000.0
    return replace(prepared, timing_us=timing_us)


def run_cost_model(
    ir: Any,
    *,
    intra_cache_num: int,
    multibuffer_num: int,
    vf_merge_level: int = 0,
    profile: CompilerProfile = DEFAULT_PROFILE,
    observed_provenance: Optional[Mapping[str, Any]] = None,
    baseline: Optional[BaselineCertificate] = None,
) -> CostEstimate:
    """Run all five stages for one parameter configuration."""

    prepared = prepare_cost_model(
        ir,
        vf_merge_level=vf_merge_level,
        profile=profile,
        observed_provenance=observed_provenance,
    )
    return evaluate_configuration(
        prepared,
        intra_cache_num=intra_cache_num,
        multibuffer_num=multibuffer_num,
        baseline=baseline,
    )


class UbCostModel:
    """Cache stages 1-4, then use stage 5 for one or more d/m pairs."""
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, str, str, str], PreparedCostModel] = {}

    def prepare(
        self,
        ir: Any,
        *,
        vf_merge_level: int = 0,
        profile: CompilerProfile = DEFAULT_PROFILE,
        observed_provenance: Optional[Mapping[str, Any]] = None,
    ) -> PreparedCostModel:
        text = ir if isinstance(ir, str) else str(ir)
        counts = {name: int(value) for name, value in COUNT_ATTRIBUTE_VALUE_RE.findall(text)}
        if counts.get("ssbuffer.intra_buf_count") not in range(1, 4):
            raise UnsupportedModelError("input must carry ssbuffer.intra_buf_count in {1,2,3}")
        for name, expected in (
            ("ssbuffer.inter_core_buf_count", profile.inter_cache_num),
            ("ssbuffer.load_store_buf_count", profile.load_cache_num),
        ):
            if counts.get(name) != expected:
                raise UnsupportedModelError(f"{name} is {counts.get(name)!r}, expected {expected}")

        provenance_key = hashlib.sha256(json.dumps(observed_provenance, sort_keys=True,
                                                   default=str).encode("utf-8")).hexdigest()
        key = (
            normalized_ir_sha256(text),
            vf_merge_level,
            profile.fingerprint,
            model_profile_fingerprint(profile),
            provenance_key,
        )
        if key not in self._cache:
            self._cache[key] = prepare_cost_model(
                text,
                vf_merge_level=vf_merge_level,
                profile=profile,
                observed_provenance=observed_provenance,
            )
        return self._cache[key]

    def evaluate(
        self,
        prepared: PreparedCostModel,
        *,
        intra_cache_num: int,
        multibuffer_num: int,
        baseline: Optional[BaselineCertificate] = None,
    ) -> CostEstimate:
        return evaluate_configuration(
            prepared,
            intra_cache_num=intra_cache_num,
            multibuffer_num=multibuffer_num,
            baseline=baseline,
        )

    def evaluate_all(
        self,
        prepared: PreparedCostModel,
        baseline: Optional[BaselineCertificate] = None,
    ) -> list[CostEstimate]:
        return evaluate_all_configurations(prepared, baseline)

    def decide(
        self,
        prepared: PreparedCostModel,
        *,
        intra_cache_num: int,
        multibuffer_num: int,
        baseline: Optional[BaselineCertificate] = None,
    ) -> AutotuneDecision:
        try:
            result = self.evaluate(
                prepared,
                intra_cache_num=intra_cache_num,
                multibuffer_num=multibuffer_num,
                baseline=baseline,
            )
        except (UnsupportedModelError, ValueError) as error:
            result = _unknown_estimate(
                intra_cache_num,
                multibuffer_num,
                reason=f"cost-model evaluation is unsupported: {error}",
            )
        if result.verdict == "overflow" and result.prune_allowed:
            return AutotuneDecision("prune_predicted_ub_overflow", result, result.reason)
        return AutotuneDecision(
            "continue_real_compilation",
            result,
            "safe and unknown outcomes preserve the existing compiler and fallback path",
        )
