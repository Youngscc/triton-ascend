"""Compiler profile, model identity, and applicability validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from ..model_types import IRGraph, UnsupportedModelError

PROFILE_NAME = "a5-pcb-dynamiccv-multibuffer-v1"

MODEL_VERSION = "a5-pcb-dependency-cut-multibuffer-v3"

DYNAMIC_REDUCTION_RULE_ID = "direct-inner-cache-count-d1-d3-v1"

DYNAMIC_FAMILY_RULE_ID = "pcb-ub-usage-min-cut-v1"

DYNAMIC_SELECTOR_EFFECTIVE_COUNTS = (1, 2, 3)

DEFAULT_UB_CAPACITY_BYTES = 2_031_616 // 8

UB_ALIGNMENT_BYTES = 256 // 8

AIV_SUBBLOCK_FACTOR = 2

DEFAULT_TRITON_ASCEND_REVISION = "eefead67a181944e7a44c3a0586c06e2d4f4d265"

DEFAULT_ASCEND_NPU_IR_REVISION = "4b9f1a56092d66a991b857ca4ca2b40f2cf06e53"

DEFAULT_LLVM_REVISION = "d3fea2c7ae5436f63fa35b4d01e0aa76d1071396"

COUNT_ATTRIBUTE_RE = re.compile(r"(ssbuffer\.(?:intra_buf_count|inter_core_buf_count|load_store_buf_count)\s*=\s*)"
                                r"-?[0-9]+(\s*:\s*i[0-9]+)")

COUNT_ATTRIBUTE_VALUE_RE = re.compile(
    r"(ssbuffer\.(?:intra_buf_count|inter_core_buf_count|load_store_buf_count))\s*=\s*"
    r"(-?[0-9]+)\s*:\s*i[0-9]+")


@dataclass(frozen=True)
class CompilerProfile:
    """Compiler facts that make the projection rules valid.

    The compiler-integrated caller constructs this record from its already
    resolved options and source identity.  The standalone CLI uses the frozen
    validation profile below, or a JSON profile supplied by the caller.
    """

    profile_id: str = PROFILE_NAME
    target_prefix: str = "Ascend950"
    triton_ascend_revision: str = DEFAULT_TRITON_ASCEND_REVISION
    ascend_npu_ir_revision: str = DEFAULT_ASCEND_NPU_IR_REVISION
    llvm_revision: str = DEFAULT_LLVM_REVISION
    enable_triton_kernel_compile: bool = True
    enable_dynamic_cv_pipeline: bool = True
    enable_auto_bind_sub_block: bool = True
    enable_auto_multi_buffer: bool = True
    enable_auto_blockify_loop: bool = True
    enable_hfusion_compile: bool = True
    enable_mixed_cv: bool = True
    enable_preload: bool = False
    enable_ubuf_saving: bool = False
    disable_align_alloc_size: bool = False
    disable_enable_stride_align: bool = False
    disable_infer_hivm_data_layout: bool = False
    set_workspace_multibuffer: int = 0
    limit_auto_multi_buffer_of_local_buffer: str = "no-limit"
    limit_auto_multi_buffer_buffer: str = "no-limit"
    enable_hfusion_auto_schedule: bool = False
    inter_cache_num: int = 1
    load_cache_num: int = 1
    aiv_subblock_factor: int = AIV_SUBBLOCK_FACTOR
    aiv_tiling_dimension: int = 0
    gm_load_vector_buffer_count: int = 2
    gm_load_cube_buffer_count: int = 1
    gm_load_skip_functions: tuple[str, ...] = ()
    ub_capacity_bytes: int = DEFAULT_UB_CAPACITY_BYTES
    ub_alignment_bytes: int = UB_ALIGNMENT_BYTES

    def __post_init__(self) -> None:
        boolean_fields = (
            "enable_triton_kernel_compile",
            "enable_dynamic_cv_pipeline",
            "enable_auto_bind_sub_block",
            "enable_auto_multi_buffer",
            "enable_auto_blockify_loop",
            "enable_hfusion_compile",
            "enable_mixed_cv",
            "enable_preload",
            "enable_ubuf_saving",
            "disable_align_alloc_size",
            "disable_enable_stride_align",
            "disable_infer_hivm_data_layout",
            "enable_hfusion_auto_schedule",
        )
        for field_name in boolean_fields:
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"compiler profile field {field_name} must be boolean")
        integer_fields = (
            "set_workspace_multibuffer",
            "inter_cache_num",
            "load_cache_num",
            "aiv_subblock_factor",
            "aiv_tiling_dimension",
            "gm_load_vector_buffer_count",
            "gm_load_cube_buffer_count",
            "ub_capacity_bytes",
            "ub_alignment_bytes",
        )
        for field_name in integer_fields:
            if type(getattr(self, field_name)) is not int:
                raise ValueError(f"compiler profile field {field_name} must be an integer")
        if type(self.gm_load_skip_functions) is not tuple or not all(
                isinstance(name, str) for name in self.gm_load_skip_functions):
            raise ValueError("compiler profile field gm_load_skip_functions must be a tuple of strings")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompilerProfile":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - known)
        if unknown:
            raise UnsupportedModelError(f"unknown compiler profile fields: {', '.join(unknown)}")
        data = dict(value)
        if "gm_load_skip_functions" in data:
            data["gm_load_skip_functions"] = tuple(data["gm_load_skip_functions"])
        return cls(**data)


DEFAULT_PROFILE = CompilerProfile()


@dataclass(frozen=True)
class ValidatedContext:
    profile: CompilerProfile
    profile_provenance_verified: bool
    profile_guard_notes: tuple[str, ...]
    source_ir_sha256: str
    normalized_ir_sha256: str
    source_intra_cache_num: int
    vf_merge_level: int


def normalize_dynamic_count_attributes(text: str) -> str:
    """Remove only DynamicCV count values from an otherwise exact IR identity."""

    return COUNT_ATTRIBUTE_RE.sub(r"\1<count>\2", text)


def normalized_ir_sha256(text: str) -> str:
    return hashlib.sha256(normalize_dynamic_count_attributes(text).encode("utf-8")).hexdigest()


def model_profile_fingerprint(profile: CompilerProfile) -> str:
    payload = {
        "compiler_profile_fingerprint": profile.fingerprint,
        "model_version": MODEL_VERSION,
        "dynamic_reduction_rule": DYNAMIC_REDUCTION_RULE_ID,
        "dynamic_family_rule": DYNAMIC_FAMILY_RULE_ID,
        "dynamic_selector_effective_counts": DYNAMIC_SELECTOR_EFFECTIVE_COUNTS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_context(
    source_text: str,
    parsed: IRGraph,
    vf_merge_level: int,
    profile: CompilerProfile,
    observed_provenance: Optional[Mapping[str, Any]],
) -> ValidatedContext:
    if not parsed.target.startswith(profile.target_prefix):
        raise UnsupportedModelError(
            f"profile requires target prefix {profile.target_prefix}, got {parsed.target or 'missing'}")
    required_true = {
        "enable_triton_kernel_compile": profile.enable_triton_kernel_compile,
        "enable_dynamic_cv_pipeline": profile.enable_dynamic_cv_pipeline,
        "enable_auto_bind_sub_block": profile.enable_auto_bind_sub_block,
        "enable_auto_multi_buffer": profile.enable_auto_multi_buffer,
        "enable_auto_blockify_loop": profile.enable_auto_blockify_loop,
        "enable_hfusion_compile": profile.enable_hfusion_compile,
        "enable_mixed_cv": profile.enable_mixed_cv,
    }
    disabled = sorted(name for name, enabled in required_true.items() if not enabled)
    if disabled:
        raise UnsupportedModelError(f"profile disables required options: {', '.join(disabled)}")
    if profile.enable_hfusion_auto_schedule:
        raise UnsupportedModelError("profile contains HFusionAutoSchedule; m-dependent tiling is unsupported")
    if profile.limit_auto_multi_buffer_buffer != "no-limit":
        raise UnsupportedModelError("profile must allow Vector ordinary buffers with MIX strategy no-limit")
    if profile.limit_auto_multi_buffer_of_local_buffer != "no-limit":
        raise UnsupportedModelError("profile local-buffer strategy must be no-limit")
    required_false = {
        "enable_preload": profile.enable_preload,
        "enable_ubuf_saving": profile.enable_ubuf_saving,
        "disable_align_alloc_size": profile.disable_align_alloc_size,
        "disable_enable_stride_align": profile.disable_enable_stride_align,
        "disable_infer_hivm_data_layout": profile.disable_infer_hivm_data_layout,
    }
    unexpected_enabled = sorted(name for name, value in required_false.items() if value)
    if unexpected_enabled:
        raise UnsupportedModelError(f"profile enables unsupported options: {', '.join(unexpected_enabled)}")
    if profile.set_workspace_multibuffer != 0:
        raise UnsupportedModelError("validated DynamicCV profile requires set_workspace_multibuffer=0")
    if parsed.has_fallback:
        raise UnsupportedModelError("DynamicCV input already carries a fallback marker")
    if profile.aiv_tiling_dimension != 0:
        raise UnsupportedModelError("validated profile supports AIV tiling dimension 0 only")
    if profile.aiv_subblock_factor != 2:
        raise UnsupportedModelError("validated profile requires two AIV sub-blocks")
    if profile.ub_capacity_bytes <= 0 or profile.ub_alignment_bytes <= 0:
        raise UnsupportedModelError("profile UB capacity and alignment must be positive")
    expected_attributes = {
        "ssbuffer.inter_core_buf_count": profile.inter_cache_num,
        "ssbuffer.load_store_buf_count": profile.load_cache_num,
    }
    for name, expected in expected_attributes.items():
        actual = parsed.module_attributes.get(name)
        if actual != expected:
            raise UnsupportedModelError(f"{name} is {actual}, expected {expected}")
    source_intra_cache_num = parsed.module_attributes.get("ssbuffer.intra_buf_count")
    if source_intra_cache_num not in range(1, 4):
        raise UnsupportedModelError("ssbuffer.intra_buf_count must be one of 1, 2, 3")
    if vf_merge_level != 0:
        raise UnsupportedModelError("validated profile supports vf_merge_level=0 only")

    notes: list[str] = []
    provenance_verified = False
    if observed_provenance is None:
        notes.append("compiler revisions are declared by the profile but were not independently verified")
    else:
        revision_pairs = {
            "top_level_revision": profile.triton_ascend_revision,
            "ascend_npu_ir_revision": profile.ascend_npu_ir_revision,
            "llvm_revision": profile.llvm_revision,
        }
        for field_name, expected in revision_pairs.items():
            actual = observed_provenance.get(field_name)
            if actual != expected:
                raise UnsupportedModelError(f"{field_name} is {actual!r}, expected {expected!r}")
        provenance_verified = True
        notes.append("compiler revisions match the frozen profile")

    return ValidatedContext(
        profile=profile,
        profile_provenance_verified=provenance_verified,
        profile_guard_notes=tuple(notes),
        source_ir_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        normalized_ir_sha256=normalized_ir_sha256(source_text),
        source_intra_cache_num=source_intra_cache_num,
        vf_merge_level=vf_merge_level,
    )
