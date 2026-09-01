"""Shared logical-to-physical UB geometry helpers."""

from __future__ import annotations

import math
import re

from .validate_context import CompilerProfile
from ..model_types import UnsupportedModelError


def dtype_bits(dtype: str) -> int:
    if dtype == "bf16":
        return 16
    match = re.fullmatch(r"[fi]([0-9]+)", dtype)
    if not match:
        raise UnsupportedModelError(f"unsupported element type: {dtype}")
    return int(match.group(1))


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def project_physical_shape(
    shape: tuple[int, ...],
    core_type: str,
    profile: CompilerProfile,
) -> tuple[int, ...]:
    if core_type != "VECTOR":
        return shape
    if not shape:
        raise UnsupportedModelError("cannot AIV-split a rank-0 buffer")
    dimension = profile.aiv_tiling_dimension
    if dimension < 0 or dimension >= len(shape):
        raise UnsupportedModelError(f"AIV tiling dimension {dimension} is invalid for rank-{len(shape)} buffer")
    factor = profile.aiv_subblock_factor
    if factor < 1 or shape[dimension] % factor != 0:
        raise UnsupportedModelError(
            f"extent {shape[dimension]} at dimension {dimension} is not divisible by AIV factor {factor}")
    result = list(shape)
    result[dimension] //= factor
    return tuple(result)


def aligned_physical_bytes(
    shape: tuple[int, ...],
    dtype: str,
    core_type: str,
    profile: CompilerProfile,
) -> tuple[tuple[int, ...], int, int]:
    physical_shape = project_physical_shape(shape, core_type, profile)
    element_bits = dtype_bits(dtype)
    raw_bytes = (math.prod(physical_shape) * element_bits + 7) // 8
    return physical_shape, element_bits, align_up(raw_bytes, profile.ub_alignment_bytes)
