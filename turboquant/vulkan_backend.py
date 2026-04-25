"""
Vulkan backend API wrapper for TurboQuant.

This module mirrors the public CUDA backend API while routing through Vulkan
capability checks. Kernel dispatch wiring is staged and will be connected to
runtime execution paths in follow-up tasks.
"""

from __future__ import annotations

import torch

_VULKAN_EXT_AVAILABLE = False
_VULKAN_LOAD_ERROR = ""
_vulkan_ext = None

try:
    import importlib

    _vulkan_ext = importlib.import_module("turboquant.vulkan_backend_ext")
    _VULKAN_EXT_AVAILABLE = True
except Exception as e:  # pragma: no cover - exercised through behavior tests.
    _VULKAN_EXT_AVAILABLE = False
    _VULKAN_LOAD_ERROR = str(e)


def is_vulkan_available():
    if not _VULKAN_EXT_AVAILABLE:
        return False
    try:
        return bool(_vulkan_ext.is_vulkan_runtime_available())
    except Exception:
        return False


def _require_vulkan_ready(op_name: str):
    if not _VULKAN_EXT_AVAILABLE:
        detail = f" ({_VULKAN_LOAD_ERROR})" if _VULKAN_LOAD_ERROR else ""
        raise RuntimeError(
            f"Vulkan backend extension is unavailable for {op_name}{detail}. "
            "Build with --vulkan and ensure the extension can be imported."
        )
    if not bool(_vulkan_ext.is_vulkan_runtime_available()):
        runtime_info = ""
        try:
            runtime_info = str(_vulkan_ext.vulkan_runtime_info())
        except Exception:
            runtime_info = "runtime info unavailable"
        raise RuntimeError(
            f"Vulkan runtime is unavailable for {op_name}: {runtime_info}. "
            "Fallback backend should be selected by higher-level dispatch."
        )


def qjl_quant(key_states, outlier_indices, rand_prj, outlier_sketch_dim):
    key_dtype = key_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): "qjl_quant_half_half",
        (torch.half, torch.float): "qjl_quant_half_float",
        (torch.float, torch.float): "qjl_quant_float_float",
        (torch.bfloat16, torch.bfloat16): "qjl_quant_bf16_bf16",
        (torch.bfloat16, torch.float): "qjl_quant_bf16_float",
    }
    fn_name = dispatch.get((key_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: key={key_dtype}, proj={rand_dtype}")

    _require_vulkan_ready("qjl_quant")
    raise NotImplementedError(
        f"Vulkan kernel dispatch for qjl_quant ({fn_name}) is not wired yet."
    )


def qjl_score(
    key_quant,
    key_outlier_quant,
    key_norm,
    key_outlier_norm,
    outlier_indices,
    query_sketch,
    query_states,
    rand_prj,
):
    query_dtype = query_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): "qjl_score_vulkan_half_half",
        (torch.half, torch.float): "qjl_score_vulkan_half_float",
        (torch.float, torch.float): "qjl_score_vulkan_float_float",
        (torch.bfloat16, torch.bfloat16): "qjl_score_vulkan_bf16_bf16",
        (torch.bfloat16, torch.float): "qjl_score_vulkan_bf16_float",
    }
    fn_name = dispatch.get((query_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: query={query_dtype}, proj={rand_dtype}")

    _require_vulkan_ready("qjl_score")
    raise NotImplementedError(
        f"Vulkan kernel dispatch for qjl_score ({fn_name}) is not wired yet."
    )


def qjl_gqa_score(
    key_quant,
    key_outlier_quant,
    key_norm,
    key_outlier_norm,
    outlier_indices,
    query_sketch,
    query_states,
    rand_prj,
):
    query_dtype = query_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): "qjl_gqa_score_vulkan_half_half",
        (torch.half, torch.float): "qjl_gqa_score_vulkan_half_float",
        (torch.float, torch.float): "qjl_gqa_score_vulkan_float_float",
        (torch.bfloat16, torch.bfloat16): "qjl_gqa_score_vulkan_bf16_bf16",
        (torch.bfloat16, torch.float): "qjl_gqa_score_vulkan_bf16_float",
    }
    fn_name = dispatch.get((query_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: query={query_dtype}, proj={rand_dtype}")

    _require_vulkan_ready("qjl_gqa_score")
    raise NotImplementedError(
        f"Vulkan kernel dispatch for qjl_gqa_score ({fn_name}) is not wired yet."
    )


def quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False):
    assert len(fA.shape) == 4 and len(qB.shape) == 4
    _ = group_size, scales, zeros, mqa
    assert bits in [2, 4]

    dispatch = {
        torch.float16: "quantized_bmm_vulkan_half",
        torch.float32: "quantized_bmm_vulkan_float",
        torch.bfloat16: "quantized_bmm_vulkan_bf16",
    }
    fn_name = dispatch.get(fA.dtype)
    if fn_name is None:
        raise TypeError(f"Unsupported dtype: {fA.dtype}")

    _require_vulkan_ready("quantized_bmm")
    raise NotImplementedError(
        f"Vulkan kernel dispatch for quantized_bmm ({fn_name}) is not wired yet."
    )
