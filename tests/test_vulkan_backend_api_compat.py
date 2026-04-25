import inspect

import pytest
import torch

from turboquant import cuda_backend
from turboquant import vulkan_backend


def test_vulkan_backend_signatures_match_cuda_backend():
    funcs = [
        "qjl_quant",
        "qjl_score",
        "qjl_gqa_score",
        "quantized_bmm",
    ]
    for name in funcs:
        assert inspect.signature(getattr(vulkan_backend, name)) == inspect.signature(
            getattr(cuda_backend, name)
        )


def test_is_vulkan_available_returns_bool():
    assert isinstance(vulkan_backend.is_vulkan_available(), bool)


def test_qjl_quant_unsupported_dtype_raises_typeerror():
    key_states = torch.zeros(1, 1, 1, 1, 8, dtype=torch.int32)
    outlier_indices = torch.zeros(1, 1, 1, 1, dtype=torch.int64)
    rand_prj = torch.zeros(8, 8, dtype=torch.float32)
    with pytest.raises(TypeError, match="Unsupported dtypes"):
        vulkan_backend.qjl_quant(key_states, outlier_indices, rand_prj, 8)


def test_qjl_score_unsupported_dtype_raises_typeerror():
    key_quant = torch.zeros(1, 1, 1, 1, 1, dtype=torch.uint8)
    key_outlier_quant = torch.zeros(1, 1, 1, 1, 1, dtype=torch.uint8)
    key_norm = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    key_outlier_norm = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    outlier_indices = torch.zeros(1, 1, 1, 1, dtype=torch.int64)
    query_sketch = torch.zeros(1, 1, 8, dtype=torch.float32)
    query_states = torch.zeros(1, 1, 8, dtype=torch.int32)
    rand_prj = torch.zeros(8, 8, dtype=torch.float32)
    with pytest.raises(TypeError, match="Unsupported dtypes"):
        vulkan_backend.qjl_score(
            key_quant,
            key_outlier_quant,
            key_norm,
            key_outlier_norm,
            outlier_indices,
            query_sketch,
            query_states,
            rand_prj,
        )


def test_quantized_bmm_invalid_bits_raises_assertion():
    fA = torch.zeros(1, 1, 1, 8, dtype=torch.float16)
    qB = torch.zeros(1, 1, 8, 1, dtype=torch.int32)
    scales = torch.zeros(1, 1, 8, 1, dtype=torch.float16)
    zeros = torch.zeros(1, 1, 8, 1, dtype=torch.float16)
    with pytest.raises(AssertionError):
        vulkan_backend.quantized_bmm(8, fA, qB, scales, zeros, bits=3)


def test_runtime_gate_raises_when_vulkan_not_ready(monkeypatch):
    monkeypatch.setattr(vulkan_backend, "_VULKAN_EXT_AVAILABLE", False)
    monkeypatch.setattr(vulkan_backend, "_VULKAN_LOAD_ERROR", "mock-load-error")

    key_states = torch.zeros(1, 1, 1, 1, 8, dtype=torch.float16)
    outlier_indices = torch.zeros(1, 1, 1, 1, dtype=torch.int64)
    rand_prj = torch.zeros(8, 8, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="Vulkan backend extension is unavailable"):
        vulkan_backend.qjl_quant(key_states, outlier_indices, rand_prj, 8)
