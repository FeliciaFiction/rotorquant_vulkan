import types

import torch

import turboquant.vulkan_backend as vk
from turboquant.vulkan.reference_ops import qjl_score_reference


def _mock_ready_report():
    return {
        "strictly_available": True,
        "checklist": {
            "runtime_available": True,
            "vulkan_1_3_minimum": True,
            "required_extensions": True,
            "required_features": True,
        },
        "missing_extensions": [],
        "missing_features": [],
    }


def test_qjl_score_uses_reference_path_when_extension_symbol_missing(monkeypatch):
    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(vk, "_vulkan_ext", types.SimpleNamespace())
    monkeypatch.setattr(vk, "get_vulkan_capability_report", _mock_ready_report)

    key_quant = torch.randint(0, 255, (1, 1, 2, 2, 1), dtype=torch.uint8)
    key_outlier_quant = torch.randint(0, 255, (1, 1, 2, 2, 1), dtype=torch.uint8)
    key_norm = torch.rand(1, 1, 2, 2, dtype=torch.float16) + 1.0
    key_outlier_norm = key_norm * 0.3
    outlier_indices = torch.tensor([[[[0], [3]]]], dtype=torch.int64)
    query_sketch = torch.randn(1, 1, 8, dtype=torch.float32)
    query_states = torch.randn(1, 1, 8, dtype=torch.float16)
    rand_prj = torch.randn(8, 8, dtype=torch.float16)

    got = vk.qjl_score(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )
    exp = qjl_score_reference(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )
    assert torch.allclose(got, exp, atol=1e-3, rtol=1e-3)


def test_qjl_score_prefers_extension_symbol_when_available(monkeypatch):
    sentinel = torch.zeros(1, 1, 1, 1, dtype=torch.float32)

    class _Ext:
        def qjl_score_vulkan_half_half(self, *_args, **_kwargs):
            return sentinel

    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(vk, "_vulkan_ext", _Ext())
    monkeypatch.setattr(vk, "get_vulkan_capability_report", _mock_ready_report)

    key_quant = torch.zeros(1, 1, 1, 1, 1, dtype=torch.uint8)
    key_outlier_quant = torch.zeros(1, 1, 1, 1, 1, dtype=torch.uint8)
    key_norm = torch.zeros(1, 1, 1, 1, dtype=torch.float16)
    key_outlier_norm = torch.zeros(1, 1, 1, 1, dtype=torch.float16)
    outlier_indices = torch.zeros(1, 1, 1, 1, dtype=torch.int64)
    query_sketch = torch.zeros(1, 1, 8, dtype=torch.float32)
    query_states = torch.zeros(1, 1, 8, dtype=torch.float16)
    rand_prj = torch.zeros(8, 8, dtype=torch.float16)

    got = vk.qjl_score(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )
    assert got is sentinel
