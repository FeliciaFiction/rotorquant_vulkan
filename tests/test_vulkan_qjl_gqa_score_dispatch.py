import types

import torch

import turboquant.vulkan_backend as vk
from turboquant.vulkan.reference_ops import qjl_gqa_score_reference


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


def test_qjl_gqa_score_uses_reference_path_when_extension_symbol_missing(monkeypatch):
    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(vk, "_vulkan_ext", types.SimpleNamespace())
    monkeypatch.setattr(vk, "get_vulkan_capability_report", _mock_ready_report)

    b, kh, qh, n, g, d = 1, 2, 4, 2, 2, 8
    s, so, o = 8, 8, 2
    key_quant = torch.randint(0, 255, (b, kh, n, g, s // 8), dtype=torch.uint8)
    key_outlier_quant = torch.randint(0, 255, (b, kh, n, g, so // 8), dtype=torch.uint8)
    key_norm = torch.rand(b, kh, n, g, dtype=torch.float16) + 1.0
    key_outlier_norm = key_norm * 0.25
    outlier_indices = torch.randint(0, d, (b, kh, n, o), dtype=torch.int64)
    query_sketch = torch.randn(b, qh, s, dtype=torch.float32)
    query_states = torch.randn(b, qh, d, dtype=torch.float16)
    rand_prj = torch.randn(d, s, dtype=torch.float16)

    got = vk.qjl_gqa_score(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )
    exp = qjl_gqa_score_reference(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )
    assert torch.allclose(got, exp, atol=2e-3, rtol=2e-3)


def test_qjl_gqa_score_prefers_extension_symbol_when_available(monkeypatch):
    sentinel = torch.zeros(1, 1, 1, 1, dtype=torch.float32)

    class _Ext:
        def qjl_gqa_score_vulkan_half_half(self, *_args, **_kwargs):
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

    got = vk.qjl_gqa_score(
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
