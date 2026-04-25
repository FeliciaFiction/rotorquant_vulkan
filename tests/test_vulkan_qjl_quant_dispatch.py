import types

import torch

import turboquant.vulkan_backend as vk
from turboquant.vulkan.reference_ops import qjl_quant_reference


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


def test_qjl_quant_uses_reference_path_when_extension_symbol_missing(monkeypatch):
    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(vk, "_vulkan_ext", types.SimpleNamespace())
    monkeypatch.setattr(vk, "get_vulkan_capability_report", _mock_ready_report)

    key_states = torch.randn(1, 1, 2, 2, 8, dtype=torch.float16)
    outlier_indices = torch.tensor([[[[0, 3], [1, 4]]]], dtype=torch.int64)
    rand_prj = torch.randn(8, 8, dtype=torch.float16)

    got = vk.qjl_quant(key_states, outlier_indices, rand_prj, 8)
    exp = qjl_quant_reference(key_states, outlier_indices, rand_prj, 8)

    assert torch.equal(got[0], exp[0])
    assert torch.equal(got[1], exp[1])
    assert torch.allclose(got[2], exp[2], atol=1e-3, rtol=1e-3)


def test_qjl_quant_prefers_extension_symbol_when_available(monkeypatch):
    sentinel = (
        torch.zeros(1, 1, 1, 1, 1, dtype=torch.uint8),
        torch.zeros(1, 1, 1, 1, 1, dtype=torch.uint8),
        torch.zeros(1, 1, 1, 1, dtype=torch.float16),
    )

    class _Ext:
        def qjl_quant_half_half(self, *_args, **_kwargs):
            return sentinel

    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(vk, "_vulkan_ext", _Ext())
    monkeypatch.setattr(vk, "get_vulkan_capability_report", _mock_ready_report)

    key_states = torch.zeros(1, 1, 1, 1, 8, dtype=torch.float16)
    outlier_indices = torch.zeros(1, 1, 1, 1, dtype=torch.int64)
    rand_prj = torch.zeros(8, 8, dtype=torch.float16)

    got = vk.qjl_quant(key_states, outlier_indices, rand_prj, 8)
    assert got is sentinel
