import types

import torch

import turboquant.vulkan_backend as vk
from turboquant.vulkan.reference_ops import quantized_bmm_reference


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


def _build_case(dtype=torch.float16):
    torch.manual_seed(12345)
    bits = 4
    group_size = 8
    b, h, m, k = 1, 2, 2, 8
    n = 32
    pack_factor = 32 // bits
    n_packed = n // pack_factor
    n_groups = n // group_size

    fA = torch.randn(b, h, m, k, dtype=dtype)
    qB = torch.randint(0, 2**bits, (b, h, k, n_packed), dtype=torch.int32)
    scales = torch.randn(b, h, k, n_groups, dtype=dtype) * 0.05 + 0.1
    zeros = torch.randn(b, h, k, n_groups, dtype=dtype) * 0.01
    return group_size, fA, qB, scales, zeros, bits


def test_quantized_bmm_uses_reference_path_when_extension_symbol_missing(monkeypatch):
    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(vk, "_vulkan_ext", types.SimpleNamespace())
    monkeypatch.setattr(vk, "get_vulkan_capability_report", _mock_ready_report)

    group_size, fA, qB, scales, zeros, bits = _build_case(dtype=torch.float16)
    got = vk.quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False)
    exp = quantized_bmm_reference(group_size, fA, qB, scales, zeros, bits, mqa=False)
    assert got.shape == exp.shape
    assert torch.allclose(got, exp, atol=2e-2, rtol=2e-2)


def test_quantized_bmm_prefers_extension_symbol_when_available(monkeypatch):
    sentinel = torch.zeros(1, 1, 1, 1, dtype=torch.float16)

    class _Ext:
        def quantized_bmm_vulkan_half(self, *_args, **_kwargs):
            return sentinel

    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(vk, "_vulkan_ext", _Ext())
    monkeypatch.setattr(vk, "get_vulkan_capability_report", _mock_ready_report)

    group_size, fA, qB, scales, zeros, bits = _build_case(dtype=torch.float16)
    got = vk.quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False)
    assert got is sentinel
