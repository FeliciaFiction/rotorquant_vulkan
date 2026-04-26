import pytest
import torch

import turboquant.vulkan_backend as vk
from turboquant.vulkan.reference_ops import quantized_bmm_reference


def _build_case(dtype=torch.float16, mqa=False):
    torch.manual_seed(20260426 + (1 if mqa else 0))
    bits = 4
    group_size = 8
    b, h, m, k = 2, 2, 3, 16
    n = 32
    pack_factor = 32 // bits
    n_packed = n // pack_factor
    n_groups = n // group_size

    fA = torch.randn(b, h, m, k, dtype=dtype)
    qB = torch.randint(-(2**30), 2**30, (b, 1 if mqa else h, k, n_packed), dtype=torch.int32)
    scales = torch.randn(b, 1 if mqa else h, k, n_groups, dtype=dtype) * 0.05 + 0.1
    zeros = torch.randn(b, 1 if mqa else h, k, n_groups, dtype=dtype) * 0.01
    return group_size, fA, qB, scales, zeros, bits


@pytest.mark.skipif(not vk._VULKAN_EXT_AVAILABLE, reason="vulkan extension unavailable")
def test_quantized_bmm_native_extension_matches_reference_half():
    report = vk.get_vulkan_capability_report()
    if not report.get("strictly_available", False):
        pytest.skip("strict Vulkan capability unavailable on this host")

    group_size, fA, qB, scales, zeros, bits = _build_case(dtype=torch.float16, mqa=False)
    got = vk._vulkan_ext.quantized_bmm_vulkan_half(group_size, fA, qB, scales, zeros, bits, False)
    exp = quantized_bmm_reference(group_size, fA, qB, scales, zeros, bits, False)
    assert got.shape == exp.shape
    assert torch.allclose(got, exp, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not vk._VULKAN_EXT_AVAILABLE, reason="vulkan extension unavailable")
def test_quantized_bmm_native_extension_matches_reference_half_mqa():
    report = vk.get_vulkan_capability_report()
    if not report.get("strictly_available", False):
        pytest.skip("strict Vulkan capability unavailable on this host")

    group_size, fA, qB, scales, zeros, bits = _build_case(dtype=torch.float16, mqa=True)
    got = vk._vulkan_ext.quantized_bmm_vulkan_half(group_size, fA, qB, scales, zeros, bits, True)
    exp = quantized_bmm_reference(group_size, fA, qB, scales, zeros, bits, True)
    assert got.shape == exp.shape
    assert torch.allclose(got, exp, atol=2e-2, rtol=2e-2)
