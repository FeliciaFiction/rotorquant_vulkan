import pytest
import torch

import turboquant.vulkan_backend as vk
from turboquant.vulkan.reference_ops import qjl_gqa_score_reference


@pytest.mark.skipif(not vk._VULKAN_EXT_AVAILABLE, reason="vulkan extension unavailable")
def test_qjl_gqa_score_native_extension_matches_reference_half_half():
    report = vk.get_vulkan_capability_report()
    if not report.get("strictly_available", False):
        pytest.skip("strict Vulkan capability unavailable on this host")

    b, kh, qh, n, g, d = 1, 2, 4, 3, 2, 16
    s, so, o = 16, 8, 2
    key_quant = torch.randint(0, 255, (b, kh, n, g, s // 8), dtype=torch.uint8)
    key_outlier_quant = torch.randint(0, 255, (b, kh, n, g, so // 8), dtype=torch.uint8)
    key_norm = torch.rand(b, kh, n, g, dtype=torch.float16) + 1.0
    key_outlier_norm = key_norm * 0.25
    outlier_indices = torch.randint(0, d, (b, kh, n, o), dtype=torch.int64)
    query_sketch = torch.randn(b, qh, s, dtype=torch.float32)
    query_states = torch.randn(b, qh, d, dtype=torch.float16)
    rand_prj = torch.randn(d, s, dtype=torch.float16)

    got = vk._vulkan_ext.qjl_gqa_score_vulkan_half_half(
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
