import pytest
import torch

import turboquant.vulkan_backend as vk
from turboquant.vulkan.reference_ops import qjl_score_reference


@pytest.mark.skipif(not vk._VULKAN_EXT_AVAILABLE, reason="vulkan extension unavailable")
def test_qjl_score_native_extension_matches_reference_half_half():
    report = vk.get_vulkan_capability_report()
    if not report.get("strictly_available", False):
        pytest.skip("strict Vulkan capability unavailable on this host")

    key_quant = torch.randint(0, 255, (1, 2, 3, 2, 2), dtype=torch.uint8)
    key_outlier_quant = torch.randint(0, 255, (1, 2, 3, 2, 1), dtype=torch.uint8)
    key_norm = torch.rand(1, 2, 3, 2, dtype=torch.float16) + 1.0
    key_outlier_norm = key_norm * 0.25
    outlier_indices = torch.tensor(
        [[[[0, 3], [1, 4], [2, 5]], [[6, 7], [8, 9], [10, 11]]]],
        dtype=torch.int64,
    )
    query_sketch = torch.randn(1, 2, 16, dtype=torch.float32)
    query_states = torch.randn(1, 2, 16, dtype=torch.float16)
    rand_prj = torch.randn(16, 16, dtype=torch.float16)

    got = vk._vulkan_ext.qjl_score_vulkan_half_half(
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
