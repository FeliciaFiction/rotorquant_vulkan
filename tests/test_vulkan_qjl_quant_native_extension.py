import pytest
import torch

import turboquant.vulkan_backend as vk
from turboquant.vulkan.reference_ops import qjl_quant_reference


@pytest.mark.skipif(not vk._VULKAN_EXT_AVAILABLE, reason="vulkan extension unavailable")
def test_qjl_quant_native_extension_matches_reference_half_half():
    report = vk.get_vulkan_capability_report()
    if not report.get("strictly_available", False):
        pytest.skip("strict Vulkan capability unavailable on this host")

    key_states = torch.randn(1, 2, 3, 2, 16, dtype=torch.float16)
    outlier_indices = torch.tensor(
        [[[[0, 3], [1, 4], [2, 5]], [[6, 7], [8, 9], [10, 11]]]],
        dtype=torch.int64,
    )
    rand_prj = torch.randn(16, 16, dtype=torch.float16)

    got = vk._vulkan_ext.qjl_quant_half_half(
        key_states,
        outlier_indices,
        rand_prj,
        8,
    )
    exp = qjl_quant_reference(
        key_states,
        outlier_indices,
        rand_prj,
        8,
    )

    assert torch.equal(got[0], exp[0])
    assert torch.equal(got[1], exp[1])
    assert torch.allclose(got[2], exp[2], atol=1e-3, rtol=1e-3)

