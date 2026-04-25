import torch

from turboquant.cuda_backend import QJLSketch
from turboquant.vulkan.reference_ops import qjl_quant_reference


def _indices_to_mask(outlier_indices: torch.Tensor, emb_dim: int, dtype: torch.dtype) -> torch.Tensor:
    b, h, n, o = outlier_indices.shape
    mask = torch.zeros((b, h, n, emb_dim), dtype=dtype)
    mask.scatter_(-1, outlier_indices.long(), 1.0)
    return mask


@torch.no_grad()
def test_qjl_quant_reference_matches_pytorch_float32():
    torch.manual_seed(42)
    b, h, n, g, d = 2, 2, 3, 4, 128
    sketch_dim = 64
    outlier_sketch_dim = 32
    outlier_count = 8

    key_states = torch.randn(b, h, n, g, d, dtype=torch.float32)
    outlier_indices = torch.randint(0, d, (b, h, n, outlier_count), dtype=torch.int64)
    rand_prj = torch.randn(sketch_dim, d, dtype=torch.float32)

    key_quant, key_outlier_quant, outlier_norms = qjl_quant_reference(
        key_states, outlier_indices, rand_prj, outlier_sketch_dim
    )

    sketch = QJLSketch(dim=(d, sketch_dim), dim_outlier=outlier_sketch_dim, device=torch.device("cpu"))
    sketch.proj_dir_quant = rand_prj.contiguous()
    mask = _indices_to_mask(outlier_indices, d, key_states.dtype)
    exp_key_quant, exp_key_outlier_quant = sketch.quantize_pytorch(key_states, mask)

    assert torch.equal(key_quant, exp_key_quant)
    assert torch.equal(key_outlier_quant, exp_key_outlier_quant)

    expected_norms = torch.sqrt(((key_states * mask.unsqueeze(-2)).float() ** 2).sum(dim=-1)).to(key_states.dtype)
    assert torch.allclose(outlier_norms, expected_norms, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_qjl_quant_reference_matches_pytorch_float16():
    torch.manual_seed(7)
    b, h, n, g, d = 1, 2, 2, 3, 128
    sketch_dim = 64
    outlier_sketch_dim = 32
    outlier_count = 4

    key_states = torch.randn(b, h, n, g, d, dtype=torch.float16)
    outlier_indices = torch.randint(0, d, (b, h, n, outlier_count), dtype=torch.int64)
    rand_prj = torch.randn(sketch_dim, d, dtype=torch.float32)

    key_quant, key_outlier_quant, outlier_norms = qjl_quant_reference(
        key_states, outlier_indices, rand_prj, outlier_sketch_dim
    )

    sketch = QJLSketch(dim=(d, sketch_dim), dim_outlier=outlier_sketch_dim, device=torch.device("cpu"))
    sketch.proj_dir_quant = rand_prj.contiguous()
    mask = _indices_to_mask(outlier_indices, d, key_states.dtype)
    exp_key_quant, exp_key_outlier_quant = sketch.quantize_pytorch(key_states, mask)

    assert torch.equal(key_quant, exp_key_quant)
    assert torch.equal(key_outlier_quant, exp_key_outlier_quant)

    expected_norms = torch.sqrt(((key_states * mask.unsqueeze(-2)).float() ** 2).sum(dim=-1)).to(key_states.dtype)
    # fp16 accumulation can differ by 1 ulp in this path.
    assert torch.allclose(outlier_norms, expected_norms, atol=1e-3, rtol=1e-3)

