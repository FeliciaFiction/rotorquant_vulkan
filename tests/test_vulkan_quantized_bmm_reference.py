import torch
import pytest

from turboquant.vulkan.reference_ops import quantized_bmm_reference


def _pack_weight_matrix(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack uint quantized weights from [N, K] into [N_packed, K] int32 words.
    """
    if bits not in (2, 4):
        raise ValueError("bits must be one of {2, 4}")
    n, k = weight.shape
    pack_factor = 32 // bits
    if n % pack_factor != 0:
        raise ValueError("N must be divisible by pack_factor")

    n_packed = n // pack_factor
    out = torch.zeros((n_packed, k), dtype=torch.int32)
    for oc in range(n):
        packed_idx = oc // pack_factor
        shift = (oc % pack_factor) * bits
        out[packed_idx] |= (weight[oc].to(torch.int32) << shift)
    return out.contiguous()


def _naive_quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False):
    if bits not in (2, 4):
        raise ValueError("bits must be one of {2, 4}")
    b, h, m, k = fA.shape
    feat_per_int = 32 // bits
    n = qB.shape[-1] * feat_per_int
    mask = (1 << bits) - 1

    fA2 = fA.view(-1, m, k).contiguous()
    qB2 = qB.reshape(-1, k, qB.shape[-1]).transpose(1, 2).contiguous()
    flatten_b = b * h if not mqa else b
    scales2 = scales.view(flatten_b, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
    zeros2 = zeros.view(flatten_b, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()

    out = torch.zeros((b * h, m, n), dtype=torch.float32)
    for bh in range(b * h):
        wb = bh if not mqa else (bh // h)
        for mi in range(m):
            for oc in range(n):
                packed_idx = oc // feat_per_int
                shift = (oc % feat_per_int) * bits
                group_idx = oc // group_size
                acc = 0.0
                for ic in range(k):
                    word = int(qB2[wb, packed_idx, ic].item())
                    qv = float((word >> shift) & mask)
                    s = float(scales2[wb, group_idx, ic].item())
                    z = float(zeros2[wb, group_idx, ic].item())
                    acc += float(fA2[bh, mi, ic].item()) * (s * qv + z)
                out[bh, mi, oc] = acc
    return out.view(b, h, m, n).to(fA.dtype).contiguous()


def _build_inputs(bits: int, group_size: int, mqa: bool, dtype: torch.dtype):
    torch.manual_seed(2026 + bits + group_size + (10 if mqa else 0))
    b, h, m, k = 2, 2, 3, 16
    n = 32
    pack_factor = 32 // bits
    n_packed = n // pack_factor
    n_groups = n // group_size

    fA = torch.randn(b, h, m, k, dtype=dtype)
    weight_batches = b if mqa else b * h

    q_vals = torch.randint(0, 2 ** bits, (weight_batches, n, k), dtype=torch.int32)
    qB2 = torch.stack([_pack_weight_matrix(q_vals[i], bits) for i in range(weight_batches)], dim=0)
    qB_input = qB2.transpose(1, 2).contiguous().view(b, 1 if mqa else h, k, n_packed)

    scales2 = torch.randn(weight_batches, n_groups, k, dtype=torch.float32) * 0.2 + 0.05
    zeros2 = torch.randn(weight_batches, n_groups, k, dtype=torch.float32) * 0.1
    scales_input = scales2.transpose(1, 2).contiguous().view(b, 1 if mqa else h, k, n_groups).to(dtype)
    zeros_input = zeros2.transpose(1, 2).contiguous().view(b, 1 if mqa else h, k, n_groups).to(dtype)
    return fA, qB_input, scales_input, zeros_input


@torch.no_grad()
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("group_size", [8, 16])
def test_quantized_bmm_reference_matches_naive(bits, group_size):
    fA, qB, scales, zeros = _build_inputs(bits=bits, group_size=group_size, mqa=False, dtype=torch.float32)
    got = quantized_bmm_reference(group_size, fA, qB, scales, zeros, bits, mqa=False)
    exp = _naive_quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False)
    assert got.shape == exp.shape
    assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("bits", [2, 4])
def test_quantized_bmm_reference_mqa_layout(bits):
    group_size = 8
    fA, qB, scales, zeros = _build_inputs(bits=bits, group_size=group_size, mqa=True, dtype=torch.float16)
    got = quantized_bmm_reference(group_size, fA, qB, scales, zeros, bits, mqa=True)
    exp = _naive_quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=True)
    assert got.shape == exp.shape
    assert torch.allclose(got, exp, atol=2e-2, rtol=2e-2)
