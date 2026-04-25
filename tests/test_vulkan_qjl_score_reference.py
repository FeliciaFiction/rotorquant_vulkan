import math

import torch

from turboquant.vulkan.reference_ops import qjl_score_reference


def _pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """
    signs: [..., S] values in {-1,+1}
    returns: [..., S/8] uint8
    """
    s = signs.shape[-1]
    if s % 8 != 0:
        raise ValueError("S must be divisible by 8")
    bits = 8
    enc = (2 ** torch.arange(bits, dtype=torch.uint8)).view(1, 1, 1, 1, 1, bits)
    b = (signs > 0).to(torch.uint8).view(*signs.shape[:-1], s // bits, bits)
    return (b * enc).sum(dim=-1, dtype=torch.uint8).contiguous()


def _naive_score(
    key_quant: torch.Tensor,
    key_outlier_quant: torch.Tensor,
    key_norm: torch.Tensor,
    key_outlier_norm: torch.Tensor,
    outlier_indices: torch.Tensor,
    query_sketch: torch.Tensor,
    query_states: torch.Tensor,
    rand_prj: torch.Tensor,
) -> torch.Tensor:
    b, h, n, g, hash_dim = key_quant.shape
    s = hash_dim * 8
    so = key_outlier_quant.shape[-1] * 8
    o = outlier_indices.shape[-1]

    out = torch.zeros((b, h, n, g), dtype=torch.float32)
    scl = math.sqrt(math.pi / 2.0) / s
    scl_o = math.sqrt(math.pi / 2.0) / so

    for bi in range(b):
        for hi in range(h):
            for ni in range(n):
                # q_outlier_sketch[p] = sum_i q[out_idx_i] * rand_prj[out_idx_i, p]
                q_out = torch.zeros(s, dtype=torch.float32)
                for oi in range(o):
                    idx = int(outlier_indices[bi, hi, ni, oi])
                    q_out += query_states[bi, hi, idx].float() * rand_prj[idx].float()

                for gi in range(g):
                    k_inner = 0.0
                    o_inner = 0.0
                    for p in range(s):
                        byte_i = p // 8
                        bit_i = p % 8
                        k_byte = int(key_quant[bi, hi, ni, gi, byte_i].item())
                        k_sign = 1.0 if ((k_byte >> bit_i) & 1) else -1.0
                        k_inner += k_sign * float(query_sketch[bi, hi, p] - q_out[p])

                        if p < so:
                            o_byte = int(key_outlier_quant[bi, hi, ni, gi, byte_i].item())
                            o_sign = 1.0 if ((o_byte >> bit_i) & 1) else -1.0
                            o_inner += o_sign * float(q_out[p])

                    norm_o = float(key_outlier_norm[bi, hi, ni, gi])
                    norm_k = math.sqrt(max(float(key_norm[bi, hi, ni, gi]) ** 2 - norm_o ** 2, 0.0))
                    out[bi, hi, ni, gi] = scl * norm_k * k_inner + scl_o * norm_o * o_inner
    return out.reshape(b, h, n * g, 1).contiguous()


@torch.no_grad()
def test_qjl_score_reference_matches_naive_float32():
    torch.manual_seed(123)
    b, h, n, g, d = 2, 2, 4, 3, 128
    s = 64
    so = 32
    o = 8

    key_signs = torch.where(torch.randn(b, h, n, g, s) > 0, 1.0, -1.0)
    key_quant = _pack_signs(key_signs)
    out_signs = torch.where(torch.randn(b, h, n, g, so) > 0, 1.0, -1.0)
    key_outlier_quant = _pack_signs(out_signs)

    key_norm = torch.rand(b, h, n, g, dtype=torch.float32) * 2.0 + 0.5
    key_outlier_norm = key_norm * 0.5
    outlier_indices = torch.randint(0, d, (b, h, n, o), dtype=torch.int64)
    query_sketch = torch.randn(b, h, s, dtype=torch.float32)
    query_states = torch.randn(b, h, d, dtype=torch.float32)
    rand_prj = torch.randn(d, s, dtype=torch.float32)

    got = qjl_score_reference(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )
    exp = _naive_score(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )
    assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_qjl_score_reference_matches_naive_float16():
    torch.manual_seed(321)
    b, h, n, g, d = 1, 2, 3, 2, 128
    s = 64
    so = 32
    o = 4

    key_signs = torch.where(torch.randn(b, h, n, g, s) > 0, 1.0, -1.0)
    key_quant = _pack_signs(key_signs)
    out_signs = torch.where(torch.randn(b, h, n, g, so) > 0, 1.0, -1.0)
    key_outlier_quant = _pack_signs(out_signs)

    key_norm = (torch.rand(b, h, n, g, dtype=torch.float16) * 2.0 + 0.5).contiguous()
    key_outlier_norm = (key_norm * 0.5).contiguous()
    outlier_indices = torch.randint(0, d, (b, h, n, o), dtype=torch.int64)
    query_sketch = torch.randn(b, h, s, dtype=torch.float32)
    query_states = torch.randn(b, h, d, dtype=torch.float16)
    rand_prj = torch.randn(d, s, dtype=torch.float32)

    got = qjl_score_reference(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )
    exp = _naive_score(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )
    # fp16 path for query/key norms can create minor deviations.
    assert torch.allclose(got, exp, atol=2e-3, rtol=2e-3)

