import math

import torch

from turboquant.vulkan.reference_ops import qjl_gqa_score_reference


def _pack_signs(signs: torch.Tensor) -> torch.Tensor:
    s = signs.shape[-1]
    if s % 8 != 0:
        raise ValueError("S must be divisible by 8")
    bits = 8
    enc = (2 ** torch.arange(bits, dtype=torch.uint8)).view(1, 1, 1, 1, 1, bits)
    b = (signs > 0).to(torch.uint8).view(*signs.shape[:-1], s // bits, bits)
    return (b * enc).sum(dim=-1, dtype=torch.uint8).contiguous()


def _naive_gqa_score(
    key_quant: torch.Tensor,
    key_outlier_quant: torch.Tensor,
    key_norm: torch.Tensor,
    key_outlier_norm: torch.Tensor,
    outlier_indices: torch.Tensor,
    query_sketch: torch.Tensor,
    query_states: torch.Tensor,
    rand_prj: torch.Tensor,
) -> torch.Tensor:
    b, kh, n, g, hash_dim = key_quant.shape
    qh = query_states.shape[1]
    s = hash_dim * 8
    so = key_outlier_quant.shape[-1] * 8
    o = outlier_indices.shape[-1]
    gqa_group_size = qh // kh

    out = torch.zeros((b, qh, n, g), dtype=torch.float32)
    scl = math.sqrt(math.pi / 2.0) / s
    scl_o = math.sqrt(math.pi / 2.0) / so

    for bi in range(b):
        for qhi in range(qh):
            khi = qhi // gqa_group_size
            for ni in range(n):
                q_out = torch.zeros(s, dtype=torch.float32)
                for oi in range(o):
                    idx = int(outlier_indices[bi, khi, ni, oi])
                    q_out += query_states[bi, qhi, idx].float() * rand_prj[idx].float()

                for gi in range(g):
                    k_inner = 0.0
                    o_inner = 0.0
                    for p in range(s):
                        byte_i = p // 8
                        bit_i = p % 8
                        k_byte = int(key_quant[bi, khi, ni, gi, byte_i].item())
                        k_sign = 1.0 if ((k_byte >> bit_i) & 1) else -1.0
                        k_inner += k_sign * float(query_sketch[bi, qhi, p] - q_out[p])

                        if p < so:
                            o_byte = int(key_outlier_quant[bi, khi, ni, gi, byte_i].item())
                            o_sign = 1.0 if ((o_byte >> bit_i) & 1) else -1.0
                            o_inner += o_sign * float(q_out[p])

                    norm_o = float(key_outlier_norm[bi, khi, ni, gi])
                    norm_k = math.sqrt(max(float(key_norm[bi, khi, ni, gi]) ** 2 - norm_o ** 2, 0.0))
                    out[bi, qhi, ni, gi] = scl * norm_k * k_inner + scl_o * norm_o * o_inner
    return out.reshape(b, qh, n * g, 1).contiguous()


@torch.no_grad()
def test_qjl_gqa_score_reference_matches_naive_float32():
    torch.manual_seed(777)
    b, kh, gqa_group_size, n, g, d = 1, 2, 3, 4, 3, 128
    qh = kh * gqa_group_size
    s = 64
    so = 32
    o = 6

    key_signs = torch.where(torch.randn(b, kh, n, g, s) > 0, 1.0, -1.0)
    key_quant = _pack_signs(key_signs)
    out_signs = torch.where(torch.randn(b, kh, n, g, so) > 0, 1.0, -1.0)
    key_outlier_quant = _pack_signs(out_signs)

    key_norm = torch.rand(b, kh, n, g, dtype=torch.float32) * 2.0 + 0.5
    key_outlier_norm = key_norm * 0.5
    outlier_indices = torch.randint(0, d, (b, kh, n, o), dtype=torch.int64)
    query_sketch = torch.randn(b, qh, s, dtype=torch.float32)
    query_states = torch.randn(b, qh, d, dtype=torch.float32)
    rand_prj = torch.randn(d, s, dtype=torch.float32)

    got = qjl_gqa_score_reference(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )
    exp = _naive_gqa_score(
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
def test_qjl_gqa_score_reference_matches_naive_float16():
    torch.manual_seed(888)
    b, kh, gqa_group_size, n, g, d = 2, 1, 2, 3, 2, 128
    qh = kh * gqa_group_size
    s = 64
    so = 32
    o = 4

    key_signs = torch.where(torch.randn(b, kh, n, g, s) > 0, 1.0, -1.0)
    key_quant = _pack_signs(key_signs)
    out_signs = torch.where(torch.randn(b, kh, n, g, so) > 0, 1.0, -1.0)
    key_outlier_quant = _pack_signs(out_signs)

    key_norm = (torch.rand(b, kh, n, g, dtype=torch.float16) * 2.0 + 0.5).contiguous()
    key_outlier_norm = (key_norm * 0.5).contiguous()
    outlier_indices = torch.randint(0, d, (b, kh, n, o), dtype=torch.int64)
    query_sketch = torch.randn(b, qh, s, dtype=torch.float32)
    query_states = torch.randn(b, qh, d, dtype=torch.float16)
    rand_prj = torch.randn(d, s, dtype=torch.float32)

    got = qjl_gqa_score_reference(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )
    exp = _naive_gqa_score(
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
