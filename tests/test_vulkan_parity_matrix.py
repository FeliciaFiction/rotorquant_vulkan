import math

import pytest
import torch

from turboquant import cuda_backend
from turboquant.cuda_backend import QJLSketch
from turboquant.vulkan.reference_ops import (
    qjl_gqa_score_reference,
    qjl_quant_reference,
    qjl_score_reference,
    quantized_bmm_reference,
)


TOLERANCE_MATRIX = {
    "qjl_quant": {
        "pytorch": {
            torch.float32: {"atol": 1e-6, "rtol": 1e-6},
            torch.float16: {"atol": 1e-3, "rtol": 1e-3},
        },
        "cuda": {
            torch.float32: {"atol": 1e-5, "rtol": 1e-5},
            torch.float16: {"atol": 2e-3, "rtol": 2e-3},
        },
    },
    "qjl_score": {
        "pytorch": {
            torch.float32: {"atol": 1e-5, "rtol": 1e-5},
            torch.float16: {"atol": 2e-3, "rtol": 2e-3},
        },
        "cuda": {
            torch.float32: {"atol": 3e-4, "rtol": 3e-4},
            torch.float16: {"atol": 5e-2, "rtol": 5e-2},
        },
    },
    "qjl_gqa_score": {
        "pytorch": {
            torch.float32: {"atol": 1e-5, "rtol": 1e-5},
            torch.float16: {"atol": 2e-3, "rtol": 2e-3},
        },
        "cuda": {
            torch.float32: {"atol": 3e-4, "rtol": 3e-4},
            torch.float16: {"atol": 5e-2, "rtol": 5e-2},
        },
    },
    "quantized_bmm": {
        "pytorch": {
            torch.float32: {"atol": 1e-5, "rtol": 1e-5},
            torch.float16: {"atol": 2e-2, "rtol": 2e-2},
        },
        "cuda": {
            torch.float32: {"atol": 3e-3, "rtol": 3e-3},
            torch.float16: {"atol": 8e-2, "rtol": 8e-2},
        },
    },
}


def _indices_to_mask(outlier_indices: torch.Tensor, emb_dim: int, dtype: torch.dtype) -> torch.Tensor:
    b, h, n, _ = outlier_indices.shape
    mask = torch.zeros((b, h, n, emb_dim), dtype=dtype, device=outlier_indices.device)
    mask.scatter_(-1, outlier_indices.long(), 1.0)
    return mask


def _pack_signs(signs: torch.Tensor) -> torch.Tensor:
    s = signs.shape[-1]
    if s % 8 != 0:
        raise ValueError("S must be divisible by 8")
    bits = 8
    enc = (2 ** torch.arange(bits, dtype=torch.uint8, device=signs.device)).view(1, 1, 1, 1, 1, bits)
    b = (signs > 0).to(torch.uint8).view(*signs.shape[:-1], s // bits, bits)
    return (b * enc).sum(dim=-1, dtype=torch.uint8).contiguous()


def _naive_qjl_score(
    key_quant,
    key_outlier_quant,
    key_norm,
    key_outlier_norm,
    outlier_indices,
    query_sketch,
    query_states,
    rand_prj,
):
    b, h, n, g, hash_dim = key_quant.shape
    s = hash_dim * 8
    so = key_outlier_quant.shape[-1] * 8
    o = outlier_indices.shape[-1]
    out = torch.zeros((b, h, n, g), dtype=torch.float32, device=query_states.device)
    scl = math.sqrt(math.pi / 2.0) / s
    scl_o = math.sqrt(math.pi / 2.0) / so
    for bi in range(b):
        for hi in range(h):
            for ni in range(n):
                q_out = torch.zeros(s, dtype=torch.float32, device=query_states.device)
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


def _naive_qjl_gqa_score(
    key_quant,
    key_outlier_quant,
    key_norm,
    key_outlier_norm,
    outlier_indices,
    query_sketch,
    query_states,
    rand_prj,
):
    b, kh, n, g, hash_dim = key_quant.shape
    qh = query_states.shape[1]
    s = hash_dim * 8
    so = key_outlier_quant.shape[-1] * 8
    o = outlier_indices.shape[-1]
    gqa_group_size = qh // kh
    out = torch.zeros((b, qh, n, g), dtype=torch.float32, device=query_states.device)
    scl = math.sqrt(math.pi / 2.0) / s
    scl_o = math.sqrt(math.pi / 2.0) / so
    for bi in range(b):
        for qhi in range(qh):
            khi = qhi // gqa_group_size
            for ni in range(n):
                q_out = torch.zeros(s, dtype=torch.float32, device=query_states.device)
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


def _pack_weight_matrix(weight: torch.Tensor, bits: int) -> torch.Tensor:
    n, k = weight.shape
    pack_factor = 32 // bits
    if n % pack_factor != 0:
        raise ValueError("N must be divisible by pack_factor")
    n_packed = n // pack_factor
    out = torch.zeros((n_packed, k), dtype=torch.int32, device=weight.device)
    for oc in range(n):
        packed_idx = oc // pack_factor
        shift = (oc % pack_factor) * bits
        out[packed_idx] |= (weight[oc].to(torch.int32) << shift)
    return out.contiguous()


def _naive_quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False):
    b, h, m, k = fA.shape
    feat_per_int = 32 // bits
    n = qB.shape[-1] * feat_per_int
    mask = (1 << bits) - 1
    fA2 = fA.view(-1, m, k).contiguous()
    qB2 = qB.reshape(-1, k, qB.shape[-1]).transpose(1, 2).contiguous()
    flatten_b = b * h if not mqa else b
    scales2 = scales.view(flatten_b, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
    zeros2 = zeros.view(flatten_b, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
    out = torch.zeros((b * h, m, n), dtype=torch.float32, device=fA.device)
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


def test_tolerance_matrix_has_all_ops():
    for op in ("qjl_quant", "qjl_score", "qjl_gqa_score", "quantized_bmm"):
        assert op in TOLERANCE_MATRIX
        assert "pytorch" in TOLERANCE_MATRIX[op]
        assert torch.float32 in TOLERANCE_MATRIX[op]["pytorch"]
        assert torch.float16 in TOLERANCE_MATRIX[op]["pytorch"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.no_grad()
def test_parity_qjl_quant_vulkan_vs_pytorch(dtype):
    torch.manual_seed(2027)
    b, h, n, g, d = 2, 2, 3, 4, 128
    sketch_dim, outlier_sketch_dim, outlier_count = 64, 32, 8
    key_states = torch.randn(b, h, n, g, d, dtype=dtype)
    outlier_indices = torch.randint(0, d, (b, h, n, outlier_count), dtype=torch.int64)
    rand_prj = torch.randn(sketch_dim, d, dtype=torch.float32)

    vk_kq, vk_koq, vk_on = qjl_quant_reference(key_states, outlier_indices, rand_prj, outlier_sketch_dim)

    sketch = QJLSketch(dim=(d, sketch_dim), dim_outlier=outlier_sketch_dim, device=torch.device("cpu"))
    sketch.proj_dir_quant = rand_prj.contiguous()
    mask = _indices_to_mask(outlier_indices, d, key_states.dtype)
    pt_kq, pt_koq = sketch.quantize_pytorch(key_states, mask)
    pt_on = torch.sqrt(((key_states * mask.unsqueeze(-2)).float() ** 2).sum(dim=-1)).to(dtype)

    tol = TOLERANCE_MATRIX["qjl_quant"]["pytorch"][dtype]
    assert torch.equal(vk_kq, pt_kq)
    assert torch.equal(vk_koq, pt_koq)
    assert torch.allclose(vk_on, pt_on, **tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.no_grad()
def test_parity_qjl_score_vulkan_vs_pytorch(dtype):
    torch.manual_seed(2028)
    b, h, n, g, d = 1, 2, 3, 2, 128
    s, so, o = 64, 32, 6

    key_signs = torch.where(torch.randn(b, h, n, g, s) > 0, 1.0, -1.0)
    out_signs = torch.where(torch.randn(b, h, n, g, so) > 0, 1.0, -1.0)
    key_quant = _pack_signs(key_signs)
    key_outlier_quant = _pack_signs(out_signs)
    key_norm = (torch.rand(b, h, n, g, dtype=torch.float32) * 2.0 + 0.5).to(dtype)
    key_outlier_norm = (key_norm * 0.5).contiguous()
    outlier_indices = torch.randint(0, d, (b, h, n, o), dtype=torch.int64)
    query_states = torch.randn(b, h, d, dtype=dtype)
    rand_prj = torch.randn(d, s, dtype=torch.float32)
    query_sketch = torch.matmul(query_states.to(rand_prj.dtype), rand_prj).to(torch.float32)

    got = qjl_score_reference(
        key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj
    )
    exp = _naive_qjl_score(
        key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj
    )
    tol = TOLERANCE_MATRIX["qjl_score"]["pytorch"][dtype]
    assert torch.allclose(got, exp, **tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.no_grad()
def test_parity_qjl_gqa_score_vulkan_vs_pytorch(dtype):
    torch.manual_seed(2029)
    b, kh, qh, n, g, d = 1, 2, 6, 3, 2, 128
    s, so, o = 64, 32, 5
    key_signs = torch.where(torch.randn(b, kh, n, g, s) > 0, 1.0, -1.0)
    out_signs = torch.where(torch.randn(b, kh, n, g, so) > 0, 1.0, -1.0)
    key_quant = _pack_signs(key_signs)
    key_outlier_quant = _pack_signs(out_signs)
    key_norm = (torch.rand(b, kh, n, g, dtype=torch.float32) * 2.0 + 0.5).to(dtype)
    key_outlier_norm = (key_norm * 0.5).contiguous()
    outlier_indices = torch.randint(0, d, (b, kh, n, o), dtype=torch.int64)
    query_states = torch.randn(b, qh, d, dtype=dtype)
    rand_prj = torch.randn(d, s, dtype=torch.float32)
    query_sketch = torch.matmul(query_states.to(rand_prj.dtype), rand_prj).to(torch.float32)

    got = qjl_gqa_score_reference(
        key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj
    )
    exp = _naive_qjl_gqa_score(
        key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj
    )
    tol = TOLERANCE_MATRIX["qjl_gqa_score"]["pytorch"][dtype]
    assert torch.allclose(got, exp, **tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bits,group_size", [(2, 8), (4, 16)])
@torch.no_grad()
def test_parity_quantized_bmm_vulkan_vs_pytorch(dtype, bits, group_size):
    torch.manual_seed(2030 + bits)
    b, h, m, k, n = 2, 2, 3, 16, 32
    pack_factor = 32 // bits
    n_packed = n // pack_factor
    n_groups = n // group_size

    fA = torch.randn(b, h, m, k, dtype=dtype)
    q_vals = torch.randint(0, 2 ** bits, (b * h, n, k), dtype=torch.int32)
    qB2 = torch.stack([_pack_weight_matrix(q_vals[i], bits) for i in range(b * h)], dim=0)
    qB = qB2.transpose(1, 2).contiguous().view(b, h, k, n_packed)
    scales2 = torch.randn(b * h, n_groups, k, dtype=torch.float32) * 0.2 + 0.05
    zeros2 = torch.randn(b * h, n_groups, k, dtype=torch.float32) * 0.1
    scales = scales2.transpose(1, 2).contiguous().view(b, h, k, n_groups).to(dtype)
    zeros = zeros2.transpose(1, 2).contiguous().view(b, h, k, n_groups).to(dtype)

    got = quantized_bmm_reference(group_size, fA, qB, scales, zeros, bits, mqa=False)
    exp = _naive_quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False)
    tol = TOLERANCE_MATRIX["quantized_bmm"]["pytorch"][dtype]
    assert torch.allclose(got, exp, **tol)


@pytest.mark.skipif(not (torch.cuda.is_available() and cuda_backend.is_cuda_available()), reason="CUDA kernels unavailable on this host")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bits,group_size", [(2, 8), (4, 16)])
@torch.no_grad()
def test_parity_quantized_bmm_cuda_vs_vulkan_reference(dtype, bits, group_size):
    torch.manual_seed(2035 + bits)
    device = torch.device("cuda")
    b, h, m, k, n = 1, 2, 3, 16, 32
    pack_factor = 32 // bits
    n_packed = n // pack_factor
    n_groups = n // group_size

    fA = torch.randn(b, h, m, k, dtype=dtype, device=device)
    q_vals = torch.randint(0, 2 ** bits, (b * h, n, k), dtype=torch.int32, device=device)
    qB2 = torch.stack([_pack_weight_matrix(q_vals[i], bits) for i in range(b * h)], dim=0)
    qB = qB2.transpose(1, 2).contiguous().view(b, h, k, n_packed)
    scales2 = torch.randn(b * h, n_groups, k, dtype=torch.float32, device=device) * 0.2 + 0.05
    zeros2 = torch.randn(b * h, n_groups, k, dtype=torch.float32, device=device) * 0.1
    scales = scales2.transpose(1, 2).contiguous().view(b, h, k, n_groups).to(dtype)
    zeros = zeros2.transpose(1, 2).contiguous().view(b, h, k, n_groups).to(dtype)

    got = cuda_backend.quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False)
    exp = quantized_bmm_reference(group_size, fA, qB, scales, zeros, bits, mqa=False)
    tol = TOLERANCE_MATRIX["quantized_bmm"]["cuda"][dtype]
    assert torch.allclose(got, exp, **tol)


@pytest.mark.skipif(not (torch.cuda.is_available() and cuda_backend.is_cuda_available()), reason="CUDA kernels unavailable on this host")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.no_grad()
def test_parity_qjl_quant_cuda_vs_vulkan_reference(dtype):
    torch.manual_seed(2041)
    device = torch.device("cuda")
    b, h, n, g, d = 1, 2, 2, 3, 128
    s, so, o = 64, 32, 8
    key_states = torch.randn(b, h, n, g, d, dtype=dtype, device=device)
    outlier_indices = torch.randint(0, d, (b, h, n, o), dtype=torch.uint8, device=device)
    rand_prj = torch.randn(s, d, dtype=torch.float32, device=device)

    got_kq, got_koq, got_on = cuda_backend.qjl_quant(key_states, outlier_indices, rand_prj, so)
    exp_kq, exp_koq, exp_on = qjl_quant_reference(key_states, outlier_indices, rand_prj, so)
    tol = TOLERANCE_MATRIX["qjl_quant"]["cuda"][dtype]
    assert torch.equal(got_kq, exp_kq)
    assert torch.equal(got_koq, exp_koq)
    assert torch.allclose(got_on, exp_on, **tol)


@pytest.mark.skipif(not (torch.cuda.is_available() and cuda_backend.is_cuda_available()), reason="CUDA kernels unavailable on this host")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.no_grad()
def test_parity_qjl_score_cuda_vs_vulkan_reference(dtype):
    torch.manual_seed(2042)
    device = torch.device("cuda")
    b, h, n, g, d = 1, 2, 2, 2, 128
    s, so, o = 64, 32, 6

    key_signs = torch.where(torch.randn(b, h, n, g, s, device=device) > 0, 1.0, -1.0)
    out_signs = torch.where(torch.randn(b, h, n, g, so, device=device) > 0, 1.0, -1.0)
    key_quant = _pack_signs(key_signs)
    key_outlier_quant = _pack_signs(out_signs)
    key_norm = (torch.rand(b, h, n, g, dtype=torch.float32, device=device) * 2.0 + 0.5).to(dtype)
    key_outlier_norm = (key_norm * 0.5).contiguous()
    outlier_indices = torch.randint(0, d, (b, h, n, o), dtype=torch.uint8, device=device)
    query_states = torch.randn(b, h, d, dtype=dtype, device=device)
    rand_prj = torch.randn(d, s, dtype=torch.float32, device=device)
    query_sketch = torch.matmul(query_states.to(rand_prj.dtype), rand_prj).to(torch.float32)

    got = cuda_backend.qjl_score(
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
    tol = TOLERANCE_MATRIX["qjl_score"]["cuda"][dtype]
    assert torch.allclose(got, exp, **tol)


@pytest.mark.skipif(not (torch.cuda.is_available() and cuda_backend.is_cuda_available()), reason="CUDA kernels unavailable on this host")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.no_grad()
def test_parity_qjl_gqa_score_cuda_vs_vulkan_reference(dtype):
    torch.manual_seed(2043)
    device = torch.device("cuda")
    b, kh, qh, n, g, d = 1, 2, 4, 2, 2, 128
    s, so, o = 64, 32, 5

    key_signs = torch.where(torch.randn(b, kh, n, g, s, device=device) > 0, 1.0, -1.0)
    out_signs = torch.where(torch.randn(b, kh, n, g, so, device=device) > 0, 1.0, -1.0)
    key_quant = _pack_signs(key_signs)
    key_outlier_quant = _pack_signs(out_signs)
    key_norm = (torch.rand(b, kh, n, g, dtype=torch.float32, device=device) * 2.0 + 0.5).to(dtype)
    key_outlier_norm = (key_norm * 0.5).contiguous()
    outlier_indices = torch.randint(0, d, (b, kh, n, o), dtype=torch.uint8, device=device)
    query_states = torch.randn(b, qh, d, dtype=dtype, device=device)
    rand_prj = torch.randn(d, s, dtype=torch.float32, device=device)
    query_sketch = torch.matmul(query_states.to(rand_prj.dtype), rand_prj).to(torch.float32)

    got = cuda_backend.qjl_gqa_score(
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
    tol = TOLERANCE_MATRIX["qjl_gqa_score"]["cuda"][dtype]
    assert torch.allclose(got, exp, **tol)
