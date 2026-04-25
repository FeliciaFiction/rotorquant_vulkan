import torch


def _unpack_sign_bits(packed: torch.Tensor, sketch_dim: int) -> torch.Tensor:
    """
    Unpack uint8 sign bits into {-1, +1} float tensor.
    Input shape:  [..., sketch_dim/8]
    Output shape: [..., sketch_dim]
    """
    bits = 8
    if sketch_dim % bits != 0:
        raise ValueError("sketch_dim must be divisible by 8")
    bytes_expected = sketch_dim // bits
    if packed.shape[-1] != bytes_expected:
        raise ValueError("packed last dim does not match sketch_dim/8")

    shifts = torch.arange(bits, device=packed.device, dtype=torch.uint8).view(1, 1, 1, 1, 1, bits)
    expanded = packed.unsqueeze(-1)
    unpacked01 = ((expanded >> shifts) & 1).to(torch.float32)
    unpacked = unpacked01.view(*packed.shape[:-1], sketch_dim)
    return unpacked * 2.0 - 1.0


def qjl_quant_reference(
    key_states: torch.Tensor,
    outlier_indices: torch.Tensor,
    rand_prj: torch.Tensor,
    outlier_sketch_dim: int,
):
    """
    Reference implementation of qjl_quant kernel logic.

    Shapes:
      key_states:      [B, H, N, G, D]
      outlier_indices: [B, H, N, O]
      rand_prj:        [S, D]
    Returns:
      key_quant:         [B, H, N, G, S/8] uint8
      key_outlier_quant: [B, H, N, G, outlier_sketch_dim/8] uint8
      outlier_norms:     [B, H, N, G] dtype(key_states)
    """
    if key_states.dim() != 5:
        raise ValueError("key_states must be 5D [B,H,N,G,D]")
    if outlier_indices.dim() != 4:
        raise ValueError("outlier_indices must be 4D [B,H,N,O]")
    if rand_prj.dim() != 2:
        raise ValueError("rand_prj must be 2D [S,D]")

    b, h, n, g, d = key_states.shape
    s = rand_prj.shape[0]
    if rand_prj.shape[1] != d:
        raise ValueError("rand_prj D dimension must match key_states D")
    if s % 8 != 0:
        raise ValueError("sketch_dim must be divisible by 8")
    if outlier_sketch_dim % 8 != 0:
        raise ValueError("outlier_sketch_dim must be divisible by 8")
    if outlier_sketch_dim > s:
        raise ValueError("outlier_sketch_dim cannot exceed sketch_dim")

    mask = torch.zeros((b, h, n, d), device=key_states.device, dtype=key_states.dtype)
    idx = outlier_indices.long().clamp(min=0, max=max(d - 1, 0))
    mask.scatter_(-1, idx, 1.0)
    mask = mask.unsqueeze(-2)  # [B,H,N,1,D], shared across group entries

    proj_dtype = rand_prj.dtype
    ks = key_states.to(proj_dtype)
    inlier = ks * (1 - mask.to(proj_dtype))
    outlier = ks * mask.to(proj_dtype)

    # [B,H,N,G,S]
    sketch_inlier = torch.einsum("...gd,sd->...gs", inlier, rand_prj)
    sketch_outlier = torch.einsum("...gd,sd->...gs", outlier, rand_prj)

    # Pack sign bits into bytes.
    bits = 8
    enc = (2 ** torch.arange(bits, device=key_states.device, dtype=torch.uint8)).view(1, 1, 1, 1, 1, bits)

    sketch_inlier = sketch_inlier.view(b, h, n, g, s // bits, bits)
    key_quant = ((sketch_inlier > 0).to(torch.uint8) * enc).sum(dim=-1, dtype=torch.uint8)

    so = outlier_sketch_dim
    sketch_outlier = sketch_outlier[..., :so].view(b, h, n, g, so // bits, bits)
    key_outlier_quant = ((sketch_outlier > 0).to(torch.uint8) * enc).sum(dim=-1, dtype=torch.uint8)

    outlier_norms = torch.sqrt((outlier.to(torch.float32) ** 2).sum(dim=-1)).to(key_states.dtype)
    return key_quant.contiguous(), key_outlier_quant.contiguous(), outlier_norms.contiguous()


def qjl_score_reference(
    key_quant: torch.Tensor,
    key_outlier_quant: torch.Tensor,
    key_norm: torch.Tensor,
    key_outlier_norm: torch.Tensor,
    outlier_indices: torch.Tensor,
    query_sketch: torch.Tensor,
    query_states: torch.Tensor,
    rand_prj: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation of qjl_score kernel logic.

    Shapes:
      key_quant:         [B,H,N,G,S/8]       uint8
      key_outlier_quant: [B,H,N,G,SO/8]      uint8
      key_norm:          [B,H,N,G]           float16/float32/bfloat16
      key_outlier_norm:  [B,H,N,G]           float16/float32/bfloat16
      outlier_indices:   [B,H,N,O]           int/uint
      query_sketch:      [B,H,S]             float32
      query_states:      [B,H,D]             float16/float32/bfloat16
      rand_prj:          [D,S]               float16/float32/bfloat16

    Returns:
      scores: [B,H,N*G,1] float32
    """
    if key_quant.dim() != 5:
        raise ValueError("key_quant must be 5D [B,H,N,G,S/8]")
    if key_outlier_quant.dim() != 5:
        raise ValueError("key_outlier_quant must be 5D [B,H,N,G,SO/8]")

    b, h, n, g, hash_dim = key_quant.shape
    so_hash_dim = key_outlier_quant.shape[-1]
    s = hash_dim * 8
    so = so_hash_dim * 8

    if query_sketch.shape != (b, h, s):
        raise ValueError("query_sketch must have shape [B,H,S]")
    if query_states.dim() != 3:
        raise ValueError("query_states must be 3D [B,H,D]")
    d = query_states.shape[-1]
    if rand_prj.shape != (d, s):
        raise ValueError("rand_prj must have shape [D,S]")
    if outlier_indices.shape[:3] != (b, h, n):
        raise ValueError("outlier_indices leading dims must match [B,H,N]")

    out_idx = outlier_indices.long().clamp(min=0, max=max(d - 1, 0))

    # q_outlier_sketch[b,h,n,s] = sum_i query_states[b,h,out_idx_i] * rand_prj[out_idx_i, s]
    q_expanded = query_states.unsqueeze(2).expand(b, h, n, d)
    q_vals = torch.gather(q_expanded, -1, out_idx)  # [B,H,N,O]
    prj_rows = rand_prj[out_idx]                     # [B,H,N,O,S]
    q_outlier_sketch = (q_vals.unsqueeze(-1).to(prj_rows.dtype) * prj_rows).sum(dim=3).to(torch.float32)

    signs_k = _unpack_sign_bits(key_quant.to(torch.uint8), s)                      # [B,H,N,G,S]
    signs_o = _unpack_sign_bits(key_outlier_quant.to(torch.uint8), so)             # [B,H,N,G,SO]

    q_sketch_corr = query_sketch.to(torch.float32).unsqueeze(2).unsqueeze(2) - q_outlier_sketch.unsqueeze(3)
    k_inner = (signs_k * q_sketch_corr).sum(dim=-1)                                 # [B,H,N,G]

    q_out = q_outlier_sketch[..., :so].unsqueeze(3)                                 # [B,H,N,1,SO]
    out_inner = (signs_o * q_out).sum(dim=-1)                                       # [B,H,N,G]

    scl = torch.sqrt(torch.tensor(torch.pi / 2.0, dtype=torch.float32)) / float(s)
    scl_o = torch.sqrt(torch.tensor(torch.pi / 2.0, dtype=torch.float32)) / float(so if so > 0 else 1)

    norm_o = key_outlier_norm.to(torch.float32)
    norm_k_sq = key_norm.to(torch.float32) ** 2 - norm_o ** 2
    norm_k = torch.sqrt(torch.clamp(norm_k_sq, min=0.0))

    scores = scl * norm_k * k_inner + scl_o * norm_o * out_inner
    return scores.reshape(b, h, n * g, 1).contiguous()


def qjl_gqa_score_reference(
    key_quant: torch.Tensor,
    key_outlier_quant: torch.Tensor,
    key_norm: torch.Tensor,
    key_outlier_norm: torch.Tensor,
    outlier_indices: torch.Tensor,
    query_sketch: torch.Tensor,
    query_states: torch.Tensor,
    rand_prj: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation of qjl_gqa_score kernel logic.

    Shapes:
      key_quant:         [B,KH,N,G,S/8]       uint8
      key_outlier_quant: [B,KH,N,G,SO/8]      uint8
      key_norm:          [B,KH,N,G]           float16/float32/bfloat16
      key_outlier_norm:  [B,KH,N,G]           float16/float32/bfloat16
      outlier_indices:   [B,KH,N,O]           int/uint
      query_sketch:      [B,QH,S]             float32
      query_states:      [B,QH,D]             float16/float32/bfloat16
      rand_prj:          [D,S]                float16/float32/bfloat16

    Returns:
      scores: [B,QH,N*G,1] float32
    """
    if key_quant.dim() != 5:
        raise ValueError("key_quant must be 5D [B,KH,N,G,S/8]")
    if key_outlier_quant.dim() != 5:
        raise ValueError("key_outlier_quant must be 5D [B,KH,N,G,SO/8]")
    if query_states.dim() != 3:
        raise ValueError("query_states must be 3D [B,QH,D]")

    b, kh, n, g, hash_dim = key_quant.shape
    so_hash_dim = key_outlier_quant.shape[-1]
    s = hash_dim * 8
    so = so_hash_dim * 8

    bq, qh, d = query_states.shape
    if bq != b:
        raise ValueError("query_states batch dim must match key tensors")
    if qh % kh != 0:
        raise ValueError("query head count must be divisible by kv head count")
    gqa_group_size = qh // kh

    if query_sketch.shape != (b, qh, s):
        raise ValueError("query_sketch must have shape [B,QH,S]")
    if rand_prj.shape != (d, s):
        raise ValueError("rand_prj must have shape [D,S]")
    if outlier_indices.shape[:3] != (b, kh, n):
        raise ValueError("outlier_indices leading dims must match [B,KH,N]")

    out_idx_kh = outlier_indices.long().clamp(min=0, max=max(d - 1, 0))
    out_idx_qh = out_idx_kh.repeat_interleave(gqa_group_size, dim=1)

    q_expanded = query_states.unsqueeze(2).expand(b, qh, n, d)
    q_vals = torch.gather(q_expanded, -1, out_idx_qh)  # [B,QH,N,O]
    prj_rows = rand_prj[out_idx_qh]                     # [B,QH,N,O,S]
    q_outlier_sketch = (q_vals.unsqueeze(-1).to(prj_rows.dtype) * prj_rows).sum(dim=3).to(torch.float32)

    signs_k = _unpack_sign_bits(key_quant.to(torch.uint8), s).repeat_interleave(gqa_group_size, dim=1)
    signs_o = _unpack_sign_bits(key_outlier_quant.to(torch.uint8), so).repeat_interleave(gqa_group_size, dim=1)

    q_sketch_corr = query_sketch.to(torch.float32).unsqueeze(2).unsqueeze(2) - q_outlier_sketch.unsqueeze(3)
    k_inner = (signs_k * q_sketch_corr).sum(dim=-1)

    q_out = q_outlier_sketch[..., :so].unsqueeze(3)
    out_inner = (signs_o * q_out).sum(dim=-1)

    scl = torch.sqrt(torch.tensor(torch.pi / 2.0, dtype=torch.float32)) / float(s)
    scl_o = torch.sqrt(torch.tensor(torch.pi / 2.0, dtype=torch.float32)) / float(so if so > 0 else 1)

    norm_o = key_outlier_norm.to(torch.float32).repeat_interleave(gqa_group_size, dim=1)
    norm_k_sq = key_norm.to(torch.float32).repeat_interleave(gqa_group_size, dim=1) ** 2 - norm_o ** 2
    norm_k = torch.sqrt(torch.clamp(norm_k_sq, min=0.0))

    scores = scl * norm_k * k_inner + scl_o * norm_o * out_inner
    return scores.reshape(b, qh, n * g, 1).contiguous()


def _dequantize_packed_weights_reference(
    q_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """
    Dequantize one packed weight matrix.

    Shapes:
      q_packed: [N_packed, K] int/uint
      scales:   [N/group_size, K] fp
      zeros:    [N/group_size, K] fp
    Returns:
      w: [N, K] fp32
    """
    if bits not in (2, 4):
        raise ValueError("bits must be one of {2, 4}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    pack_factor = 32 // bits
    n_packed, k = q_packed.shape
    n = n_packed * pack_factor
    if n % group_size != 0:
        raise ValueError("N must be divisible by group_size")

    groups = n // group_size
    if scales.shape != (groups, k) or zeros.shape != (groups, k):
        raise ValueError("scales/zeros shape mismatch for packed matrix")

    oc = torch.arange(n, device=q_packed.device, dtype=torch.long)
    packed_idx = oc // pack_factor
    shift = (oc % pack_factor) * bits
    group_idx = oc // group_size

    words = q_packed.to(torch.int64)[packed_idx]  # [N, K]
    qvals = ((words >> shift[:, None]) & ((1 << bits) - 1)).to(torch.float32)
    w = qvals * scales.to(torch.float32)[group_idx] + zeros.to(torch.float32)[group_idx]
    return w.contiguous()


def quantized_bmm_reference(
    group_size: int,
    fA: torch.Tensor,
    qB: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bits: int,
    mqa: bool = False,
) -> torch.Tensor:
    """
    Reference implementation of cuda_backend.quantized_bmm behavior.

    Shapes:
      fA:     [B, H, M, K]
      qB:     [B, H_or_1, K, N_packed]
      scales: [B, H_or_1, K, N/group_size]
      zeros:  [B, H_or_1, K, N/group_size]
    Returns:
      out:    [B, H, M, N]
    """
    if fA.dim() != 4 or qB.dim() != 4:
        raise ValueError("fA and qB must be 4D")
    if bits not in (2, 4):
        raise ValueError("bits must be one of {2, 4}")

    b, h, m, k = fA.shape
    feat_per_int = 32 // bits
    n = qB.shape[-1] * feat_per_int

    fA2 = fA.view(-1, m, k).contiguous()
    qB2 = qB.reshape(-1, k, qB.shape[-1]).transpose(1, 2).contiguous()

    flatten_b = b * h if not mqa else b
    scales2 = scales.view(flatten_b, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
    zeros2 = zeros.view(flatten_b, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()

    if qB2.shape[0] != flatten_b:
        raise ValueError("qB flattened batch does not match expected backend layout")
    if scales2.shape[0] != flatten_b or zeros2.shape[0] != flatten_b:
        raise ValueError("scales/zeros flattened batch does not match expected backend layout")
    if scales2.shape != zeros2.shape:
        raise ValueError("scales and zeros shapes must match")

    out = torch.empty((b * h, m, n), device=fA.device, dtype=fA.dtype)
    for batch_idx in range(b * h):
        w_batch = batch_idx if not mqa else (batch_idx // h)
        w = _dequantize_packed_weights_reference(
            qB2[w_batch],
            scales2[w_batch],
            zeros2[w_batch],
            bits=bits,
            group_size=group_size,
        )
        out[batch_idx] = torch.matmul(fA2[batch_idx].to(torch.float32), w.transpose(0, 1)).to(fA.dtype)

    return out.view(b, h, m, n).contiguous()
