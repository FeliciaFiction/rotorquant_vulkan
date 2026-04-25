import torch


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

