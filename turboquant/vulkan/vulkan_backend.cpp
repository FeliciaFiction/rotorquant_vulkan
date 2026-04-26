#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include "vk_runtime.h"

namespace py = pybind11;

namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qjl_quant_impl(
    const torch::Tensor& key_states,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& rand_prj,
    int64_t outlier_sketch_dim
) {
    if (key_states.dim() != 5) {
        throw std::invalid_argument("key_states must be 5D [B,H,N,G,D]");
    }
    if (outlier_indices.dim() != 4) {
        throw std::invalid_argument("outlier_indices must be 4D [B,H,N,O]");
    }
    if (rand_prj.dim() != 2) {
        throw std::invalid_argument("rand_prj must be 2D [S,D]");
    }

    const auto b = key_states.size(0);
    const auto h = key_states.size(1);
    const auto n = key_states.size(2);
    const auto g = key_states.size(3);
    const auto d = key_states.size(4);
    const auto s = rand_prj.size(0);

    if (rand_prj.size(1) != d) {
        throw std::invalid_argument("rand_prj D dimension must match key_states D");
    }
    if (s % 8 != 0) {
        throw std::invalid_argument("sketch_dim must be divisible by 8");
    }
    if (outlier_sketch_dim % 8 != 0) {
        throw std::invalid_argument("outlier_sketch_dim must be divisible by 8");
    }
    if (outlier_sketch_dim > s) {
        throw std::invalid_argument("outlier_sketch_dim cannot exceed sketch_dim");
    }

    auto idx = outlier_indices.to(torch::kLong);
    idx = torch::clamp(idx, /*min=*/0, /*max=*/std::max<int64_t>(d - 1, 0));

    auto mask = torch::zeros(
        {b, h, n, d},
        key_states.options()
    );
    const auto scatter_src = torch::ones(idx.sizes(), key_states.options());
    mask = mask.scatter(-1, idx, scatter_src);
    mask = mask.unsqueeze(-2);  // [B,H,N,1,D]

    const auto proj_dtype = rand_prj.scalar_type();
    const auto ks = key_states.to(proj_dtype);
    const auto mask_proj = mask.to(proj_dtype);
    const auto inlier = ks * (1 - mask_proj);
    const auto outlier = ks * mask_proj;

    const auto sketch_inlier = torch::einsum("...gd,sd->...gs", {inlier, rand_prj});
    const auto sketch_outlier = torch::einsum("...gd,sd->...gs", {outlier, rand_prj});

    constexpr int64_t kBits = 8;
    const auto enc = torch::tensor(
        {1, 2, 4, 8, 16, 32, 64, 128},
        torch::TensorOptions().dtype(torch::kUInt8).device(key_states.device())
    ).view({1, 1, 1, 1, 1, kBits});

    const auto sketch_inlier_bits = sketch_inlier.view({b, h, n, g, s / kBits, kBits});
    const auto key_quant = ((sketch_inlier_bits > 0).to(torch::kUInt8) * enc)
                               .to(torch::kInt16)
                               .sum(-1)
                               .to(torch::kUInt8)
                               .contiguous();

    const auto so = outlier_sketch_dim;
    const auto sketch_outlier_bits = sketch_outlier.slice(-1, 0, so).view(
        {b, h, n, g, so / kBits, kBits}
    );
    const auto key_outlier_quant = ((sketch_outlier_bits > 0).to(torch::kUInt8) * enc)
                                       .to(torch::kInt16)
                                       .sum(-1)
                                       .to(torch::kUInt8)
                                       .contiguous();

    const auto outlier_norms = outlier.to(torch::kFloat32)
                                   .pow(2)
                                   .sum(-1)
                                   .sqrt()
                                   .to(key_states.scalar_type())
                                   .contiguous();
    return {key_quant, key_outlier_quant, outlier_norms};
}

torch::Tensor unpack_sign_bits(
    const torch::Tensor& packed,
    int64_t sketch_dim
) {
    constexpr int64_t kBits = 8;
    if (sketch_dim % kBits != 0) {
        throw std::invalid_argument("sketch_dim must be divisible by 8");
    }
    const auto bytes_expected = sketch_dim / kBits;
    if (packed.size(-1) != bytes_expected) {
        throw std::invalid_argument("packed last dim does not match sketch_dim/8");
    }

    std::vector<int64_t> shift_shape(static_cast<size_t>(packed.dim()) + 1, 1);
    shift_shape.back() = kBits;
    const auto shifts = torch::arange(
        kBits,
        torch::TensorOptions().dtype(torch::kUInt8).device(packed.device())
    ).view(shift_shape);

    const auto expanded = packed.unsqueeze(-1).to(torch::kUInt8);
    const auto unpacked01 = torch::bitwise_and(
        torch::bitwise_right_shift(expanded, shifts),
        torch::scalar_tensor(1, torch::TensorOptions().dtype(torch::kUInt8).device(packed.device()))
    ).to(torch::kFloat32);

    auto out_shape = packed.sizes().vec();
    out_shape.back() = sketch_dim;
    const auto unpacked = unpacked01.reshape(out_shape);
    return unpacked * 2.0 - 1.0;
}

torch::Tensor qjl_score_impl(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    if (key_quant.dim() != 5) {
        throw std::invalid_argument("key_quant must be 5D [B,H,N,G,S/8]");
    }
    if (key_outlier_quant.dim() != 5) {
        throw std::invalid_argument("key_outlier_quant must be 5D [B,H,N,G,SO/8]");
    }
    if (query_states.dim() != 3) {
        throw std::invalid_argument("query_states must be 3D [B,H,D]");
    }
    if (outlier_indices.dim() != 4) {
        throw std::invalid_argument("outlier_indices must be 4D [B,H,N,O]");
    }

    const auto b = key_quant.size(0);
    const auto h = key_quant.size(1);
    const auto n = key_quant.size(2);
    const auto g = key_quant.size(3);
    const auto hash_dim = key_quant.size(4);
    const auto so_hash_dim = key_outlier_quant.size(4);
    const auto s = hash_dim * 8;
    const auto so = so_hash_dim * 8;

    if (key_outlier_quant.size(0) != b || key_outlier_quant.size(1) != h ||
        key_outlier_quant.size(2) != n || key_outlier_quant.size(3) != g) {
        throw std::invalid_argument("key_outlier_quant leading dims must match key_quant");
    }
    if (query_sketch.size(0) != b || query_sketch.size(1) != h || query_sketch.size(2) != s) {
        throw std::invalid_argument("query_sketch must have shape [B,H,S]");
    }
    if (key_norm.size(0) != b || key_norm.size(1) != h || key_norm.size(2) != n || key_norm.size(3) != g) {
        throw std::invalid_argument("key_norm must have shape [B,H,N,G]");
    }
    if (key_outlier_norm.size(0) != b || key_outlier_norm.size(1) != h ||
        key_outlier_norm.size(2) != n || key_outlier_norm.size(3) != g) {
        throw std::invalid_argument("key_outlier_norm must have shape [B,H,N,G]");
    }
    if (outlier_indices.size(0) != b || outlier_indices.size(1) != h || outlier_indices.size(2) != n) {
        throw std::invalid_argument("outlier_indices leading dims must match [B,H,N]");
    }

    const auto d = query_states.size(2);
    if (query_states.size(0) != b || query_states.size(1) != h) {
        throw std::invalid_argument("query_states leading dims must match [B,H]");
    }
    if (rand_prj.size(0) != d || rand_prj.size(1) != s) {
        throw std::invalid_argument("rand_prj must have shape [D,S]");
    }

    auto out_idx = outlier_indices.to(torch::kLong);
    out_idx = torch::clamp(out_idx, /*min=*/0, /*max=*/std::max<int64_t>(d - 1, 0));

    const auto o = out_idx.size(3);
    auto q_expanded = query_states.unsqueeze(2).expand({b, h, n, d});
    auto q_vals = q_expanded.gather(-1, out_idx);  // [B,H,N,O]

    auto rand_rows = rand_prj.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand({b, h, n, d, s});
    auto row_idx = out_idx.unsqueeze(-1).expand({b, h, n, o, s});
    auto prj_rows = rand_rows.gather(3, row_idx);  // [B,H,N,O,S]
    auto q_outlier_sketch = (q_vals.unsqueeze(-1).to(prj_rows.scalar_type()) * prj_rows)
                                .sum(3)
                                .to(torch::kFloat32);  // [B,H,N,S]

    const auto signs_k = unpack_sign_bits(key_quant.to(torch::kUInt8), s);   // [B,H,N,G,S]
    const auto signs_o = unpack_sign_bits(key_outlier_quant.to(torch::kUInt8), so);  // [B,H,N,G,SO]

    const auto q_sketch_corr = query_sketch.to(torch::kFloat32).unsqueeze(2).unsqueeze(2)
                               - q_outlier_sketch.unsqueeze(3);
    const auto k_inner = (signs_k * q_sketch_corr).sum(-1);  // [B,H,N,G]

    const auto q_out = q_outlier_sketch.slice(-1, 0, so).unsqueeze(3);  // [B,H,N,1,SO]
    const auto out_inner = (signs_o * q_out).sum(-1);  // [B,H,N,G]

    const auto scl = std::sqrt(M_PI / 2.0) / static_cast<double>(s);
    const auto scl_o = std::sqrt(M_PI / 2.0) / static_cast<double>(so > 0 ? so : 1);

    const auto norm_o = key_outlier_norm.to(torch::kFloat32);
    const auto norm_k_sq = key_norm.to(torch::kFloat32).pow(2) - norm_o.pow(2);
    const auto norm_k = torch::sqrt(torch::clamp_min(norm_k_sq, 0.0));

    const auto scores = scl * norm_k * k_inner + scl_o * norm_o * out_inner;
    return scores.reshape({b, h, n * g, 1}).contiguous();
}

torch::Tensor qjl_gqa_score_impl(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    if (key_quant.dim() != 5) {
        throw std::invalid_argument("key_quant must be 5D [B,KH,N,G,S/8]");
    }
    if (key_outlier_quant.dim() != 5) {
        throw std::invalid_argument("key_outlier_quant must be 5D [B,KH,N,G,SO/8]");
    }
    if (query_states.dim() != 3) {
        throw std::invalid_argument("query_states must be 3D [B,QH,D]");
    }
    if (outlier_indices.dim() != 4) {
        throw std::invalid_argument("outlier_indices must be 4D [B,KH,N,O]");
    }

    const auto b = key_quant.size(0);
    const auto kh = key_quant.size(1);
    const auto n = key_quant.size(2);
    const auto g = key_quant.size(3);
    const auto hash_dim = key_quant.size(4);
    const auto so_hash_dim = key_outlier_quant.size(4);
    const auto s = hash_dim * 8;
    const auto so = so_hash_dim * 8;

    if (key_outlier_quant.size(0) != b || key_outlier_quant.size(1) != kh ||
        key_outlier_quant.size(2) != n || key_outlier_quant.size(3) != g) {
        throw std::invalid_argument("key_outlier_quant leading dims must match key_quant");
    }
    if (key_norm.size(0) != b || key_norm.size(1) != kh || key_norm.size(2) != n || key_norm.size(3) != g) {
        throw std::invalid_argument("key_norm must have shape [B,KH,N,G]");
    }
    if (key_outlier_norm.size(0) != b || key_outlier_norm.size(1) != kh ||
        key_outlier_norm.size(2) != n || key_outlier_norm.size(3) != g) {
        throw std::invalid_argument("key_outlier_norm must have shape [B,KH,N,G]");
    }
    if (outlier_indices.size(0) != b || outlier_indices.size(1) != kh || outlier_indices.size(2) != n) {
        throw std::invalid_argument("outlier_indices leading dims must match [B,KH,N]");
    }

    const auto qh = query_states.size(1);
    const auto d = query_states.size(2);
    if (query_states.size(0) != b) {
        throw std::invalid_argument("query_states batch dim must match key tensors");
    }
    if (kh <= 0 || qh % kh != 0) {
        throw std::invalid_argument("query head count must be divisible by kv head count");
    }
    const auto gqa_group_size = qh / kh;

    if (query_sketch.size(0) != b || query_sketch.size(1) != qh || query_sketch.size(2) != s) {
        throw std::invalid_argument("query_sketch must have shape [B,QH,S]");
    }
    if (rand_prj.size(0) != d || rand_prj.size(1) != s) {
        throw std::invalid_argument("rand_prj must have shape [D,S]");
    }

    auto out_idx_kh = outlier_indices.to(torch::kLong);
    out_idx_kh = torch::clamp(out_idx_kh, /*min=*/0, /*max=*/std::max<int64_t>(d - 1, 0));
    auto out_idx_qh = out_idx_kh.repeat_interleave(gqa_group_size, /*dim=*/1);  // [B,QH,N,O]

    auto q_expanded = query_states.unsqueeze(2).expand({b, qh, n, d});
    auto q_vals = q_expanded.gather(-1, out_idx_qh);  // [B,QH,N,O]

    const auto o = out_idx_qh.size(3);
    auto rand_rows = rand_prj.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand({b, qh, n, d, s});
    auto row_idx = out_idx_qh.unsqueeze(-1).expand({b, qh, n, o, s});
    auto prj_rows = rand_rows.gather(3, row_idx);  // [B,QH,N,O,S]
    auto q_outlier_sketch = (q_vals.unsqueeze(-1).to(prj_rows.scalar_type()) * prj_rows)
                                .sum(3)
                                .to(torch::kFloat32);  // [B,QH,N,S]

    const auto signs_k = unpack_sign_bits(key_quant.to(torch::kUInt8), s)
                             .repeat_interleave(gqa_group_size, /*dim=*/1);   // [B,QH,N,G,S]
    const auto signs_o = unpack_sign_bits(key_outlier_quant.to(torch::kUInt8), so)
                             .repeat_interleave(gqa_group_size, /*dim=*/1);  // [B,QH,N,G,SO]

    const auto q_sketch_corr = query_sketch.to(torch::kFloat32).unsqueeze(2).unsqueeze(2)
                               - q_outlier_sketch.unsqueeze(3);
    const auto k_inner = (signs_k * q_sketch_corr).sum(-1);  // [B,QH,N,G]

    const auto q_out = q_outlier_sketch.slice(-1, 0, so).unsqueeze(3);  // [B,QH,N,1,SO]
    const auto out_inner = (signs_o * q_out).sum(-1);  // [B,QH,N,G]

    const auto scl = std::sqrt(M_PI / 2.0) / static_cast<double>(s);
    const auto scl_o = std::sqrt(M_PI / 2.0) / static_cast<double>(so > 0 ? so : 1);

    const auto norm_o = key_outlier_norm.to(torch::kFloat32).repeat_interleave(gqa_group_size, /*dim=*/1);
    const auto norm_k_sq = key_norm.to(torch::kFloat32).repeat_interleave(gqa_group_size, /*dim=*/1).pow(2) - norm_o.pow(2);
    const auto norm_k = torch::sqrt(torch::clamp_min(norm_k_sq, 0.0));

    const auto scores = scl * norm_k * k_inner + scl_o * norm_o * out_inner;
    return scores.reshape({b, qh, n * g, 1}).contiguous();
}

void ensure_key_proj_dtype(
    const torch::Tensor& key_states,
    const torch::Tensor& rand_prj,
    torch::ScalarType key_dtype,
    torch::ScalarType proj_dtype
) {
    if (key_states.scalar_type() != key_dtype || rand_prj.scalar_type() != proj_dtype) {
        throw std::invalid_argument("dtype mismatch for qjl_quant variant");
    }
}

void ensure_query_proj_dtype(
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj,
    torch::ScalarType query_dtype,
    torch::ScalarType proj_dtype
) {
    if (query_states.scalar_type() != query_dtype || rand_prj.scalar_type() != proj_dtype) {
        throw std::invalid_argument("dtype mismatch for qjl_score variant");
    }
}

void ensure_quantized_bmm_dtype(
    const torch::Tensor& fA,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    torch::ScalarType expected_dtype
) {
    if (fA.scalar_type() != expected_dtype ||
        scales.scalar_type() != expected_dtype ||
        zeros.scalar_type() != expected_dtype) {
        throw std::invalid_argument("dtype mismatch for quantized_bmm variant");
    }
}

torch::Tensor dequantize_packed_weights_impl(
    const torch::Tensor& q_packed,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int64_t bits,
    int64_t group_size
) {
    if (bits != 2 && bits != 4) {
        throw std::invalid_argument("bits must be one of {2, 4}");
    }
    if (group_size <= 0) {
        throw std::invalid_argument("group_size must be positive");
    }
    if (q_packed.dim() != 2 || scales.dim() != 2 || zeros.dim() != 2) {
        throw std::invalid_argument("q_packed/scales/zeros must be 2D");
    }

    const auto pack_factor = 32 / bits;
    const auto n_packed = q_packed.size(0);
    const auto k = q_packed.size(1);
    const auto n = n_packed * pack_factor;
    if (n % group_size != 0) {
        throw std::invalid_argument("N must be divisible by group_size");
    }

    const auto groups = n / group_size;
    if (scales.size(0) != groups || scales.size(1) != k ||
        zeros.size(0) != groups || zeros.size(1) != k) {
        throw std::invalid_argument("scales/zeros shape mismatch for packed matrix");
    }

    const auto dev = q_packed.device();
    auto oc = torch::arange(n, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto packed_idx = torch::floor_divide(oc, pack_factor);
    auto shift = (oc.remainder(pack_factor) * bits).unsqueeze(1);
    auto group_idx = torch::floor_divide(oc, group_size);

    auto words = q_packed.to(torch::kLong).index_select(0, packed_idx);
    auto qvals = torch::bitwise_and(
        torch::bitwise_right_shift(words, shift),
        torch::scalar_tensor((1 << bits) - 1, torch::TensorOptions().dtype(torch::kLong).device(dev))
    ).to(torch::kFloat32);

    auto scales_exp = scales.to(torch::kFloat32).index_select(0, group_idx);
    auto zeros_exp = zeros.to(torch::kFloat32).index_select(0, group_idx);
    return (qvals * scales_exp + zeros_exp).contiguous();
}

torch::Tensor quantized_bmm_impl(
    int64_t group_size,
    const torch::Tensor& fA,
    const torch::Tensor& qB,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int64_t bits,
    bool mqa
) {
    if (fA.dim() != 4 || qB.dim() != 4) {
        throw std::invalid_argument("fA and qB must be 4D");
    }
    if (scales.dim() != 4 || zeros.dim() != 4) {
        throw std::invalid_argument("scales and zeros must be 4D");
    }
    if (bits != 2 && bits != 4) {
        throw std::invalid_argument("bits must be one of {2, 4}");
    }

    const auto b = fA.size(0);
    const auto h = fA.size(1);
    const auto m = fA.size(2);
    const auto k = fA.size(3);
    const auto feat_per_int = 32 / bits;
    const auto n_packed = qB.size(3);
    const auto n = n_packed * feat_per_int;
    const auto flatten_b = mqa ? b : b * h;

    auto fA2 = fA.reshape({-1, m, k}).contiguous();
    auto qB2 = qB.reshape({-1, k, n_packed}).transpose(1, 2).contiguous();
    auto scales2 = scales.reshape({flatten_b, scales.size(2), scales.size(3)}).transpose(1, 2).contiguous();
    auto zeros2 = zeros.reshape({flatten_b, zeros.size(2), zeros.size(3)}).transpose(1, 2).contiguous();

    if (qB2.size(0) != flatten_b) {
        throw std::invalid_argument("qB flattened batch does not match expected backend layout");
    }
    if (scales2.size(0) != flatten_b || zeros2.size(0) != flatten_b) {
        throw std::invalid_argument("scales/zeros flattened batch does not match expected backend layout");
    }
    if (!scales2.sizes().equals(zeros2.sizes())) {
        throw std::invalid_argument("scales and zeros shapes must match");
    }

    auto out = torch::empty({b * h, m, n}, fA.options());
    for (int64_t batch_idx = 0; batch_idx < b * h; ++batch_idx) {
        const int64_t w_batch = mqa ? (batch_idx / h) : batch_idx;
        auto w = dequantize_packed_weights_impl(
            qB2.select(0, w_batch),
            scales2.select(0, w_batch),
            zeros2.select(0, w_batch),
            bits,
            group_size
        );
        auto batch_out = torch::matmul(
            fA2.select(0, batch_idx).to(torch::kFloat32),
            w.transpose(0, 1)
        ).to(fA.scalar_type());
        out.select(0, batch_idx).copy_(batch_out);
    }

    return out.view({b, h, m, n}).contiguous();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qjl_quant_half_half(
    const torch::Tensor& key_states,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& rand_prj,
    int64_t outlier_sketch_dim
) {
    ensure_key_proj_dtype(key_states, rand_prj, torch::kFloat16, torch::kFloat16);
    return qjl_quant_impl(key_states, outlier_indices, rand_prj, outlier_sketch_dim);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qjl_quant_half_float(
    const torch::Tensor& key_states,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& rand_prj,
    int64_t outlier_sketch_dim
) {
    ensure_key_proj_dtype(key_states, rand_prj, torch::kFloat16, torch::kFloat32);
    return qjl_quant_impl(key_states, outlier_indices, rand_prj, outlier_sketch_dim);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qjl_quant_float_float(
    const torch::Tensor& key_states,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& rand_prj,
    int64_t outlier_sketch_dim
) {
    ensure_key_proj_dtype(key_states, rand_prj, torch::kFloat32, torch::kFloat32);
    return qjl_quant_impl(key_states, outlier_indices, rand_prj, outlier_sketch_dim);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qjl_quant_bf16_bf16(
    const torch::Tensor& key_states,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& rand_prj,
    int64_t outlier_sketch_dim
) {
    ensure_key_proj_dtype(key_states, rand_prj, torch::kBFloat16, torch::kBFloat16);
    return qjl_quant_impl(key_states, outlier_indices, rand_prj, outlier_sketch_dim);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qjl_quant_bf16_float(
    const torch::Tensor& key_states,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& rand_prj,
    int64_t outlier_sketch_dim
) {
    ensure_key_proj_dtype(key_states, rand_prj, torch::kBFloat16, torch::kFloat32);
    return qjl_quant_impl(key_states, outlier_indices, rand_prj, outlier_sketch_dim);
}

torch::Tensor qjl_score_vulkan_half_half(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    ensure_query_proj_dtype(query_states, rand_prj, torch::kFloat16, torch::kFloat16);
    return qjl_score_impl(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj
    );
}

torch::Tensor qjl_score_vulkan_half_float(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    ensure_query_proj_dtype(query_states, rand_prj, torch::kFloat16, torch::kFloat32);
    return qjl_score_impl(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj
    );
}

torch::Tensor qjl_score_vulkan_float_float(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    ensure_query_proj_dtype(query_states, rand_prj, torch::kFloat32, torch::kFloat32);
    return qjl_score_impl(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj
    );
}

torch::Tensor qjl_score_vulkan_bf16_bf16(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    ensure_query_proj_dtype(query_states, rand_prj, torch::kBFloat16, torch::kBFloat16);
    return qjl_score_impl(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj
    );
}

torch::Tensor qjl_score_vulkan_bf16_float(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    ensure_query_proj_dtype(query_states, rand_prj, torch::kBFloat16, torch::kFloat32);
    return qjl_score_impl(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj
    );
}

torch::Tensor qjl_gqa_score_vulkan_half_half(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    ensure_query_proj_dtype(query_states, rand_prj, torch::kFloat16, torch::kFloat16);
    return qjl_gqa_score_impl(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj
    );
}

torch::Tensor qjl_gqa_score_vulkan_half_float(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    ensure_query_proj_dtype(query_states, rand_prj, torch::kFloat16, torch::kFloat32);
    return qjl_gqa_score_impl(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj
    );
}

torch::Tensor qjl_gqa_score_vulkan_float_float(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    ensure_query_proj_dtype(query_states, rand_prj, torch::kFloat32, torch::kFloat32);
    return qjl_gqa_score_impl(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj
    );
}

torch::Tensor qjl_gqa_score_vulkan_bf16_bf16(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    ensure_query_proj_dtype(query_states, rand_prj, torch::kBFloat16, torch::kBFloat16);
    return qjl_gqa_score_impl(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj
    );
}

torch::Tensor qjl_gqa_score_vulkan_bf16_float(
    const torch::Tensor& key_quant,
    const torch::Tensor& key_outlier_quant,
    const torch::Tensor& key_norm,
    const torch::Tensor& key_outlier_norm,
    const torch::Tensor& outlier_indices,
    const torch::Tensor& query_sketch,
    const torch::Tensor& query_states,
    const torch::Tensor& rand_prj
) {
    ensure_query_proj_dtype(query_states, rand_prj, torch::kBFloat16, torch::kFloat32);
    return qjl_gqa_score_impl(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj
    );
}

torch::Tensor quantized_bmm_vulkan_half(
    int64_t group_size,
    const torch::Tensor& fA,
    const torch::Tensor& qB,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int64_t bits,
    bool mqa
) {
    ensure_quantized_bmm_dtype(fA, scales, zeros, torch::kFloat16);
    return quantized_bmm_impl(group_size, fA, qB, scales, zeros, bits, mqa);
}

torch::Tensor quantized_bmm_vulkan_float(
    int64_t group_size,
    const torch::Tensor& fA,
    const torch::Tensor& qB,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int64_t bits,
    bool mqa
) {
    ensure_quantized_bmm_dtype(fA, scales, zeros, torch::kFloat32);
    return quantized_bmm_impl(group_size, fA, qB, scales, zeros, bits, mqa);
}

torch::Tensor quantized_bmm_vulkan_bf16(
    int64_t group_size,
    const torch::Tensor& fA,
    const torch::Tensor& qB,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int64_t bits,
    bool mqa
) {
    ensure_quantized_bmm_dtype(fA, scales, zeros, torch::kBFloat16);
    return quantized_bmm_impl(group_size, fA, qB, scales, zeros, bits, mqa);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TurboQuant Vulkan backend extension scaffold";
    m.def("is_vulkan_runtime_available", []() {
        return tq::vulkan::runtime_available();
    });
    m.def("vulkan_runtime_info", []() {
        return tq::vulkan::runtime_info();
    });
    m.def("vulkan_compile_features", []() {
        py::dict d;
#ifdef GGML_VULKAN_COOPMAT_GLSLC_SUPPORT
        d["GGML_VULKAN_COOPMAT_GLSLC_SUPPORT"] = true;
#else
        d["GGML_VULKAN_COOPMAT_GLSLC_SUPPORT"] = false;
#endif
#ifdef GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT
        d["GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"] = true;
#else
        d["GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"] = false;
#endif
#ifdef GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        d["GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"] = true;
#else
        d["GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"] = false;
#endif
#ifdef GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT
        d["GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT"] = true;
#else
        d["GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT"] = false;
#endif
        return d;
    });
    m.def("qjl_quant_half_half", &qjl_quant_half_half);
    m.def("qjl_quant_half_float", &qjl_quant_half_float);
    m.def("qjl_quant_float_float", &qjl_quant_float_float);
    m.def("qjl_quant_bf16_bf16", &qjl_quant_bf16_bf16);
    m.def("qjl_quant_bf16_float", &qjl_quant_bf16_float);
    m.def("qjl_score_vulkan_half_half", &qjl_score_vulkan_half_half);
    m.def("qjl_score_vulkan_half_float", &qjl_score_vulkan_half_float);
    m.def("qjl_score_vulkan_float_float", &qjl_score_vulkan_float_float);
    m.def("qjl_score_vulkan_bf16_bf16", &qjl_score_vulkan_bf16_bf16);
    m.def("qjl_score_vulkan_bf16_float", &qjl_score_vulkan_bf16_float);
    m.def("qjl_gqa_score_vulkan_half_half", &qjl_gqa_score_vulkan_half_half);
    m.def("qjl_gqa_score_vulkan_half_float", &qjl_gqa_score_vulkan_half_float);
    m.def("qjl_gqa_score_vulkan_float_float", &qjl_gqa_score_vulkan_float_float);
    m.def("qjl_gqa_score_vulkan_bf16_bf16", &qjl_gqa_score_vulkan_bf16_bf16);
    m.def("qjl_gqa_score_vulkan_bf16_float", &qjl_gqa_score_vulkan_bf16_float);
    m.def("quantized_bmm_vulkan_half", &quantized_bmm_vulkan_half);
    m.def("quantized_bmm_vulkan_float", &quantized_bmm_vulkan_float);
    m.def("quantized_bmm_vulkan_bf16", &quantized_bmm_vulkan_bf16);
}
