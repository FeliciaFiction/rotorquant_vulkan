#include <torch/extension.h>

#include <algorithm>
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
}
