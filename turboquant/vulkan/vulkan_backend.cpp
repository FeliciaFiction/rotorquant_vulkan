#include <torch/extension.h>

#include "vk_runtime.h"

namespace py = pybind11;

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
}
