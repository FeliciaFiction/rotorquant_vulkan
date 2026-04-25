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
}

