#pragma once

#include <string>

namespace tq {
namespace vulkan {

// Placeholder runtime query used by the initial extension scaffold.
// Real Vulkan instance/device probing will be added in follow-up tasks.
bool runtime_available();
std::string runtime_info();

}  // namespace vulkan
}  // namespace tq

