import turboquant.vulkan_backend as vk


def _base_caps():
    return {
        "runtime_available": True,
        "runtime_info_raw": "",
        "api_version": "1.3.290",
        "vendor_id": None,
        "vendor_name": None,
        "device_name": "mock-device",
        "extensions": ["VK_KHR_storage_buffer_storage_class"],
        "features": {"computeShader": True},
        "compile_features": {
            "GGML_VULKAN_COOPMAT_GLSLC_SUPPORT": False,
            "GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT": False,
            "GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT": False,
        },
    }


def test_vendor_detection_from_pci_vendor_id_intel(monkeypatch):
    caps = _base_caps()
    caps["vendor_id"] = "0x8086"
    monkeypatch.setattr(vk, "_collect_vulkan_capabilities", lambda: caps)

    report = vk.get_vulkan_capability_report()
    assert report["vendor_name_normalized"] == "intel"
    assert report["vendor_id_parsed"] == 0x8086


def test_vendor_detection_from_vendor_name_amd(monkeypatch):
    caps = _base_caps()
    caps["vendor_name"] = "Advanced Micro Devices, Inc."
    monkeypatch.setattr(vk, "_collect_vulkan_capabilities", lambda: caps)

    report = vk.get_vulkan_capability_report()
    assert report["vendor_name_normalized"] == "amd"


def test_intel_fast_path_enabled_only_with_required_support(monkeypatch):
    caps = _base_caps()
    caps["vendor_id"] = 0x8086
    caps["extensions"] = caps["extensions"] + ["VK_KHR_shader_integer_dot_product"]
    caps["features"]["shaderIntegerDotProduct"] = True
    caps["compile_features"]["GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"] = True
    monkeypatch.setattr(vk, "_collect_vulkan_capabilities", lambda: caps)

    report = vk.get_vulkan_capability_report()
    assert report["optional_fast_paths"]["intel_integer_dot"] is True
    assert report["optional_fast_paths"]["nvidia_coopmat2"] is False
    assert report["optional_fast_paths"]["amd_coopmat"] is False


def test_nvidia_fast_path_disabled_gracefully_when_extension_missing(monkeypatch):
    caps = _base_caps()
    caps["vendor_id"] = "0x10DE"
    caps["compile_features"]["GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"] = True
    monkeypatch.setattr(vk, "_collect_vulkan_capabilities", lambda: caps)

    report = vk.get_vulkan_capability_report()
    assert report["vendor_name_normalized"] == "nvidia"
    assert report["optional_fast_paths"]["nvidia_coopmat2"] is False
    reasons = report["optional_fast_paths_disabled_reasons"]["nvidia_coopmat2"]
    assert any("missing extension VK_NV_cooperative_matrix2" in r for r in reasons)


def test_amd_fast_path_enabled_with_coopmat_support(monkeypatch):
    caps = _base_caps()
    caps["vendor_id"] = 0x1002
    caps["extensions"] = caps["extensions"] + ["VK_KHR_cooperative_matrix"]
    caps["compile_features"]["GGML_VULKAN_COOPMAT_GLSLC_SUPPORT"] = True
    monkeypatch.setattr(vk, "_collect_vulkan_capabilities", lambda: caps)

    report = vk.get_vulkan_capability_report()
    assert report["vendor_name_normalized"] == "amd"
    assert report["optional_fast_paths"]["amd_coopmat"] is True


def test_unknown_vendor_disables_optional_fast_paths(monkeypatch):
    caps = _base_caps()
    caps["vendor_id"] = 0x9999
    caps["extensions"] = caps["extensions"] + [
        "VK_KHR_shader_integer_dot_product",
        "VK_NV_cooperative_matrix2",
        "VK_KHR_cooperative_matrix",
    ]
    caps["features"]["shaderIntegerDotProduct"] = True
    caps["compile_features"]["GGML_VULKAN_COOPMAT_GLSLC_SUPPORT"] = True
    caps["compile_features"]["GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"] = True
    caps["compile_features"]["GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"] = True
    monkeypatch.setattr(vk, "_collect_vulkan_capabilities", lambda: caps)

    report = vk.get_vulkan_capability_report()
    assert report["vendor_name_normalized"] == "unknown"
    assert report["optional_fast_paths"]["intel_integer_dot"] is False
    assert report["optional_fast_paths"]["nvidia_coopmat2"] is False
    assert report["optional_fast_paths"]["amd_coopmat"] is False
