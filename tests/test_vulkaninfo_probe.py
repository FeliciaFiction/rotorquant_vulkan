import turboquant.vulkan_backend as vk


def test_parse_vulkaninfo_summary_and_choose_preferred_gpu():
    summary = """
    GPU0:
        apiVersion         = 1.3.292
        vendorID           = 0x1002
        deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
        deviceName         = AMD Radeon(TM) Graphics
    GPU1:
        apiVersion         = 1.4.344
        vendorID           = 0x8086
        deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        deviceName         = Intel(R) Arc(TM) Pro B60 Graphics
    """
    gpus = vk._parse_vulkaninfo_summary(summary)
    assert len(gpus) == 2
    preferred = vk._choose_preferred_gpu(gpus)
    assert preferred is not None
    assert preferred.get("index") == 1
    assert "Arc" in preferred.get("deviceName", "")


def test_collect_caps_uses_vulkaninfo_fallback_when_runtime_scaffold(monkeypatch):
    class _Ext:
        def is_vulkan_runtime_available(self):
            return False

        def vulkan_runtime_info(self):
            return "vulkan-runtime-scaffold"

        def vulkan_compile_features(self):
            return {"GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT": True}

    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(vk, "_vulkan_ext", _Ext())
    monkeypatch.setattr(
        vk,
        "_collect_vulkan_capabilities_from_vulkaninfo",
        lambda: {
            "runtime_available": True,
            "runtime_info_raw": {"source": "vulkaninfo"},
            "api_version": "1.4.344",
            "vendor_id": "0x8086",
            "vendor_name": "Intel(R) Arc(TM) Pro B60 Graphics",
            "device_name": "Intel(R) Arc(TM) Pro B60 Graphics",
            "extensions": ["VK_KHR_storage_buffer_storage_class"],
            "features": {"computeShader": True, "shaderIntegerDotProduct": True},
        },
    )

    caps = vk._collect_vulkan_capabilities()
    assert caps["runtime_available"] is True
    assert caps["device_name"] == "Intel(R) Arc(TM) Pro B60 Graphics"
    assert "VK_KHR_storage_buffer_storage_class" in caps["extensions"]
    assert caps["features"]["computeShader"] is True
    assert caps["compile_features"]["GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"] is True

