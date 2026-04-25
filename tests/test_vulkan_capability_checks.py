import turboquant.vulkan_backend as vk


def test_capability_report_fails_when_runtime_unavailable(monkeypatch):
    monkeypatch.setattr(
        vk,
        "_collect_vulkan_capabilities",
        lambda: {
            "runtime_available": False,
            "api_version": "1.3.0",
            "extensions": ["VK_KHR_storage_buffer_storage_class"],
            "features": {"computeShader": True},
            "compile_features": {},
        },
    )
    report = vk.get_vulkan_capability_report()
    assert not report["strictly_available"]
    assert report["checklist"]["runtime_available"] is False


def test_capability_report_fails_on_vulkan_version(monkeypatch):
    monkeypatch.setattr(
        vk,
        "_collect_vulkan_capabilities",
        lambda: {
            "runtime_available": True,
            "api_version": "1.2.203",
            "extensions": ["VK_KHR_storage_buffer_storage_class"],
            "features": {"computeShader": True},
            "compile_features": {},
        },
    )
    report = vk.get_vulkan_capability_report()
    assert not report["strictly_available"]
    assert report["checklist"]["vulkan_1_3_minimum"] is False


def test_capability_report_fails_on_missing_extension(monkeypatch):
    monkeypatch.setattr(
        vk,
        "_collect_vulkan_capabilities",
        lambda: {
            "runtime_available": True,
            "api_version": "1.3.290",
            "extensions": [],
            "features": {"computeShader": True},
            "compile_features": {},
        },
    )
    report = vk.get_vulkan_capability_report()
    assert not report["strictly_available"]
    assert report["checklist"]["required_extensions"] is False
    assert "VK_KHR_storage_buffer_storage_class" in report["missing_extensions"]


def test_capability_report_fails_on_missing_feature(monkeypatch):
    monkeypatch.setattr(
        vk,
        "_collect_vulkan_capabilities",
        lambda: {
            "runtime_available": True,
            "api_version": "1.3.290",
            "extensions": ["VK_KHR_storage_buffer_storage_class"],
            "features": {"computeShader": False},
            "compile_features": {},
        },
    )
    report = vk.get_vulkan_capability_report()
    assert not report["strictly_available"]
    assert report["checklist"]["required_features"] is False
    assert "computeShader" in report["missing_features"]


def test_capability_report_passes_with_required_capabilities(monkeypatch):
    monkeypatch.setattr(
        vk,
        "_collect_vulkan_capabilities",
        lambda: {
            "runtime_available": True,
            "api_version": "1.3.290",
            "extensions": ["VK_KHR_storage_buffer_storage_class"],
            "features": {"computeShader": True},
            "compile_features": {
                "GGML_VULKAN_COOPMAT_GLSLC_SUPPORT": True,
                "GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT": True,
            },
        },
    )
    report = vk.get_vulkan_capability_report()
    assert report["strictly_available"]
    assert vk.is_vulkan_available()


def test_require_vulkan_ready_reports_clear_fallback(monkeypatch):
    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(
        vk,
        "_collect_vulkan_capabilities",
        lambda: {
            "runtime_available": True,
            "api_version": "1.2.100",
            "extensions": [],
            "features": {},
            "compile_features": {},
        },
    )
    try:
        vk._require_vulkan_ready("qjl_score")
    except RuntimeError as e:
        msg = str(e)
        assert "Vulkan capability checks failed for qjl_score" in msg
        assert "Fallback backend should be selected" in msg
    else:
        raise AssertionError("Expected RuntimeError for missing Vulkan capabilities")
