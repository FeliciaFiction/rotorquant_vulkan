import turboquant as tq
import turboquant.vulkan_backend as vk


def test_fallback_missing_vulkan_runtime_prefers_cuda(monkeypatch):
    monkeypatch.setattr(tq, "is_vulkan_available", lambda: False)
    monkeypatch.setattr(tq, "is_cuda_available", lambda: True)

    backend, reason = tq.select_backend(request_vulkan=True)
    assert backend == "cuda"
    assert "falling back to CUDA" in reason


def test_fallback_missing_vulkan_runtime_to_pytorch_when_cuda_missing(monkeypatch):
    monkeypatch.setattr(tq, "is_vulkan_available", lambda: False)
    monkeypatch.setattr(tq, "is_cuda_available", lambda: False)

    backend, reason = tq.select_backend(request_vulkan=True)
    assert backend == "pytorch"
    assert "falling back to PyTorch" in reason


def test_missing_extensions_runtime_error_is_actionable(monkeypatch):
    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(
        vk,
        "get_vulkan_capability_report",
        lambda: {
            "strictly_available": False,
            "checklist": {
                "runtime_available": True,
                "vulkan_1_3_minimum": True,
                "required_extensions": False,
                "required_features": True,
            },
            "api_version_parsed": (1, 3, 290),
            "missing_extensions": ["VK_KHR_storage_buffer_storage_class"],
            "missing_features": [],
        },
    )

    try:
        vk._require_vulkan_ready("qjl_quant")
    except RuntimeError as e:
        msg = str(e)
        assert "missing extensions" in msg
        assert "Fallback backend should be selected" in msg
    else:
        raise AssertionError("Expected RuntimeError for missing extension fallback path")


def test_invalid_shader_artifacts_smoke_does_not_crash(monkeypatch):
    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(
        vk,
        "get_vulkan_capability_report",
        lambda: {
            "strictly_available": False,
            "checklist": {"runtime_available": False},
            "vendor_name_normalized": "unknown",
            "device_name": None,
        },
    )
    monkeypatch.setattr(
        vk,
        "_shader_status_report",
        lambda: {
            "ok": False,
            "shader_dir": "mock-shaders",
            "spv_dir": "mock-spv",
            "found_sources": ["qjl_quant.comp"],
            "missing_sources": ["qjl_score.comp"],
            "found_spv": [],
            "missing_spv": [
                "qjl_quant.comp.spv",
                "qjl_score.comp.spv",
            ],
            "detail": "missing shader sources or SPIR-V artifacts; rerun `python setup.py --vulkan build_ext --inplace`",
        },
    )

    report = vk.run_vulkan_smoke_checks()
    assert report["overall_ok"] is False
    assert report["shader_artifacts"]["ok"] is False
    assert "rerun `python setup.py --vulkan build_ext --inplace`" in report["shader_artifacts"]["detail"]
    # Regression guard: fallback path should remain reachable and non-crashing.
    assert report["single_pass_probe"]["status"] == "blocked"


def test_scenario_matrix_backend_selected(monkeypatch):
    scenarios = [
        # scenario, vulkan_available, cuda_available, expected_backend
        ("missing runtime", False, True, "cuda"),
        ("missing runtime, no cuda", False, False, "pytorch"),
        ("missing extension", False, True, "cuda"),
    ]
    for _, vk_ok, cuda_ok, expected in scenarios:
        monkeypatch.setattr(tq, "is_vulkan_available", lambda v=vk_ok: v)
        monkeypatch.setattr(tq, "is_cuda_available", lambda c=cuda_ok: c)
        backend, _ = tq.select_backend(request_vulkan=True)
        assert backend == expected
