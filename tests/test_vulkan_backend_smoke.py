import turboquant.vulkan_backend as vk


def test_run_vulkan_smoke_checks_structure():
    report = vk.run_vulkan_smoke_checks()
    for key in (
        "overall_ok",
        "extension_load",
        "device_discovery",
        "shader_artifacts",
        "single_pass_probe",
        "capability_report",
    ):
        assert key in report
    assert "ok" in report["extension_load"]
    assert "ok" in report["device_discovery"]
    assert "ok" in report["shader_artifacts"]
    assert "status" in report["single_pass_probe"]


def test_smoke_diagnostics_actionable_when_extension_missing(monkeypatch):
    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", False)
    monkeypatch.setattr(vk, "_VULKAN_LOAD_ERROR", "mock-ext-load-failure")

    report = vk.run_vulkan_smoke_checks()
    assert report["overall_ok"] is False
    assert report["extension_load"]["ok"] is False
    assert "mock-ext-load-failure" in report["extension_load"]["detail"]
    assert "fallback backend" in report["device_discovery"]["detail"].lower()
    assert report["single_pass_probe"]["status"] == "blocked"


def test_smoke_reports_blocked_single_pass_when_dispatch_not_wired(monkeypatch):
    monkeypatch.setattr(vk, "_VULKAN_EXT_AVAILABLE", True)
    monkeypatch.setattr(
        vk,
        "get_vulkan_capability_report",
        lambda: {
            "strictly_available": True,
            "checklist": {"runtime_available": True},
            "vendor_name_normalized": "intel",
            "device_name": "mock-intel-device",
        },
    )
    monkeypatch.setattr(
        vk,
        "_shader_status_report",
        lambda: {
            "ok": True,
            "detail": "all required shader sources and SPIR-V artifacts are present",
            "found_sources": [],
            "missing_sources": [],
            "found_spv": [],
            "missing_spv": [],
            "shader_dir": "",
            "spv_dir": "",
        },
    )
    monkeypatch.setattr(vk, "qjl_quant", lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("not wired")))

    report = vk.run_vulkan_smoke_checks()
    assert report["device_discovery"]["ok"] is True
    assert report["single_pass_probe"]["status"] == "blocked"
    assert "not wired" in report["single_pass_probe"]["detail"].lower()
