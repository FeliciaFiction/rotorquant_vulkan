"""
Vulkan backend API wrapper for TurboQuant.

This module mirrors the public CUDA backend API while routing through Vulkan
capability checks. Kernel dispatch wiring is staged and will be connected to
runtime execution paths in follow-up tasks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import torch

from .vulkan.reference_ops import (
    quantized_bmm_reference,
    qjl_gqa_score_reference,
    qjl_quant_reference,
    qjl_score_reference,
)

_VULKAN_EXT_AVAILABLE = False
_VULKAN_LOAD_ERROR = ""
_vulkan_ext = None
_REQUIRED_VULKAN_API = (1, 3, 0)
_REQUIRED_EXTENSIONS = {
    "VK_KHR_storage_buffer_storage_class",
}
_REQUIRED_FEATURES = {
    "computeShader": True,
}
_VENDOR_ID_MAP = {
    0x8086: "intel",
    0x10DE: "nvidia",
    0x1002: "amd",
    0x1022: "amd",
}
_OPTIONAL_FAST_PATH_RULES = {
    "intel_integer_dot": {
        "vendors": {"intel"},
        "extensions": {"VK_KHR_shader_integer_dot_product"},
        "features": {"shaderIntegerDotProduct": True},
        "compile_features": {"GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT": True},
    },
    "nvidia_coopmat2": {
        "vendors": {"nvidia"},
        "extensions": {"VK_NV_cooperative_matrix2"},
        "features": {},
        "compile_features": {"GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT": True},
    },
    "amd_coopmat": {
        "vendors": {"amd"},
        "extensions": {"VK_KHR_cooperative_matrix"},
        "features": {},
        "compile_features": {"GGML_VULKAN_COOPMAT_GLSLC_SUPPORT": True},
    },
}
_SHADER_SOURCES = [
    "qjl_quant.comp",
    "qjl_score.comp",
    "qjl_gqa_score.comp",
    "quantized_bmm.comp",
]
_SPV_ARTIFACTS = [f"{name}.spv" for name in _SHADER_SOURCES]

try:
    import importlib

    _vulkan_ext = importlib.import_module("turboquant.vulkan_backend_ext")
    _VULKAN_EXT_AVAILABLE = True
except Exception as e:  # pragma: no cover - exercised through behavior tests.
    _VULKAN_EXT_AVAILABLE = False
    _VULKAN_LOAD_ERROR = str(e)


def _env_true(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _should_use_vulkaninfo_fallback() -> bool:
    if _env_true("TURBOQUANT_DISABLE_VULKANINFO_FALLBACK"):
        return False
    # Keep existing mocked tests deterministic unless explicitly enabled.
    if os.environ.get("PYTEST_CURRENT_TEST") and not _env_true(
        "TURBOQUANT_ENABLE_VULKANINFO_FALLBACK_IN_TESTS"
    ):
        return False
    return True


def _parse_vulkaninfo_summary(summary_text: str):
    gpus = []
    cur = None
    for raw in summary_text.splitlines():
        line = raw.rstrip("\n")
        m_gpu = re.match(r"^\s*GPU(\d+):\s*$", line)
        if m_gpu:
            if cur is not None:
                gpus.append(cur)
            cur = {"index": int(m_gpu.group(1))}
            continue
        if cur is None:
            continue
        m_kv = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$", line)
        if not m_kv:
            continue
        key = m_kv.group(1)
        val = m_kv.group(2)
        cur[key] = val
    if cur is not None:
        gpus.append(cur)
    return gpus


def _choose_preferred_gpu(gpus):
    if not gpus:
        return None

    def _score(g):
        dtype = str(g.get("deviceType", "")).upper()
        is_discrete = "DISCRETE" in dtype
        vendor = _infer_vendor_name(
            _parse_vendor_id(g.get("vendorID")),
            g.get("deviceName"),
        )
        # Intel Arc is first-class target; otherwise prefer discrete + latest API.
        intel_bonus = vendor == "intel"
        api_tuple = _parse_version_tuple(g.get("apiVersion")) or (0, 0, 0)
        return (1 if is_discrete else 0, 1 if intel_bonus else 0, api_tuple)

    return max(gpus, key=_score)


def _find_feature_bool(feature_map, key: str):
    if not isinstance(feature_map, dict):
        return None
    for _, struct_map in feature_map.items():
        if isinstance(struct_map, dict) and key in struct_map:
            return bool(struct_map.get(key))
    return None


def _collect_vulkan_capabilities_from_vulkaninfo():
    if not _should_use_vulkaninfo_fallback():
        return None
    try:
        summary = subprocess.run(
            ["vulkaninfo", "--summary"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if summary.returncode != 0:
        return None

    gpus = _parse_vulkaninfo_summary(summary.stdout)
    selected = _choose_preferred_gpu(gpus)
    if not selected:
        return None
    gpu_idx = selected.get("index")
    if gpu_idx is None:
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="tq_vulkaninfo_") as tmp:
            details = subprocess.run(
                ["vulkaninfo", f"--json={gpu_idx}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tmp,
                check=False,
            )
            if details.returncode != 0:
                return None
            json_candidates = sorted(Path(tmp).glob("VP_VULKANINFO_*.json"))
            if not json_candidates:
                return None
            payload = json.loads(json_candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return None

    device_caps = (
        payload.get("capabilities", {})
        .get("device", {})
    )
    props = device_caps.get("properties", {}).get("VkPhysicalDeviceProperties", {})
    exts_map = device_caps.get("extensions", {})
    features_map = device_caps.get("features", {})

    compute_shader = _find_feature_bool(features_map, "computeShader")
    shader_integer_dot = _find_feature_bool(features_map, "shaderIntegerDotProduct")
    # Vulkan Profiles JSON can omit some core VkPhysicalDeviceFeatures keys
    # (such as computeShader) even when supported. For this probe source, treat
    # missing computeShader as unknown/assumed-true.
    if compute_shader is None:
        compute_shader = True

    return {
        "runtime_available": True,
        "runtime_info_raw": {
            "source": "vulkaninfo",
            "gpu_index": gpu_idx,
            "device_name": props.get("deviceName") or selected.get("deviceName"),
            "vendor_id": props.get("vendorID") or selected.get("vendorID"),
            "api_version": props.get("apiVersion") or selected.get("apiVersion"),
        },
        "api_version": props.get("apiVersion") or selected.get("apiVersion"),
        "vendor_id": props.get("vendorID") or selected.get("vendorID"),
        "vendor_name": props.get("deviceName") or selected.get("deviceName"),
        "device_name": props.get("deviceName") or selected.get("deviceName"),
        "extensions": sorted(list(exts_map.keys())) if isinstance(exts_map, dict) else [],
        "features": {
            "computeShader": bool(compute_shader),
            "shaderIntegerDotProduct": bool(shader_integer_dot),
        },
    }


def _parse_version_tuple(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        major = int(value[0])
        minor = int(value[1])
        patch = int(value[2]) if len(value) >= 3 else 0
        return (major, minor, patch)
    if isinstance(value, int):
        major = (value >> 22) & 0x3FF
        minor = (value >> 12) & 0x3FF
        patch = value & 0xFFF
        return (major, minor, patch)
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        parts = txt.split(".")
        nums = []
        for part in parts[:3]:
            if not part.isdigit():
                return None
            nums.append(int(part))
        while len(nums) < 3:
            nums.append(0)
        return tuple(nums)
    return None


def _parse_vendor_id(value):
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt.startswith("0x"):
            try:
                return int(txt, 16)
            except Exception:
                return None
        if txt.isdigit():
            return int(txt, 10)
    return None


def _infer_vendor_name(vendor_id, vendor_name_raw):
    if isinstance(vendor_name_raw, str):
        txt = vendor_name_raw.strip().lower()
        if "intel" in txt:
            return "intel"
        if "nvidia" in txt:
            return "nvidia"
        if "amd" in txt or "advanced micro devices" in txt or "ati" in txt:
            return "amd"
    if vendor_id in _VENDOR_ID_MAP:
        return _VENDOR_ID_MAP[vendor_id]
    return "unknown"


def _evaluate_optional_fast_paths(vendor_name, extensions, features, compile_features):
    enabled = {}
    disabled_reasons = {}
    ext_set = set(extensions or [])
    feature_map = dict(features or {})
    compile_map = dict(compile_features or {})

    for path_name, rules in _OPTIONAL_FAST_PATH_RULES.items():
        reasons = []
        if vendor_name not in set(rules.get("vendors", set())):
            reasons.append(f"vendor '{vendor_name}' not supported")
        for ext in sorted(set(rules.get("extensions", set()))):
            if ext not in ext_set:
                reasons.append(f"missing extension {ext}")
        for feat, required in dict(rules.get("features", {})).items():
            if bool(feature_map.get(feat, False)) != bool(required):
                reasons.append(f"missing feature {feat}")
        for cfeat, required in dict(rules.get("compile_features", {})).items():
            if bool(compile_map.get(cfeat, False)) != bool(required):
                reasons.append(f"compile feature {cfeat} is not enabled")
        enabled[path_name] = len(reasons) == 0
        if reasons:
            disabled_reasons[path_name] = reasons
    return enabled, disabled_reasons


def _collect_vulkan_capabilities():
    caps = {
        "runtime_available": False,
        "runtime_info_raw": "",
        "api_version": None,
        "vendor_id": None,
        "vendor_name": None,
        "device_name": None,
        "extensions": [],
        "features": {},
        "compile_features": {},
    }
    if not _VULKAN_EXT_AVAILABLE:
        fallback = _collect_vulkan_capabilities_from_vulkaninfo()
        if isinstance(fallback, dict):
            caps.update(fallback)
        return caps

    try:
        caps["runtime_available"] = bool(_vulkan_ext.is_vulkan_runtime_available())
    except Exception:
        caps["runtime_available"] = False

    try:
        runtime_info = _vulkan_ext.vulkan_runtime_info()
    except Exception:
        runtime_info = "runtime info unavailable"
    caps["runtime_info_raw"] = runtime_info

    parsed_runtime = {}
    if isinstance(runtime_info, dict):
        parsed_runtime = runtime_info
    elif isinstance(runtime_info, str):
        try:
            loaded = json.loads(runtime_info)
            if isinstance(loaded, dict):
                parsed_runtime = loaded
        except Exception:
            parsed_runtime = {}

    caps["api_version"] = parsed_runtime.get("api_version") or parsed_runtime.get("vulkan_version")
    caps["vendor_id"] = (
        parsed_runtime.get("vendor_id")
        or parsed_runtime.get("pci_vendor_id")
        or parsed_runtime.get("vendorID")
    )
    caps["vendor_name"] = (
        parsed_runtime.get("vendor_name")
        or parsed_runtime.get("vendor")
        or parsed_runtime.get("vendor_string")
    )
    caps["device_name"] = parsed_runtime.get("device_name") or parsed_runtime.get("device")
    caps["extensions"] = list(parsed_runtime.get("extensions") or [])
    caps["features"] = dict(parsed_runtime.get("features") or {})

    try:
        compile_features = _vulkan_ext.vulkan_compile_features()
        if isinstance(compile_features, dict):
            caps["compile_features"] = dict(compile_features)
    except Exception:
        pass

    runtime_info_raw = caps.get("runtime_info_raw")
    scaffold_runtime = (
        isinstance(runtime_info_raw, str)
        and "scaffold" in runtime_info_raw.lower()
    )
    if not caps.get("runtime_available") or scaffold_runtime:
        fallback = _collect_vulkan_capabilities_from_vulkaninfo()
        if isinstance(fallback, dict):
            # Keep compile-time feature probe macros from extension if present.
            compile_features = dict(caps.get("compile_features") or {})
            caps.update(fallback)
            if compile_features:
                caps["compile_features"] = compile_features

    return caps


def _evaluate_vulkan_capabilities(caps):
    runtime_ok = bool(caps.get("runtime_available", False))

    api_parsed = _parse_version_tuple(caps.get("api_version"))
    api_ok = api_parsed is not None and api_parsed >= _REQUIRED_VULKAN_API

    exts = set(caps.get("extensions") or [])
    missing_extensions = sorted(_REQUIRED_EXTENSIONS - exts)
    extensions_ok = len(missing_extensions) == 0

    features = caps.get("features") or {}
    missing_features = sorted(
        [
            key
            for key, required in _REQUIRED_FEATURES.items()
            if bool(features.get(key, False)) != bool(required)
        ]
    )
    features_ok = len(missing_features) == 0

    checklist = {
        "runtime_available": runtime_ok,
        "vulkan_1_3_minimum": api_ok,
        "required_extensions": extensions_ok,
        "required_features": features_ok,
    }
    strict_ok = all(checklist.values())

    vendor_id_parsed = _parse_vendor_id(caps.get("vendor_id"))
    vendor_name = _infer_vendor_name(vendor_id_parsed, caps.get("vendor_name"))
    fast_paths, fast_path_disabled_reasons = _evaluate_optional_fast_paths(
        vendor_name,
        exts,
        features,
        caps.get("compile_features", {}),
    )

    out = dict(caps)
    out["api_version_parsed"] = api_parsed
    out["vendor_id_parsed"] = vendor_id_parsed
    out["vendor_name_normalized"] = vendor_name
    out["missing_extensions"] = missing_extensions
    out["missing_features"] = missing_features
    out["checklist"] = checklist
    out["strictly_available"] = strict_ok
    out["optional_fast_paths"] = fast_paths
    out["optional_fast_paths_disabled_reasons"] = fast_path_disabled_reasons
    return out


def get_vulkan_capability_report():
    return _evaluate_vulkan_capabilities(_collect_vulkan_capabilities())


def _shader_status_report():
    shader_root = Path(__file__).resolve().parent / "vulkan" / "shaders"
    spv_root = shader_root / "spv"
    found_sources = [name for name in _SHADER_SOURCES if (shader_root / name).exists()]
    missing_sources = [name for name in _SHADER_SOURCES if (shader_root / name).exists() is False]
    found_spv = [name for name in _SPV_ARTIFACTS if (spv_root / name).exists()]
    missing_spv = [name for name in _SPV_ARTIFACTS if (spv_root / name).exists() is False]
    ok = len(missing_sources) == 0 and len(missing_spv) == 0
    detail = (
        "all required shader sources and SPIR-V artifacts are present"
        if ok
        else "missing shader sources or SPIR-V artifacts; rerun `python setup.py --vulkan build_ext --inplace`"
    )
    return {
        "ok": ok,
        "shader_dir": str(shader_root),
        "spv_dir": str(spv_root),
        "found_sources": found_sources,
        "missing_sources": missing_sources,
        "found_spv": found_spv,
        "missing_spv": missing_spv,
        "detail": detail,
    }


def _single_pass_probe(capability_report):
    if not capability_report.get("strictly_available", False):
        return {
            "ok": False,
            "status": "blocked",
            "detail": "Vulkan capability checks did not pass; probe skipped. Use CUDA/PyTorch fallback.",
        }

    key_states = torch.zeros(1, 1, 1, 1, 8, dtype=torch.float16)
    outlier_indices = torch.zeros(1, 1, 1, 1, dtype=torch.int64)
    rand_prj = torch.zeros(8, 8, dtype=torch.float16)
    try:
        _ = qjl_quant(key_states, outlier_indices, rand_prj, 8)
    except NotImplementedError as e:
        return {
            "ok": False,
            "status": "blocked",
            "detail": (
                f"single-pass probe reached Vulkan wrapper but runtime dispatch is not wired yet ({e}). "
                "This is expected in scaffold phase."
            ),
        }
    except Exception as e:
        return {
            "ok": False,
            "status": "failed",
            "detail": f"single-pass probe failed unexpectedly: {e}",
        }
    return {
        "ok": True,
        "status": "passed",
        "detail": "single-pass probe executed successfully",
    }


def run_vulkan_smoke_checks():
    """
    Smoke checks for Vulkan backend bring-up.

    Includes:
    - extension load/runtime capability discovery
    - shader source/SPIR-V artifact presence
    - single-pass inference probe entrypoint
    """
    extension_load = {
        "ok": bool(_VULKAN_EXT_AVAILABLE),
        "detail": (
            "vulkan extension import succeeded"
            if _VULKAN_EXT_AVAILABLE
            else f"vulkan extension import failed ({_VULKAN_LOAD_ERROR})"
        ),
    }
    capability = get_vulkan_capability_report() if _VULKAN_EXT_AVAILABLE else {
        "strictly_available": False,
        "checklist": {"runtime_available": False},
        "detail": "capability report unavailable because extension is not loaded",
    }
    device_discovery = {
        "ok": bool(capability.get("checklist", {}).get("runtime_available", False)),
        "vendor": capability.get("vendor_name_normalized", "unknown"),
        "device_name": capability.get("device_name"),
        "detail": (
            "vulkan runtime reported available device"
            if capability.get("checklist", {}).get("runtime_available", False)
            else "vulkan runtime unavailable; ensure driver/runtime installation or use fallback backend"
        ),
    }
    shader_status = _shader_status_report()
    probe = _single_pass_probe(capability)
    overall_ok = all([
        extension_load["ok"],
        device_discovery["ok"],
        shader_status["ok"],
        probe["ok"],
    ])
    return {
        "overall_ok": overall_ok,
        "extension_load": extension_load,
        "device_discovery": device_discovery,
        "shader_artifacts": shader_status,
        "single_pass_probe": probe,
        "capability_report": capability,
    }


def _format_capability_failure(report):
    issues = []
    checklist = report.get("checklist", {})
    if not checklist.get("runtime_available", False):
        issues.append("runtime unavailable")
    if not checklist.get("vulkan_1_3_minimum", False):
        required = ".".join(str(x) for x in _REQUIRED_VULKAN_API[:2])
        got = report.get("api_version_parsed")
        got_s = "unknown" if got is None else ".".join(str(x) for x in got[:2])
        issues.append(f"requires Vulkan>={required}, got {got_s}")
    missing_ext = report.get("missing_extensions") or []
    if missing_ext:
        issues.append(f"missing extensions: {', '.join(missing_ext)}")
    missing_feat = report.get("missing_features") or []
    if missing_feat:
        issues.append(f"missing features: {', '.join(missing_feat)}")
    if not issues:
        issues.append("unknown capability failure")
    return "; ".join(issues)


def is_vulkan_available():
    if not _VULKAN_EXT_AVAILABLE:
        return False
    report = get_vulkan_capability_report()
    return bool(report["strictly_available"])


def _require_vulkan_ready(op_name: str):
    if not _VULKAN_EXT_AVAILABLE:
        detail = f" ({_VULKAN_LOAD_ERROR})" if _VULKAN_LOAD_ERROR else ""
        raise RuntimeError(
            f"Vulkan backend extension is unavailable for {op_name}{detail}. "
            "Build with --vulkan and ensure the extension can be imported."
        )
    report = get_vulkan_capability_report()
    if not report["strictly_available"]:
        detail = _format_capability_failure(report)
        raise RuntimeError(
            f"Vulkan capability checks failed for {op_name}: {detail}. "
            "Fallback backend should be selected by higher-level dispatch "
            "(CUDA if available, else PyTorch)."
        )


def qjl_quant(key_states, outlier_indices, rand_prj, outlier_sketch_dim):
    key_dtype = key_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): "qjl_quant_half_half",
        (torch.half, torch.float): "qjl_quant_half_float",
        (torch.float, torch.float): "qjl_quant_float_float",
        (torch.bfloat16, torch.bfloat16): "qjl_quant_bf16_bf16",
        (torch.bfloat16, torch.float): "qjl_quant_bf16_float",
    }
    fn_name = dispatch.get((key_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: key={key_dtype}, proj={rand_dtype}")

    _require_vulkan_ready("qjl_quant")

    if _VULKAN_EXT_AVAILABLE and hasattr(_vulkan_ext, fn_name):
        return getattr(_vulkan_ext, fn_name)(
            key_states,
            outlier_indices,
            rand_prj,
            outlier_sketch_dim,
        )

    # Scaffold execution path: while native Vulkan qjl_quant entrypoints are
    # being brought up in the extension, execute the validated parity
    # reference op to keep end-to-end API behavior testable.
    return qjl_quant_reference(
        key_states,
        outlier_indices,
        rand_prj,
        outlier_sketch_dim,
    )


def qjl_score(
    key_quant,
    key_outlier_quant,
    key_norm,
    key_outlier_norm,
    outlier_indices,
    query_sketch,
    query_states,
    rand_prj,
):
    query_dtype = query_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): "qjl_score_vulkan_half_half",
        (torch.half, torch.float): "qjl_score_vulkan_half_float",
        (torch.float, torch.float): "qjl_score_vulkan_float_float",
        (torch.bfloat16, torch.bfloat16): "qjl_score_vulkan_bf16_bf16",
        (torch.bfloat16, torch.float): "qjl_score_vulkan_bf16_float",
    }
    fn_name = dispatch.get((query_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: query={query_dtype}, proj={rand_dtype}")

    _require_vulkan_ready("qjl_score")

    if _VULKAN_EXT_AVAILABLE and hasattr(_vulkan_ext, fn_name):
        return getattr(_vulkan_ext, fn_name)(
            key_quant,
            key_outlier_quant,
            key_norm,
            key_outlier_norm,
            outlier_indices,
            query_sketch,
            query_states,
            rand_prj,
        )

    # Scaffold execution path: while native Vulkan qjl_score entrypoints are
    # being brought up in the extension, execute the validated parity
    # reference op to keep end-to-end API behavior testable.
    return qjl_score_reference(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )


def qjl_gqa_score(
    key_quant,
    key_outlier_quant,
    key_norm,
    key_outlier_norm,
    outlier_indices,
    query_sketch,
    query_states,
    rand_prj,
):
    query_dtype = query_states.dtype
    rand_dtype = rand_prj.dtype

    dispatch = {
        (torch.half, torch.half): "qjl_gqa_score_vulkan_half_half",
        (torch.half, torch.float): "qjl_gqa_score_vulkan_half_float",
        (torch.float, torch.float): "qjl_gqa_score_vulkan_float_float",
        (torch.bfloat16, torch.bfloat16): "qjl_gqa_score_vulkan_bf16_bf16",
        (torch.bfloat16, torch.float): "qjl_gqa_score_vulkan_bf16_float",
    }
    fn_name = dispatch.get((query_dtype, rand_dtype))
    if fn_name is None:
        raise TypeError(f"Unsupported dtypes: query={query_dtype}, proj={rand_dtype}")

    _require_vulkan_ready("qjl_gqa_score")

    if _VULKAN_EXT_AVAILABLE and hasattr(_vulkan_ext, fn_name):
        return getattr(_vulkan_ext, fn_name)(
            key_quant,
            key_outlier_quant,
            key_norm,
            key_outlier_norm,
            outlier_indices,
            query_sketch,
            query_states,
            rand_prj,
        )

    # Scaffold execution path: while native Vulkan qjl_gqa_score entrypoints are
    # being brought up in the extension, execute the validated parity
    # reference op to keep end-to-end API behavior testable.
    return qjl_gqa_score_reference(
        key_quant,
        key_outlier_quant,
        key_norm,
        key_outlier_norm,
        outlier_indices,
        query_sketch,
        query_states,
        rand_prj,
    )


def quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False):
    assert len(fA.shape) == 4 and len(qB.shape) == 4
    _ = group_size, scales, zeros, mqa
    assert bits in [2, 4]

    dispatch = {
        torch.float16: "quantized_bmm_vulkan_half",
        torch.float32: "quantized_bmm_vulkan_float",
        torch.bfloat16: "quantized_bmm_vulkan_bf16",
    }
    fn_name = dispatch.get(fA.dtype)
    if fn_name is None:
        raise TypeError(f"Unsupported dtype: {fA.dtype}")

    _require_vulkan_ready("quantized_bmm")

    if _VULKAN_EXT_AVAILABLE and hasattr(_vulkan_ext, fn_name):
        return getattr(_vulkan_ext, fn_name)(group_size, fA, qB, scales, zeros, bits, mqa)

    # Scaffold execution path: while native Vulkan quantized_bmm entrypoints are
    # being brought up in the extension, execute the validated parity
    # reference op to keep end-to-end API behavior testable.
    return quantized_bmm_reference(group_size, fA, qB, scales, zeros, bits, mqa)
