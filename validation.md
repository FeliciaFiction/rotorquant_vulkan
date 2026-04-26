# Validation Ledger

Date: 2026-04-26  
Scope: Up-to-date Vulkan backend validations, including regression coverage.

## Policy
- Every new validation run must be appended here with command, result, and date.
- Every new validation change requires rerunning:
  - Full Vulkan validation suite
  - Regression tests

## Latest Runs (2026-04-26)

| Command | Result |
|---|---|
| `python -m pytest tests/test_backend_selection_policy.py tests/test_vulkan_backend_api_compat.py tests/test_vulkan_capability_checks.py tests/test_vulkan_vendor_guardrails.py tests/test_vulkan_backend_smoke.py tests/test_vulkan_fallback_regressions.py tests/test_vulkan_qjl_quant_dispatch.py tests/test_vulkan_qjl_score_dispatch.py tests/test_vulkan_qjl_gqa_score_dispatch.py tests/test_vulkan_quantized_bmm_dispatch.py tests/test_vulkan_qjl_quant_reference.py tests/test_vulkan_qjl_score_reference.py tests/test_vulkan_qjl_gqa_score_reference.py tests/test_vulkan_quantized_bmm_reference.py tests/test_vulkan_parity_matrix.py tests/test_vulkan_benchmark_thresholds.py tests/test_vulkaninfo_probe.py -q` | `74 passed, 10 skipped in 2.99s` |
| `python -m turboquant.benchmark_vulkan --quick --dtype fp16` | Pass. Metadata detected Intel Arc Pro B60 (`vulkan strict available: True`, `runtime available: True`). CUDA path blocked (expected on this host). |
| `python -m turboquant.benchmark_vulkan --quick --dtype fp16 --check-thresholds` | Pass. Threshold table emitted; PyTorch ratios passed for all ops, CUDA ratios blocked; overall status `blocked` due unavailable CUDA path. |
| `python C:\\Git\\rotorquant\\setup.py --name` | Pass (`turboquant`) |
| `python C:\\Git\\rotorquant\\setup.py --vulkan --name` | Pass (`turboquant`) |
| `python -c "import turboquant.vulkan_backend as vk; print(vk.is_vulkan_available()); print(vk.get_vulkan_capability_report().get('device_name')); print(vk.run_vulkan_smoke_checks().get('overall_ok'))"` | Pass (`True`, `Intel(R) Arc(TM) Pro B60 Graphics`, `True`) |
| `python setup.py --vulkan build_ext --inplace` | Pass. Vulkan extension rebuilt with native `qjl_quant_*` entrypoints exported. |
| `python -m pytest tests/test_backend_selection_policy.py tests/test_vulkan_backend_api_compat.py tests/test_vulkan_capability_checks.py tests/test_vulkan_vendor_guardrails.py tests/test_vulkan_backend_smoke.py tests/test_vulkan_fallback_regressions.py tests/test_vulkan_qjl_quant_dispatch.py tests/test_vulkan_qjl_score_dispatch.py tests/test_vulkan_qjl_gqa_score_dispatch.py tests/test_vulkan_quantized_bmm_dispatch.py tests/test_vulkan_qjl_quant_reference.py tests/test_vulkan_qjl_score_reference.py tests/test_vulkan_qjl_gqa_score_reference.py tests/test_vulkan_quantized_bmm_reference.py tests/test_vulkan_parity_matrix.py tests/test_vulkan_benchmark_thresholds.py tests/test_vulkaninfo_probe.py tests/test_vulkan_qjl_quant_native_extension.py -q` | `74 passed, 11 skipped in 3.42s` |

## Regression Coverage Included

Regression scenarios are included in the full pytest command above, including:
- `tests/test_vulkan_fallback_regressions.py`
- `tests/test_vulkan_backend_smoke.py`
- Capability/guardrail regressions:
  - `tests/test_vulkan_capability_checks.py`
  - `tests/test_vulkan_vendor_guardrails.py`
  - `tests/test_vulkaninfo_probe.py`
