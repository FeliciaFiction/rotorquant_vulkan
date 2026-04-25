from turboquant.benchmark_vulkan import evaluate_acceptance_thresholds


def _ok_row(path, latency):
    return {
        "path": path,
        "ok": True,
        "blocked": False,
        "latency_ms": float(latency),
        "throughput_ops_s": 1000.0 / float(latency),
        "detail": "ok",
    }


def _blocked_row(path, detail="blocked"):
    return {
        "path": path,
        "ok": False,
        "blocked": True,
        "latency_ms": None,
        "throughput_ops_s": None,
        "detail": detail,
    }


def test_thresholds_blocked_when_cuda_missing():
    results = {
        "qjl_quant": [_ok_row("vulkan-reference", 1.0), _ok_row("pytorch", 1.0), _blocked_row("cuda")],
        "qjl_score": [_ok_row("vulkan-reference", 1.2), _ok_row("pytorch-naive", 1.0), _blocked_row("cuda")],
        "qjl_gqa_score": [_ok_row("vulkan-reference", 1.0), _ok_row("pytorch-naive", 1.1), _blocked_row("cuda")],
        "quantized_bmm": [_ok_row("vulkan-reference", 1.0), _ok_row("pytorch-naive", 1.0), _blocked_row("cuda")],
    }
    report = evaluate_acceptance_thresholds(results)
    assert report["overall_status"] == "blocked"
    for op in results.keys():
        assert report["per_op"][op]["vs_pytorch"]["status"] == "pass"
        assert report["per_op"][op]["vs_cuda"]["status"] == "blocked"


def test_thresholds_fail_when_vulkan_too_slow_vs_pytorch():
    results = {
        "qjl_quant": [_ok_row("vulkan-reference", 3.0), _ok_row("pytorch", 1.0), _ok_row("cuda", 1.0)],
    }
    report = evaluate_acceptance_thresholds(results)
    assert report["overall_status"] == "fail"
    assert report["per_op"]["qjl_quant"]["vs_pytorch"]["status"] == "fail"


def test_thresholds_pass_when_all_comparisons_within_limits():
    results = {
        "qjl_quant": [_ok_row("vulkan-reference", 1.0), _ok_row("pytorch", 1.0), _ok_row("cuda", 0.8)],
        "qjl_score": [_ok_row("vulkan-reference", 1.5), _ok_row("pytorch-naive", 1.0), _ok_row("cuda", 0.5)],
        "qjl_gqa_score": [_ok_row("vulkan-reference", 1.8), _ok_row("pytorch-naive", 1.0), _ok_row("cuda", 0.6)],
        "quantized_bmm": [_ok_row("vulkan-reference", 1.2), _ok_row("pytorch-naive", 1.0), _ok_row("cuda", 0.7)],
    }
    report = evaluate_acceptance_thresholds(results)
    assert report["overall_status"] == "pass"
