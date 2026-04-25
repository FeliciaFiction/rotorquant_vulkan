"""
Benchmark script: Vulkan vs CUDA vs PyTorch key-kernel paths.

Usage:
    python -m turboquant.benchmark_vulkan
    python -m turboquant.benchmark_vulkan --quick
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Callable

import torch

from turboquant import cuda_backend
from turboquant import vulkan_backend
from turboquant.cuda_backend import QJLSketch
from turboquant.vulkan.reference_ops import (
    qjl_gqa_score_reference,
    qjl_quant_reference,
    qjl_score_reference,
    quantized_bmm_reference,
)


def _device_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(name: str, fn: Callable[[], None], iters: int, warmup: int):
    try:
        _device_sync()
        for _ in range(warmup):
            fn()
        _device_sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _device_sync()
        elapsed = time.perf_counter() - t0
        latency_ms = elapsed * 1000.0 / iters
        return {
            "path": name,
            "ok": True,
            "blocked": False,
            "latency_ms": latency_ms,
            "throughput_ops_s": 1000.0 / latency_ms if latency_ms > 0 else float("inf"),
            "detail": "ok",
        }
    except Exception as e:
        return {
            "path": name,
            "ok": False,
            "blocked": True,
            "latency_ms": None,
            "throughput_ops_s": None,
            "detail": str(e),
        }


def _pack_signs(signs: torch.Tensor) -> torch.Tensor:
    s = signs.shape[-1]
    bits = 8
    enc = (2 ** torch.arange(bits, dtype=torch.uint8, device=signs.device)).view(1, 1, 1, 1, 1, bits)
    b = (signs > 0).to(torch.uint8).view(*signs.shape[:-1], s // bits, bits)
    return (b * enc).sum(dim=-1, dtype=torch.uint8).contiguous()


def _naive_quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False):
    b, h, m, k = fA.shape
    feat_per_int = 32 // bits
    n = qB.shape[-1] * feat_per_int
    mask = (1 << bits) - 1
    fA2 = fA.view(-1, m, k).contiguous()
    qB2 = qB.reshape(-1, k, qB.shape[-1]).transpose(1, 2).contiguous()
    flatten_b = b * h if not mqa else b
    scales2 = scales.view(flatten_b, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
    zeros2 = zeros.view(flatten_b, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
    out = torch.zeros((b * h, m, n), dtype=torch.float32, device=fA.device)
    for bh in range(b * h):
        wb = bh if not mqa else (bh // h)
        for mi in range(m):
            for oc in range(n):
                packed_idx = oc // feat_per_int
                shift = (oc % feat_per_int) * bits
                group_idx = oc // group_size
                acc = 0.0
                for ic in range(k):
                    word = int(qB2[wb, packed_idx, ic].item())
                    qv = float((word >> shift) & mask)
                    s = float(scales2[wb, group_idx, ic].item())
                    z = float(zeros2[wb, group_idx, ic].item())
                    acc += float(fA2[bh, mi, ic].item()) * (s * qv + z)
                out[bh, mi, oc] = acc
    return out.view(b, h, m, n).to(fA.dtype).contiguous()


def _run_qjl_quant_case(device: torch.device, dtype: torch.dtype, iters: int, warmup: int):
    torch.manual_seed(1234)
    b, h, n, g, d = 1, 2, 3, 4, 128
    sketch_dim, outlier_sketch_dim, outlier_count = 64, 32, 8
    key_states = torch.randn(b, h, n, g, d, dtype=dtype, device=device)
    outlier_indices = torch.randint(0, d, (b, h, n, outlier_count), dtype=torch.int64, device=device)
    rand_prj = torch.randn(sketch_dim, d, dtype=torch.float32, device=device)

    sketch = QJLSketch(dim=(d, sketch_dim), dim_outlier=outlier_sketch_dim, device=device)
    sketch.proj_dir_quant = rand_prj.contiguous()
    mask = torch.zeros((b, h, n, d), dtype=dtype, device=device)
    mask.scatter_(-1, outlier_indices.long(), 1.0)

    rows = []
    rows.append(_bench("vulkan-reference", lambda: qjl_quant_reference(key_states, outlier_indices, rand_prj, outlier_sketch_dim), iters, warmup))
    rows.append(_bench("pytorch", lambda: sketch.quantize_pytorch(key_states, mask), iters, warmup))

    if torch.cuda.is_available() and cuda_backend.is_cuda_available() and device.type == "cuda":
        outlier_u8 = outlier_indices.to(torch.uint8)
        rows.append(_bench("cuda", lambda: cuda_backend.qjl_quant(key_states, outlier_u8, rand_prj, outlier_sketch_dim), iters, warmup))
    else:
        rows.append({"path": "cuda", "ok": False, "blocked": True, "latency_ms": None, "throughput_ops_s": None, "detail": "CUDA kernels unavailable"})
    return rows


def _run_qjl_score_case(device: torch.device, dtype: torch.dtype, iters: int, warmup: int):
    torch.manual_seed(1235)
    b, h, n, g, d = 1, 2, 3, 2, 128
    s, so, o = 64, 32, 6
    key_signs = torch.where(torch.randn(b, h, n, g, s, device=device) > 0, 1.0, -1.0)
    out_signs = torch.where(torch.randn(b, h, n, g, so, device=device) > 0, 1.0, -1.0)
    key_quant = _pack_signs(key_signs)
    key_outlier_quant = _pack_signs(out_signs)
    key_norm = (torch.rand(b, h, n, g, dtype=torch.float32, device=device) * 2.0 + 0.5).to(dtype)
    key_outlier_norm = (key_norm * 0.5).contiguous()
    outlier_indices = torch.randint(0, d, (b, h, n, o), dtype=torch.int64, device=device)
    query_states = torch.randn(b, h, d, dtype=dtype, device=device)
    rand_prj = torch.randn(d, s, dtype=torch.float32, device=device)
    query_sketch = torch.matmul(query_states.to(rand_prj.dtype), rand_prj).to(torch.float32)

    rows = []
    rows.append(_bench("vulkan-reference", lambda: qjl_score_reference(key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj), iters, warmup))
    rows.append(_bench("pytorch-naive", lambda: qjl_score_reference(key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj), iters, warmup))
    if torch.cuda.is_available() and cuda_backend.is_cuda_available() and device.type == "cuda":
        rows.append(_bench("cuda", lambda: cuda_backend.qjl_score(key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices.to(torch.uint8), query_sketch, query_states, rand_prj), iters, warmup))
    else:
        rows.append({"path": "cuda", "ok": False, "blocked": True, "latency_ms": None, "throughput_ops_s": None, "detail": "CUDA kernels unavailable"})
    return rows


def _run_qjl_gqa_score_case(device: torch.device, dtype: torch.dtype, iters: int, warmup: int):
    torch.manual_seed(1236)
    b, kh, qh, n, g, d = 1, 2, 4, 2, 2, 128
    s, so, o = 64, 32, 5
    key_signs = torch.where(torch.randn(b, kh, n, g, s, device=device) > 0, 1.0, -1.0)
    out_signs = torch.where(torch.randn(b, kh, n, g, so, device=device) > 0, 1.0, -1.0)
    key_quant = _pack_signs(key_signs)
    key_outlier_quant = _pack_signs(out_signs)
    key_norm = (torch.rand(b, kh, n, g, dtype=torch.float32, device=device) * 2.0 + 0.5).to(dtype)
    key_outlier_norm = (key_norm * 0.5).contiguous()
    outlier_indices = torch.randint(0, d, (b, kh, n, o), dtype=torch.int64, device=device)
    query_states = torch.randn(b, qh, d, dtype=dtype, device=device)
    rand_prj = torch.randn(d, s, dtype=torch.float32, device=device)
    query_sketch = torch.matmul(query_states.to(rand_prj.dtype), rand_prj).to(torch.float32)

    rows = []
    rows.append(_bench("vulkan-reference", lambda: qjl_gqa_score_reference(key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj), iters, warmup))
    rows.append(_bench("pytorch-naive", lambda: qjl_gqa_score_reference(key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj), iters, warmup))
    if torch.cuda.is_available() and cuda_backend.is_cuda_available() and device.type == "cuda":
        rows.append(_bench("cuda", lambda: cuda_backend.qjl_gqa_score(key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices.to(torch.uint8), query_sketch, query_states, rand_prj), iters, warmup))
    else:
        rows.append({"path": "cuda", "ok": False, "blocked": True, "latency_ms": None, "throughput_ops_s": None, "detail": "CUDA kernels unavailable"})
    return rows


def _run_quantized_bmm_case(device: torch.device, dtype: torch.dtype, iters: int, warmup: int):
    torch.manual_seed(1237)
    bits, group_size = 4, 16
    b, h, m, k, n = 1, 2, 3, 16, 32
    pack_factor = 32 // bits
    n_packed = n // pack_factor
    n_groups = n // group_size

    fA = torch.randn(b, h, m, k, dtype=dtype, device=device)
    qB = torch.randint(0, 2 ** 31 - 1, (b, h, k, n_packed), dtype=torch.int32, device=device)
    scales = (torch.randn(b, h, k, n_groups, dtype=torch.float32, device=device) * 0.2 + 0.05).to(dtype)
    zeros = (torch.randn(b, h, k, n_groups, dtype=torch.float32, device=device) * 0.1).to(dtype)

    rows = []
    rows.append(_bench("vulkan-reference", lambda: quantized_bmm_reference(group_size, fA, qB, scales, zeros, bits, mqa=False), iters, warmup))
    rows.append(_bench("pytorch-naive", lambda: _naive_quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False), max(1, iters // 2), max(1, warmup // 2)))
    if torch.cuda.is_available() and cuda_backend.is_cuda_available() and device.type == "cuda":
        rows.append(_bench("cuda", lambda: cuda_backend.quantized_bmm(group_size, fA, qB, scales, zeros, bits, mqa=False), iters, warmup))
    else:
        rows.append({"path": "cuda", "ok": False, "blocked": True, "latency_ms": None, "throughput_ops_s": None, "detail": "CUDA kernels unavailable"})
    return rows


def _print_metadata(device: torch.device):
    cap = vulkan_backend.get_vulkan_capability_report()
    print("=" * 78)
    print("Vulkan Benchmark Metadata")
    print("=" * 78)
    print(f"torch version: {torch.__version__}")
    print(f"device: {device}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda kernels available: {cuda_backend.is_cuda_available()}")
    print(f"vulkan strict available: {vulkan_backend.is_vulkan_available()}")
    print(f"vulkan runtime available: {cap.get('checklist', {}).get('runtime_available', False)}")
    print(f"vulkan vendor: {cap.get('vendor_name_normalized', 'unknown')}")
    print(f"vulkan device name: {cap.get('device_name')}")
    print(f"vulkan api: {cap.get('api_version_parsed')}")
    print()


def _print_rows(op_name: str, rows: list[dict]):
    print(f"[{op_name}]")
    print(f"{'path':<20} {'latency_ms':>12} {'throughput_ops/s':>18} {'status':>10} detail")
    for row in rows:
        if row["ok"]:
            print(f"{row['path']:<20} {row['latency_ms']:>12.4f} {row['throughput_ops_s']:>18.2f} {'ok':>10} {row['detail']}")
        else:
            print(f"{row['path']:<20} {'-':>12} {'-':>18} {'blocked':>10} {row['detail']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Vulkan/CUDA/PyTorch kernel paths with fixed seeds")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Use very small benchmark loops")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    if args.quick:
        args.iters = 5
        args.warmup = 2

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _print_metadata(device)

    rows_quant = _run_qjl_quant_case(device, dtype, args.iters, args.warmup)
    rows_score = _run_qjl_score_case(device, dtype, args.iters, args.warmup)
    rows_gqa = _run_qjl_gqa_score_case(device, dtype, args.iters, args.warmup)
    rows_bmm = _run_quantized_bmm_case(device, dtype, args.iters, args.warmup)

    print("=" * 78)
    print("Benchmark Results")
    print("=" * 78)
    _print_rows("qjl_quant", rows_quant)
    _print_rows("qjl_score", rows_score)
    _print_rows("qjl_gqa_score", rows_gqa)
    _print_rows("quantized_bmm", rows_bmm)


if __name__ == "__main__":
    main()
