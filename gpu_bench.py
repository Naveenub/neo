"""Phase 0 — GPU Benchmark.

Run: python scripts/gpu_bench.py
Prints measured TFLOPS/bandwidth against this GPU's spec sheet peak.
"""
import statistics
import sys
import time

import torch

MATMUL_DIM = 4096
MATMUL_ITERS = 100
BANDWIDTH_BYTES = 1 << 30  # 1GB transfer for bandwidth probe

# Peak bf16 TFLOPS by GPU name substring (spec-sheet values, used as the
# expected baseline for print_bench_report's pass/fail threshold).
GPU_SPEC_TFLOPS = {
    "H100": 989.0,
    "A100": 312.0,
    "4090": 165.0,
    "3090": 71.0,
    "A6000": 155.0,
}


def _device_check() -> None:
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device available")
        sys.exit(1)


def bench_matmul_tflops() -> float:
    _device_check()
    device = torch.device("cuda")
    a = torch.randn(MATMUL_DIM, MATMUL_DIM, dtype=torch.bfloat16, device=device)
    b = torch.randn(MATMUL_DIM, MATMUL_DIM, dtype=torch.bfloat16, device=device)

    for _ in range(10):  # warmup
        torch.matmul(a, b)
    torch.cuda.synchronize()

    flops_per_matmul = 2 * (MATMUL_DIM ** 3)
    times = []
    for _ in range(MATMUL_ITERS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    median_s = statistics.median(times)
    tflops = flops_per_matmul / median_s / 1e12
    return tflops


def bench_memory_bandwidth() -> float:
    _device_check()
    device = torch.device("cuda")
    n_elems = BANDWIDTH_BYTES // 4  # float32
    src = torch.randn(n_elems, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)

    for _ in range(5):  # warmup
        dst.copy_(src)
    torch.cuda.synchronize()

    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_moved = BANDWIDTH_BYTES * 2 * iters  # read + write
    gb_per_s = bytes_moved / elapsed / 1e9
    return gb_per_s


def bench_tokens_per_second(model_size_b: float) -> float:
    """Rough inference tok/s estimate from measured bandwidth (memory-bound
    decode: 2 bytes/param in bf16 must move from HBM per token)."""
    bandwidth_gb_s = bench_memory_bandwidth()
    bytes_per_token = model_size_b * 1e9 * 2
    tok_s = (bandwidth_gb_s * 1e9) / bytes_per_token
    return tok_s


def _expected_tflops() -> float | None:
    name = torch.cuda.get_device_name(0)
    for key, val in GPU_SPEC_TFLOPS.items():
        if key in name:
            return val
    return None


def print_bench_report() -> None:
    _device_check()
    device_name = torch.cuda.get_device_name(0)
    print(f"=== NEO PHASE 0 — GPU BENCHMARK ({device_name}) ===\n")

    measured_tflops = bench_matmul_tflops()
    bandwidth = bench_memory_bandwidth()
    tok_s_7b = bench_tokens_per_second(7.0)

    expected = _expected_tflops()
    print(f"  Matmul (bf16, {MATMUL_DIM}x{MATMUL_DIM}): {measured_tflops:.1f} TFLOPS median")
    if expected is not None:
        pct = measured_tflops / expected * 100
        print(f"    Spec peak: {expected:.1f} TFLOPS  ->  {pct:.1f}% of peak")
    else:
        print(f"    Spec peak: unknown for '{device_name}' — add to GPU_SPEC_TFLOPS to enable pass/fail")
    print(f"  HBM bandwidth: {bandwidth:.1f} GB/s")
    print(f"  Est. tok/s @ 7B params (bf16, memory-bound): {tok_s_7b:.1f}")
    print()

    if expected is None:
        print("SKIP: unknown GPU spec, cannot compute pass/fail threshold")
        sys.exit(1)
    if measured_tflops >= 0.60 * expected:
        print(f"PASS: {measured_tflops:.1f} TFLOPS >= 60% of {expected:.1f} TFLOPS spec")
    else:
        print(f"FAIL: {measured_tflops:.1f} TFLOPS < 60% of {expected:.1f} TFLOPS spec")
        sys.exit(1)


if __name__ == "__main__":
    print_bench_report()
