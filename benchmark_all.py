"""
Benchmark inference speed of Depthwise MNIST across backends:
  - PyTorch MPS
  - MLX (baseline)
  - MLX (fused Metal kernel)
"""
import time
import numpy as np
from pathlib import Path
from contextlib import contextmanager

import torch
import mlx.core as mx

from train_mnist import DepthwiseMNIST as PTModel
from mlx_model import get_model, forward_fused

BATCH_SIZES = [1, 16, 64, 128, 256, 512]
WARMUP = 30
TRIALS = 200
WEIGHTS = "output/best_model.pth"

DTYPE = mx.float32


@contextmanager
def time_block(desc):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    print(f"  {desc}: {(t1 - t0) * 1000:.3f} ms")


def benchmark_pt(batch_size):
    model = PTModel()
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    model = model.to("mps")

    x = torch.randn(batch_size, 1, 28, 28, device="mps")

    # Warmup
    for _ in range(WARMUP):
        _ = model(x)
    torch.mps.synchronize()

    # Measure
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(TRIALS):
        _ = model(x)
    torch.mps.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / TRIALS * 1000
    throughput = batch_size / (avg_ms / 1000)
    return avg_ms, throughput


def benchmark_mlx(batch_size, fused=False):
    model = get_model(WEIGHTS)

    x = mx.random.normal(shape=(batch_size, 28, 28, 1), dtype=DTYPE)

    if fused:
        fn = lambda: forward_fused(model, x)
    else:
        fn = lambda: model(x)

    # Warmup
    for _ in range(WARMUP):
        y = fn()
        mx.eval(y)

    # Measure
    t0 = time.perf_counter()
    for _ in range(TRIALS):
        y = fn()
        mx.eval(y)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / TRIALS * 1000
    throughput = batch_size / (avg_ms / 1000)
    return avg_ms, throughput


def main():
    results = {}

    # PyTorch MPS
    print("\n=== PyTorch MPS ===")
    results["pt"] = {}
    for bs in BATCH_SIZES:
        print(f"  BS={bs}...", end=" ", flush=True)
        lat, thru = benchmark_pt(bs)
        results["pt"][bs] = (lat, thru)
        print(f"{lat:.3f} ms, {thru:,.0f} samp/s")

    # MLX baseline
    print("\n=== MLX (baseline) ===")
    results["mlx"] = {}
    for bs in BATCH_SIZES:
        print(f"  BS={bs}...", end=" ", flush=True)
        lat, thru = benchmark_mlx(bs, fused=False)
        results["mlx"][bs] = (lat, thru)
        print(f"{lat:.3f} ms, {thru:,.0f} samp/s")

    # MLX fused
    print("\n=== MLX (fused kernel) ===")
    results["fused"] = {}
    for bs in BATCH_SIZES:
        print(f"  BS={bs}...", end=" ", flush=True)
        lat, thru = benchmark_mlx(bs, fused=True)
        results["fused"][bs] = (lat, thru)
        print(f"{lat:.3f} ms, {thru:,.0f} samp/s")

    # Summary table
    print("\n\n## Summary\n")
    print(f"| BS | PT MPS (ms) | PT thru | MLX (ms) | MLX thru | Fused (ms) | Fused thru |")
    print(f"|:--:|:-----------:|:-------:|:--------:|:--------:|:----------:|:----------:|")
    for bs in BATCH_SIZES:
        pt_lat, pt_thru = results["pt"][bs]
        mx_lat, mx_thru = results["mlx"][bs]
        fu_lat, fu_thru = results["fused"][bs]
        print(f"| {bs} | {pt_lat:.3f} | {pt_thru:,.0f} | {mx_lat:.3f} | {mx_thru:,.0f} | {fu_lat:.3f} | {fu_thru:,.0f} |")


if __name__ == "__main__":
    main()
