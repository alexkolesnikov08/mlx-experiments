"""
Benchmark inference speed of Depthwise MNIST across backends:
  - PyTorch MPS
  - MLX (baseline)
  - MLX (fused Metal kernel)
"""
import time
import numpy as np
from pathlib import Path

import torch
import mlx.core as mx

from train_mnist import DepthwiseMNIST as PTModel
from mlx_model import get_model, forward_fused

# Theoretical FLOPs per forward pass (one example)
# Conv1: 2 * 3*3 * 1*48 * 28*28 = 677,376
# Stage 0: DW(2*3*3*48*14*14=169,344) + PW(2*1*1*48*96*14*14=1,806,336) + Skip(1,806,336) = 3,782,016
# Stage 1: DW(2*3*3*96*14*14=338,688) + PW(2*1*1*96*96*14*14=3,612,672) = 3,951,360
# Stage 2: DW(2*3*3*96*7*7=84,672) + PW(2*1*1*96*192*7*7=1,806,336) + Skip(1,806,336) = 3,697,344
# Stage 3: DW(2*3*3*192*7*7=169,344) + PW(2*1*1*192*192*7*7=3,612,672) = 3,782,016
# FC: 2 * 192*10 = 3,840
# Total: ~15.9M FLOPs + element-wise ~1.9M ≈ 17.8M FLOPs
FLOPS_PER_PASS = 17_800_000  # approximate for this model

BATCH_SIZES = [1, 16, 64, 128, 256, 512]
WARMUP = 30
TRIALS = 200
WEIGHTS = "output/best_model.pth"

DTYPE = mx.float32


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

    # Measure per-trial
    torch.mps.synchronize()
    lats = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        _ = model(x)
        torch.mps.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)

    lats = np.array(lats)
    p50 = np.median(lats)
    p95 = np.percentile(lats, 95)
    mean_ms = np.mean(lats)
    cv = np.std(lats) / mean_ms
    if cv > 0.10:
        print(f"  ⚠ CV={cv:.2%} > 10%, возможен throttle")
    throughput = batch_size / (p50 / 1000)
    return p50, p95, mean_ms, cv, throughput, lats


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

    # Measure per-trial
    lats = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        lats.append((time.perf_counter() - t0) * 1000)

    lats = np.array(lats)
    p50 = np.median(lats)
    p95 = np.percentile(lats, 95)
    mean_ms = np.mean(lats)
    cv = np.std(lats) / mean_ms
    if cv > 0.10:
        print(f"  ⚠ CV={cv:.2%} > 10%, возможен throttle")
    throughput = batch_size / (p50 / 1000)
    return p50, p95, mean_ms, cv, throughput, lats


def main():
    results = {}

    # PyTorch MPS
    print("\n=== PyTorch MPS ===")
    results["pt"] = {}
    for bs in BATCH_SIZES:
        print(f"  BS={bs}...", end=" ", flush=True)
        p50, p95, mean_ms, cv, thru, lats = benchmark_pt(bs)
        results["pt"][bs] = (p50, p95, mean_ms, cv, thru, lats)
        print(f"P50={p50:.3f} ms, P95={p95:.3f} ms, thru={thru:,.0f} samp/s, CV={cv:.2%}")

    # MLX baseline
    print("\n=== MLX (baseline) ===")
    results["mlx"] = {}
    for bs in BATCH_SIZES:
        print(f"  BS={bs}...", end=" ", flush=True)
        p50, p95, mean_ms, cv, thru, lats = benchmark_mlx(bs, fused=False)
        results["mlx"][bs] = (p50, p95, mean_ms, cv, thru, lats)
        print(f"P50={p50:.3f} ms, P95={p95:.3f} ms, thru={thru:,.0f} samp/s, CV={cv:.2%}")

    # MLX fused
    print("\n=== MLX (fused kernel) ===")
    results["fused"] = {}
    for bs in BATCH_SIZES:
        print(f"  BS={bs}...", end=" ", flush=True)
        p50, p95, mean_ms, cv, thru, lats = benchmark_mlx(bs, fused=True)
        results["fused"][bs] = (p50, p95, mean_ms, cv, thru, lats)
        print(f"P50={p50:.3f} ms, P95={p95:.3f} ms, thru={thru:,.0f} samp/s, CV={cv:.2%}")

    # Summary table
    print("\n\n## Summary (P50 latency)\n")
    print(f"| BS | PT MPS (ms) | PT thru | MLX (ms) | MLX thru | Fused (ms) | Fused thru |")
    print(f"|:--:|:-----------:|:-------:|:--------:|:--------:|:----------:|:----------:|")
    for bs in BATCH_SIZES:
        pt_p50, pt_p95, pt_mean, pt_cv, pt_thru, _ = results["pt"][bs]
        mx_p50, mx_p95, mx_mean, mx_cv, mx_thru, _ = results["mlx"][bs]
        fu_p50, fu_p95, fu_mean, fu_cv, fu_thru, _ = results["fused"][bs]
        print(f"| {bs} | {pt_p50:.3f} | {pt_thru:,.0f} | {mx_p50:.3f} | {mx_thru:,.0f} | {fu_p50:.3f} | {fu_thru:,.0f} |")

    print("\n## CV check\n")
    for bs in BATCH_SIZES:
        _, _, _, pt_cv, _, _ = results["pt"][bs]
        _, _, _, mx_cv, _, _ = results["mlx"][bs]
        _, _, _, fu_cv, _, _ = results["fused"][bs]
        flags = []
        if pt_cv > 0.10:
            flags.append(f"PT CV={pt_cv:.2%} ⚠")
        if mx_cv > 0.10:
            flags.append(f"MLX CV={mx_cv:.2%} ⚠")
        if fu_cv > 0.10:
            flags.append(f"Fused CV={fu_cv:.2%} ⚠")
        if flags:
            print(f"  BS={bs}: " + " | ".join(flags))

    # Save ALL raw and aggregate data to NPZ
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    np.savez(out_dir / "per_trial.npz",
        batch_sizes=np.array(BATCH_SIZES),
        # raw latencies: shape (len(BATCH_SIZES), TRIALS) each
        pt_lats=np.array([results["pt"][bs][5] for bs in BATCH_SIZES]),
        mlx_lats=np.array([results["mlx"][bs][5] for bs in BATCH_SIZES]),
        fused_lats=np.array([results["fused"][bs][5] for bs in BATCH_SIZES]),
        # aggregates: shape (len(BATCH_SIZES),) each
        pt_p50=np.array([results["pt"][bs][0] for bs in BATCH_SIZES]),
        pt_p95=np.array([results["pt"][bs][1] for bs in BATCH_SIZES]),
        pt_mean=np.array([results["pt"][bs][2] for bs in BATCH_SIZES]),
        pt_cv=np.array([results["pt"][bs][3] for bs in BATCH_SIZES]),
        pt_throughput=np.array([results["pt"][bs][4] for bs in BATCH_SIZES]),
        mlx_p50=np.array([results["mlx"][bs][0] for bs in BATCH_SIZES]),
        mlx_p95=np.array([results["mlx"][bs][1] for bs in BATCH_SIZES]),
        mlx_mean=np.array([results["mlx"][bs][2] for bs in BATCH_SIZES]),
        mlx_cv=np.array([results["mlx"][bs][3] for bs in BATCH_SIZES]),
        mlx_throughput=np.array([results["mlx"][bs][4] for bs in BATCH_SIZES]),
        fused_p50=np.array([results["fused"][bs][0] for bs in BATCH_SIZES]),
        fused_p95=np.array([results["fused"][bs][1] for bs in BATCH_SIZES]),
        fused_mean=np.array([results["fused"][bs][2] for bs in BATCH_SIZES]),
        fused_cv=np.array([results["fused"][bs][3] for bs in BATCH_SIZES]),
        fused_throughput=np.array([results["fused"][bs][4] for bs in BATCH_SIZES]),
        metadata_warmup=np.array(WARMUP),
        metadata_trials=np.array(TRIALS),
    )
    print(f"Saved {out_dir / 'per_trial.npz'}")


if __name__ == "__main__":
    main()
