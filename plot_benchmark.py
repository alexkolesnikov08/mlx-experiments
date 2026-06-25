"""
Generate comparison plot for all three backends.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZES = [1, 16, 64, 128, 256, 512]

data = {
    "PyTorch MPS": {
        "latency_ms": [2.583, 4.407, 9.911, 57.838, 58.683, 94.670],
        "throughput": [387, 3631, 6457, 2213, 4362, 5408],
    },
    "MLX (baseline)": {
        "latency_ms": [2.069, 8.981, 38.069, 83.536, 171.113, 344.660],
        "throughput": [483, 1782, 1681, 1532, 1496, 1486],
    },
    "MLX (fused)": {
        "latency_ms": [1.369, 4.196, 14.664, 29.811, 70.234, 99.253],
        "throughput": [731, 3813, 4364, 4294, 3645, 5159],
    },
}

colors = {"PyTorch MPS": "#ee6c4d", "MLX (baseline)": "#3d5a80", "MLX (fused)": "#2a9d8f"}
markers = {"PyTorch MPS": "o", "MLX (baseline)": "s", "MLX (fused)": "^"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for name, d in data.items():
    ax1.plot(BATCH_SIZES, d["latency_ms"], marker=markers[name], color=colors[name], label=name, linewidth=2, markersize=7)
    ax2.plot(BATCH_SIZES, d["throughput"], marker=markers[name], color=colors[name], label=name, linewidth=2, markersize=7)

ax1.set_xlabel("Batch Size", fontsize=12)
ax1.set_ylabel("Avg Latency (ms)", fontsize=12)
ax1.set_title("Inference Latency", fontsize=14, fontweight="bold")
ax1.set_xscale("log", base=2)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(BATCH_SIZES)
ax1.set_xticklabels([str(b) for b in BATCH_SIZES])

ax2.set_xlabel("Batch Size", fontsize=12)
ax2.set_ylabel("Throughput (samples/sec)", fontsize=12)
ax2.set_title("Inference Throughput", fontsize=14, fontweight="bold")
ax2.set_xscale("log", base=2)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(BATCH_SIZES)
ax2.set_xticklabels([str(b) for b in BATCH_SIZES])
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

plt.suptitle("Depthwise MNIST — Backend Comparison (Apple M1)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("output/benchmark_comparison.png", dpi=150, bbox_inches="tight")
print("Saved output/benchmark_comparison.png")
