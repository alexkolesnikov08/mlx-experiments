"""
Theoretical performance analysis and profiling tools for the Depthwise MNIST model.
"""
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_model import DepthwiseMNIST, forward_fused, fused_dw_bn_relu


def compute_flops_per_pass() -> dict:
    """Compute theoretical FLOPs per layer for one forward pass."""
    conv1 = 2 * 3 * 3 * 1 * 48 * 28 * 28

    s0_dw = 2 * 3 * 3 * 48 * 14 * 14
    s0_pw = 2 * 1 * 1 * 48 * 96 * 14 * 14
    s0_skip = 2 * 1 * 1 * 48 * 96 * 14 * 14

    s1_dw = 2 * 3 * 3 * 96 * 14 * 14
    s1_pw = 2 * 1 * 1 * 96 * 96 * 14 * 14

    s2_dw = 2 * 3 * 3 * 96 * 7 * 7
    s2_pw = 2 * 1 * 1 * 96 * 192 * 7 * 7
    s2_skip = 2 * 1 * 1 * 96 * 192 * 7 * 7

    s3_dw = 2 * 3 * 3 * 192 * 7 * 7
    s3_pw = 2 * 1 * 1 * 192 * 192 * 7 * 7

    fc = 2 * 192 * 10

    layers = {
        "conv1": conv1,
        "stage0_dw": s0_dw, "stage0_pw": s0_pw, "stage0_skip": s0_skip,
        "stage1_dw": s1_dw, "stage1_pw": s1_pw,
        "stage2_dw": s2_dw, "stage2_pw": s2_pw, "stage2_skip": s2_skip,
        "stage3_dw": s3_dw, "stage3_pw": s3_pw,
        "fc": fc,
    }
    total_compute = sum(layers.values())

    # Elementwise: BN ~5 FLOPs/elem, ReLU 1 FLOP/elem, Add 1 FLOP/elem
    t_c1 = 28 * 28 * 48
    t_s0dw = 14 * 14 * 48
    t_s0pw = 14 * 14 * 96
    t_s1dw = 14 * 14 * 96
    t_s2dw = 7 * 7 * 96
    t_s2pw = 7 * 7 * 192
    t_s3dw = 7 * 7 * 192

    bn = 5 * (t_c1 + t_s0dw + t_s0pw + t_s0pw + t_s1dw + t_s1dw + t_s2dw + t_s2pw + t_s2pw + t_s3dw + t_s3dw)
    relu = t_c1 + t_s0dw + t_s0pw + t_s1dw + t_s1dw + t_s2dw + t_s2pw + t_s3dw + t_s3dw
    add = t_s0pw + t_s1dw + t_s2pw + t_s3dw
    total_elementwise = bn + relu + add

    print(f"{'Layer':<20} {'FLOPs':>12}")
    print("-" * 33)
    for name, flops_val in layers.items():
        print(f"{name:<20} {flops_val:>12,}")
    print("-" * 33)
    print(f"{'total_compute':<20} {total_compute:>12,}")
    print(f"{'total_elementwise':<20} {total_elementwise:>12,}")
    print(f"{'total_all':<20} {total_compute + total_elementwise:>12,}")

    return {
        **layers,
        "total_compute": total_compute,
        "total_elementwise": total_elementwise,
        "total_all": total_compute + total_elementwise,
    }


def count_params_per_layer() -> dict:
    """Count parameters per layer for the MLX model."""
    model = DepthwiseMNIST()

    params = {}
    params["conv1"] = (model.conv1.weight.size
                       + model.bn1.weight.size + model.bn1.bias.size)

    for i in range(4):
        layer = model.stages.layers[i]
        total = layer.dw.weight.size
        total += layer.bn1.weight.size + layer.bn1.bias.size
        total += layer.pw.weight.size
        total += layer.bn2.weight.size + layer.bn2.bias.size
        if layer.use_skip:
            total += layer.skip_conv.weight.size
            total += layer.skip_bn.weight.size + layer.skip_bn.bias.size
        params[f"stage{i}"] = total

    params["fc"] = model.fc.weight.size + model.fc.bias.size
    total = sum(params.values())
    params["total"] = total

    print(f"\n{'Layer':<20} {'Params':>10}")
    print("-" * 31)
    for name, count in params.items():
        if name == "total":
            continue
        print(f"{name:<20} {count:>10,}")
    print("-" * 31)
    print(f"{'total':<20} {total:>10,}")

    return params


def compute_arithmetic_intensity(layer_name: str, in_c: int, out_c: int,
                                  h_in: int, w_in: int, stride: int,
                                  kernel_size: int = 3, fused: bool = False) -> dict:
    """
    FLOPs / bytes_moved for a given conv layer.

    For depthwise conv:
      bytes_read = (C_in * H_in * W_in + 9 * C_in) * 4
      bytes_written = C_in * H_out * W_out * 4
      bytes_total = bytes_read + bytes_written
      FLOPs = 2 * 9 * C_in * H_out * W_out
      AI = FLOPs / bytes_total

    For fused (DW+BN+ReLU):
      bytes_saved = intermediate tensor read/write eliminated
      AI_fused = FLOPs_total / (bytes_total - bytes_saved)
    """
    h_out = h_in // stride
    w_out = w_in // stride

    is_depthwise = (kernel_size > 1 and in_c == out_c)

    if is_depthwise:
        weight_elems = in_c * kernel_size * kernel_size
        flops = 2 * kernel_size * kernel_size * in_c * h_out * w_out
    else:
        weight_elems = in_c * out_c * kernel_size * kernel_size
        flops = 2 * kernel_size * kernel_size * in_c * out_c * h_out * w_out

    bytes_read = (in_c * h_in * w_in + weight_elems) * 4
    bytes_written = out_c * h_out * w_out * 4
    bytes_total = bytes_read + bytes_written

    if fused:
        bytes_saved = in_c * h_out * w_out * 4
        denominator = bytes_total - bytes_saved
        ai = flops / denominator if denominator > 0 else float('inf')
    else:
        bytes_saved = 0
        ai = flops / bytes_total if bytes_total > 0 else float('inf')

    return {
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "bytes_total": bytes_total,
        "bytes_saved": bytes_saved,
        "FLOPs": flops,
        "AI": ai,
    }


def layer_wise_profile(model, x, fused: bool = False):
    """
    Run forward pass with per-layer timing.

    If fused=False: use standard model.__call__
    If fused=True: use forward_fused but instrument each stage

    Returns dict of {layer_name: time_ms}
    """
    times = {}

    mx.eval()
    t0 = time.perf_counter()
    x = nn.relu(model.bn1(model.conv1(x)))
    mx.eval()
    t1 = time.perf_counter()
    times["conv1+batch+relu"] = (t1 - t0) * 1000

    if not fused:
        for i, stage in enumerate(model.stages.layers):
            t0 = time.perf_counter()
            x = stage(x)
            mx.eval()
            t1 = time.perf_counter()
            times[f"stage{i}"] = (t1 - t0) * 1000
    else:
        for i, stage in enumerate(model.stages.layers):
            s = stage
            identity = x
            stride = s.dw.stride[0]
            t0 = time.perf_counter()
            x = fused_dw_bn_relu(
                x, s.dw.weight, s.bn1.weight, s.bn1.bias,
                s.bn1.running_mean, s.bn1.running_var, stride,
            )
            x = s.bn2(s.pw(x))
            if s.use_skip:
                identity = s.skip_bn(s.skip_conv(identity))
            x = nn.relu(x + identity)
            mx.eval()
            t1 = time.perf_counter()
            times[f"stage{i}"] = (t1 - t0) * 1000

    t0 = time.perf_counter()
    x = x.mean(axis=(1, 2))
    x = model.fc(x)
    mx.eval()
    t1 = time.perf_counter()
    times["head"] = (t1 - t0) * 1000

    return times


def roofline_analysis(batch_sizes: list = None, latencies_ms: dict = None):
    """
    Generate roofline model data.

    Peak M1 GPU (default values):
      peak_fp32_tflops = 2.662  # TFLOPS
      peak_bandwidth_gbps = 68  # GB/s

    For each (backend, batch_size):
      achieved_tflops = total_flops * batch_size / (latency_s * 1e12)
      achieved_bw_gbps = total_bytes * batch_size / (latency_s * 1e9)
      arithmetic_intensity = total_flops / total_bytes

    Print formatted table.
    """
    if batch_sizes is None:
        batch_sizes = [1, 32, 64, 128, 256]
    if latencies_ms is None:
        latencies_ms = {("standard", bs): 0.5 * bs ** 0.3 for bs in batch_sizes}

    peak_fp32_tflops = 2.662
    peak_bandwidth_gbps = 68.0

    total_flops_per_sample = compute_flops_per_pass()["total_all"]

    layer_configs = [
        ("conv1", 1, 48, 28, 28, 1, 3),
        ("stage0_dw", 48, 48, 28, 28, 2, 3),
        ("stage0_pw", 48, 96, 14, 14, 1, 1),
        ("stage0_skip", 48, 96, 28, 28, 2, 1),
        ("stage1_dw", 96, 96, 14, 14, 1, 3),
        ("stage1_pw", 96, 96, 14, 14, 1, 1),
        ("stage2_dw", 96, 96, 14, 14, 2, 3),
        ("stage2_pw", 96, 192, 7, 7, 1, 1),
        ("stage2_skip", 96, 192, 14, 14, 2, 1),
        ("stage3_dw", 192, 192, 7, 7, 1, 3),
        ("stage3_pw", 192, 192, 7, 7, 1, 1),
        ("fc", 192, 10, 1, 1, 1, 1),
    ]

    total_bytes_per_sample = 0
    for name, in_c, out_c, h_in, w_in, stride, k in layer_configs:
        ai_info = compute_arithmetic_intensity(name, in_c, out_c, h_in, w_in, stride, k)
        total_bytes_per_sample += ai_info["bytes_total"]

    print(f"\n{' Roofline Analysis ':=^60}")
    print(f"Peak FP32: {peak_fp32_tflops} TFLOPS  |  Peak BW: {peak_bandwidth_gbps} GB/s")
    print(f"Total FLOPs/sample: {total_flops_per_sample:,}")
    print(f"Total Bytes/sample: {total_bytes_per_sample:,}")
    print(f"Arithmetic Intensity: {total_flops_per_sample / total_bytes_per_sample:.2f} FLOPs/byte")
    print()
    print(f"{'Backend':<12} {'Batch':>6} {'Lat(ms)':>10} {'TFLOPS':>10} {'GB/s':>10} {'AI':>8}")
    print("-" * 58)

    for (backend, bs), lat_ms in sorted(latencies_ms.items()):
        lat_s = lat_ms / 1000.0
        total_ops = total_flops_per_sample * bs
        total_bytes = total_bytes_per_sample * bs
        achieved_tflops = total_ops / (lat_s * 1e12)
        achieved_bw = total_bytes / (lat_s * 1e9)
        ai = total_flops_per_sample / total_bytes_per_sample
        print(f"{backend:<12} {bs:>6} {lat_ms:>10.3f} {achieved_tflops:>10.4f} {achieved_bw:>10.2f} {ai:>8.2f}")


def print_model_summary():
    """Print a comprehensive summary: params, FLOPs, AI, and memory footprint per layer."""
    print("=" * 70)
    print("  Depthwise MNIST - Model Summary")
    print("=" * 70)

    flops = compute_flops_per_pass()
    params = count_params_per_layer()

    layer_configs = [
        ("conv1", 1, 48, 28, 28, 1, 3, "conv1"),
        ("stage0_dw", 48, 48, 28, 28, 2, 3, "stage0"),
        ("stage0_pw", 48, 96, 14, 14, 1, 1, "stage0"),
        ("stage0_skip", 48, 96, 28, 28, 2, 1, "stage0"),
        ("stage1_dw", 96, 96, 14, 14, 1, 3, "stage1"),
        ("stage1_pw", 96, 96, 14, 14, 1, 1, "stage1"),
        ("stage2_dw", 96, 96, 14, 14, 2, 3, "stage2"),
        ("stage2_pw", 96, 192, 7, 7, 1, 1, "stage2"),
        ("stage2_skip", 96, 192, 14, 14, 2, 1, "stage2"),
        ("stage3_dw", 192, 192, 7, 7, 1, 3, "stage3"),
        ("stage3_pw", 192, 192, 7, 7, 1, 1, "stage3"),
        ("fc", 192, 10, 1, 1, 1, 1, "fc"),
    ]

    print(f"\n{'Layer':<20} {'Params':>8} {'FLOPs':>12} {'Bytes':>10} {'AI':>8}")
    print("-" * 60)
    total_params = 0
    total_flops_sum = 0
    total_bytes = 0
    for cfg in layer_configs:
        name, in_c, out_c, h_in, w_in, stride, k, param_key = cfg
        ai_info = compute_arithmetic_intensity(name, in_c, out_c, h_in, w_in, stride, k)
        pcount = params.get(param_key, 0) if param_key in params else 0
        fcount = flops.get(name, 0)
        total_params += pcount
        total_flops_sum += fcount
        total_bytes += ai_info["bytes_total"]
        print(f"{name:<20} {pcount:>8,} {fcount:>12,} {ai_info['bytes_total']:>10,} {ai_info['AI']:>8.2f}")
    print("-" * 60)
    print(f"{'Total':<20} {total_params:>8,} {total_flops_sum:>12,} {total_bytes:>10,} {total_flops_sum/total_bytes:>8.2f}")


if __name__ == "__main__":
    print_model_summary()
    print("\n---\n")
    flops = compute_flops_per_pass()
    print(f"Total FLOPs per pass: {flops['total_all']:,.0f}")
    print(f"Total params: {count_params_per_layer()['total']:,}")
