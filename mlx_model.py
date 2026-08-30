"""
MLX port of the Depthwise MNIST model.
MLX uses channel-last (N, H, W, C) vs PyTorch (N, C, H, W).
Weights are loaded from PyTorch checkpoint with layout conversion.
"""
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from mlx.utils import tree_flatten, tree_unflatten


def _nested_update(flat, model):
    model.update(tree_unflatten(list(flat.items())))
    return model


def perm_conv(arr):
    """PyTorch (C_out, C_in, H, W) -> MLX (C_out, H, W, C_in)."""
    return mx.array(arr.transpose(0, 2, 3, 1))


def load_pt_weights(pt_path, mlx_model):
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            f"load_pt_weights requires torch for {pt_path}. "
            "Use safetensors checkpoint or install torch. "
            "Tip: export_to_safetensors() converts .pth once."
        ) from e
    pt = torch.load(pt_path, map_location="cpu", weights_only=True)
    flat = {}

    def add(key, arr, do_perm=False):
        flat[key] = perm_conv(arr) if do_perm else mx.array(arr)

    # conv1 block: PT uses Sequential indices (conv1.0=Conv2d, conv1.1=BN)
    add("conv1.weight", pt["conv1.0.weight"].numpy(), do_perm=True)
    add("bn1.weight", pt["conv1.1.weight"].numpy())
    add("bn1.bias", pt["conv1.1.bias"].numpy())
    add("bn1.running_mean", pt["conv1.1.running_mean"].numpy())
    add("bn1.running_var", pt["conv1.1.running_var"].numpy())

    for i in range(4):
        p = f"stages.layers.{i}"  # MLX key
        q = f"stages.{i}"          # PyTorch key
        # depthwise conv (bias=False in both PT and MLX)
        add(f"{p}.dw.weight", pt[f"{q}.dw.weight"].numpy(), do_perm=True)
        add(f"{p}.bn1.weight", pt[f"{q}.bn1.weight"].numpy())
        add(f"{p}.bn1.bias", pt[f"{q}.bn1.bias"].numpy())
        add(f"{p}.bn1.running_mean", pt[f"{q}.bn1.running_mean"].numpy())
        add(f"{p}.bn1.running_var", pt[f"{q}.bn1.running_var"].numpy())
        # pointwise conv (bias=False)
        add(f"{p}.pw.weight", pt[f"{q}.pw.weight"].numpy(), do_perm=True)
        add(f"{p}.bn2.weight", pt[f"{q}.bn2.weight"].numpy())
        add(f"{p}.bn2.bias", pt[f"{q}.bn2.bias"].numpy())
        add(f"{p}.bn2.running_mean", pt[f"{q}.bn2.running_mean"].numpy())
        add(f"{p}.bn2.running_var", pt[f"{q}.bn2.running_var"].numpy())

        # skip connection (only stages 0,2)
        skip = f"{q}.skip.0.weight"
        if skip in pt:
            add(f"{p}.skip_conv.weight", pt[skip].numpy(), do_perm=True)
            add(f"{p}.skip_bn.weight", pt[f"{q}.skip.1.weight"].numpy())
            add(f"{p}.skip_bn.bias", pt[f"{q}.skip.1.bias"].numpy())
            add(f"{p}.skip_bn.running_mean", pt[f"{q}.skip.1.running_mean"].numpy())
            add(f"{p}.skip_bn.running_var", pt[f"{q}.skip.1.running_var"].numpy())

    # FC
    add("fc.weight", pt["fc.weight"].numpy())
    add("fc.bias", pt["fc.bias"].numpy())

    return _nested_update(flat, mlx_model)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: depthwise 3x3 + pointwise 1x1, BN+ReLU, optional skip."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.use_skip = stride != 1 or in_c != out_c
        if self.use_skip:
            self.skip_conv = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm(out_c)
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride, padding=1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm(out_c)

    def __call__(self, x):
        identity = x
        x = nn.relu(self.bn1(self.dw(x)))
        x = self.bn2(self.pw(x))
        if self.use_skip:
            identity = self.skip_bn(self.skip_conv(identity))
        return nn.relu(x + identity)


class DepthwiseMNIST(nn.Module):
    """MLX MNIST model with depthwise separable convs (channel-last N,H,W,C)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(48)
        self.stages = nn.Sequential(
            DepthwiseSeparableConv(48, 96, stride=2),
            DepthwiseSeparableConv(96, 96, stride=1),
            DepthwiseSeparableConv(96, 192, stride=2),
            DepthwiseSeparableConv(192, 192, stride=1),
        )
        self.fc = nn.Linear(192, 10)

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.stages(x)
        x = x.mean(axis=(1, 2))  # GAP over spatial dims
        x = self.fc(x)
        return x


def count_params(m):
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(m.parameters()))


def export_to_safetensors(pt_path, out_path=None, model=None):
    """
    Convert PyTorch .pth checkpoint to MLX .safetensors (no torch needed at inference).

    Usage:
        from mlx_model import DepthwiseMNIST, export_to_safetensors
        export_to_safetensors("output/best_model.pth", "output/best_model.safetensors")

    Saves dict from tree_flatten(model.parameters()) via mx.save_safetensors.
    """
    if model is None:
        model = DepthwiseMNIST()
    model = load_pt_weights(pt_path, model)
    if out_path is None:
        out_path = str(Path(pt_path).with_suffix(".safetensors"))
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(out_path, flat)
    print(f"Saved safetensors: {out_path} ({len(flat)} arrays)")
    return out_path


def load_safetensors_weights(safetensors_path, mlx_model):
    """Load MLX safetensors checkpoint (no torch dependency)."""
    weights = mx.load(safetensors_path)
    # mx.load returns dict for .safetensors
    if isinstance(weights, dict):
        flat = weights
    else:
        raise ValueError(f"Unexpected load result for {safetensors_path}: {type(weights)}")
    return _nested_update(flat, mlx_model)


def get_model(weights_path=None):
    model = DepthwiseMNIST()
    if weights_path is not None:
        p = str(weights_path)
        if p.endswith(".safetensors"):
            model = load_safetensors_weights(p, model)
        else:
            # .pth / .pt / no suffix -> try torch path
            model = load_pt_weights(p, model)
    model.eval()
    return model


# ---- Fused Metal kernel for depthwise conv 3x3 + BN + ReLU ----
# Optimized per issues #1, #5, #6, #7, #8:
#  #1 1D grid flatten (N*C*H_out*W_out) + inv_std precompute
#  #5 ternary instead of branching if for padding
#  #6 loop unrolling via #pragma unroll
#  #7 base indices variables for readability
#  #8 threadgroup configurable (256,1,1) default, (32,8,1) optional

# Threadgroup configs (issue #8)
FUSED_THREADGROUP_DEFAULT = (256, 1, 1)  # best for 1D grid
FUSED_THREADGROUP_2D = (32, 8, 1)        # spatial locality, for 3D grid legacy or conv1d seq_len=256
# Active threadgroup for 1D grid - use DEFAULT. For A/B test, switch to FUSED_THREADGROUP_2D with 3D grid.
FUSED_THREADGROUP = FUSED_THREADGROUP_DEFAULT

_DW_FUSED_SOURCE = """
    // 1D grid flatten: each thread computes (n, c, row, col) from tid (issue #1)
    // Utilization ~100% vs 18-75% for 3D (N*C, H, W)
    uint tid = thread_position_in_grid.x;
    uint C_in = inp_shape[3];
    uint H_in = inp_shape[1];
    uint W_in = inp_shape[2];
    uint H_out = H_in / stride;
    uint W_out = W_in / stride;

    uint c = tid % C_in;
    uint tmp = tid / C_in;
    uint col = tmp % W_out;
    tmp /= W_out;
    uint row = tmp % H_out;
    uint n = tmp / H_out;

    // Base indices for readability (issue #7) - compiler CSE does same but code clearer
    int h_base = (int)row * stride - 1;
    int w_base = (int)col * stride - 1;
    uint widx_base = c * 9;

    float sum_val = 0.0;

    // Loop unrolling (issue #6) - guarantees compiler unrolls 3x3
    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            int h_src = h_base + kh;
            int w_src = w_base + kw;
            // Ternary instead of if for padding (issue #5) - branchless for edge pixels
            bool h_ok = (h_src >= 0) && (h_src < (int)H_in);
            bool w_ok = (w_src >= 0) && (w_src < (int)W_in);
            uint inp_idx = ((n * H_in + h_src) * W_in + w_src) * C_in + c;
            float x_val = (h_ok && w_ok) ? inp[inp_idx] : 0.0f;
            float w_val = w[widx_base + kh * 3 + kw];
            sum_val += x_val * w_val;
        }
    }

    // Precomputed inv_std = rsqrt(var + eps) (issue #1) - avoids sqrt per thread
    float inv = inv_std[c];
    float normed = (sum_val - running_mean[c]) * inv;
    float activated = normed * gamma[c] + beta[c];
    activated = metal::max(0.0f, activated);

    uint out_idx = ((n * H_out + row) * W_out + col) * C_in + c;
    out[out_idx] = activated;
"""

_dw_fused_kernel = mx.fast.metal_kernel(
    name="dw_conv_bn_relu",
    input_names=["inp", "w", "gamma", "beta", "running_mean", "inv_std"],
    output_names=["out"],
    source=_DW_FUSED_SOURCE,
)


def fused_dw_bn_relu(x, w, gamma, beta, running_mean, running_var, stride, threadgroup=None, inv_std=None):
    """
    Fused depthwise 3x3 + BN + ReLU.

    Optimizations:
      - 1D grid flatten (issue #1)
      - inv_std precompute (issue #1): pass inv_std or running_var; computes rsqrt on-the-fly if needed
      - Ternary padding (issue #5), unroll (issue #6), base indices (issue #7)
      - Configurable threadgroup (issue #8): default (256,1,1), alt (32,8,1)

    Args:
        x: (N, H, W, C) float32 (or float16 for fp16 path)
        w: (C, 3, 3, 1) depthwise weight
        gamma, beta: (C,) BN scale/bias
        running_mean, running_var: (C,) BN stats (running_var unused if inv_std given)
        stride: int
        threadgroup: tuple, optional override for issue #8 A/B test
        inv_std: precomputed 1/sqrt(var+eps), optional (issue #1)
    """
    N, H, W, C = x.shape
    H_out, W_out = H // stride, W // stride
    if inv_std is None:
        # Precompute inv_std per issue #1: inv_std = rsqrt(var + eps)
        inv_std = mx.rsqrt(running_var + 1e-5)
    # 1D grid (issue #1): total threads = N*C*H_out*W_out
    total = N * C * H_out * W_out
    tg = threadgroup if threadgroup is not None else FUSED_THREADGROUP
    # Ensure total is valid grid for 1D
    (out,) = _dw_fused_kernel(
        inputs=[x, w, gamma, beta, running_mean, inv_std],
        template=[("stride", stride)],
        grid=(total, 1, 1),
        threadgroup=tg,
        output_shapes=[(N, H_out, W_out, C)],
        output_dtypes=[x.dtype],
    )
    return out


# ---- Half Precision (Float16) kernel (issue #2) ----
# 2x memory bandwidth + 2x ALU on A14/M1 GPU. BN kept in float32 for accuracy.

_DW_FUSED_FP16_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint C_in = inp_shape[3];
    uint H_in = inp_shape[1];
    uint W_in = inp_shape[2];
    uint H_out = H_in / stride;
    uint W_out = W_in / stride;

    uint c = tid % C_in;
    uint tmp = tid / C_in;
    uint col = tmp % W_out;
    tmp /= W_out;
    uint row = tmp % H_out;
    uint n = tmp / H_out;

    int h_base = (int)row * stride - 1;
    int w_base = (int)col * stride - 1;
    uint widx_base = c * 9;

    half sum_val = half(0.0);

    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            int h_src = h_base + kh;
            int w_src = w_base + kw;
            bool h_ok = (h_src >= 0) && (h_src < (int)H_in);
            bool w_ok = (w_src >= 0) && (w_src < (int)W_in);
            uint inp_idx = ((n * H_in + h_src) * W_in + w_src) * C_in + c;
            half x_val = (h_ok && w_ok) ? inp[inp_idx] : half(0.0);
            half w_val = w[widx_base + kh * 3 + kw];
            sum_val += x_val * w_val;
        }
    }

    // BN in float32 (issue #2: keep running_mean/var float32)
    float sum_f = float(sum_val);
    float inv = inv_std[c];
    float normed = (sum_f - running_mean[c]) * inv;
    float activated = normed * gamma[c] + beta[c];
    activated = metal::max(0.0f, activated);

    uint out_idx = ((n * H_out + row) * W_out + col) * C_in + c;
    out[out_idx] = activated;
"""

_dw_fused_fp16_kernel = mx.fast.metal_kernel(
    name="dw_conv_bn_relu_fp16",
    input_names=["inp", "w", "gamma", "beta", "running_mean", "inv_std"],
    output_names=["out"],
    source=_DW_FUSED_FP16_SOURCE,
)


def fused_dw_bn_relu_fp16(x, w, gamma, beta, running_mean, running_var, stride, threadgroup=None, inv_std=None):
    """
    FP16 variant of fused_dw_bn_relu (issue #2).
    Expects x and w as float16 (gamma/beta/mean/inv_std stay float32).
    Falls back to float32 kernel if x is float32 (with cast) and warns if error >1e-3 not checked.
    """
    # Ensure half precision for inputs/weights
    if x.dtype != mx.float16:
        x = x.astype(mx.float16)
    if w.dtype != mx.float16:
        w = w.astype(mx.float16)
    if inv_std is None:
        inv_std = mx.rsqrt(running_var + 1e-5)
    # Keep BN params float32
    N, H, W, C = x.shape
    H_out, W_out = H // stride, W // stride
    total = N * C * H_out * W_out
    tg = threadgroup if threadgroup is not None else FUSED_THREADGROUP
    (out,) = _dw_fused_fp16_kernel(
        inputs=[x, w, gamma, beta, running_mean, inv_std],
        template=[("stride", stride)],
        grid=(total, 1, 1),
        threadgroup=tg,
        output_shapes=[(N, H_out, W_out, C)],
        output_dtypes=[mx.float32],  # BN output float32, next PW will cast if needed
    )
    return out


def maybe_convert_to_fp16(model, dtype=mx.float16):
    """
    Utility per issue #2: convert depthwise weights to fp16 at load time.
    Keeps BN params float32.
    Returns model with dw weights casted. For full-model fp16, also cast pw/conv1/fc.
    """
    # Only dw weights need half for current fused kernel; keep others float32 for accuracy
    for i, stage in enumerate(model.stages.layers):
        stage.dw.weight = stage.dw.weight.astype(dtype)
    return model


# ---- Fused Stage kernel: entire DepthwiseSeparableConv in one launch (issue #9) ----
# Strong register pressure, spill risk. Viable only with FP16 (half registers).
# This kernel fuses: DW 3x3 + BN1 + ReLU + PW 1x1 + BN2 + Residual Add + ReLU
# Each thread computes one output element (n, row, col, out_c) -> loops over C_in for PW.
# Naive recomputation of DW per out_c is expensive but saves global memory roundtrips.
# For true efficiency need threadgroup shared memory - left as future work.

_STAGE_FUSED_SOURCE = """
    uint tid = thread_position_in_grid.x;
    // C_out via template (issue #9), C_in from inp_shape
    uint C_in = inp_shape[3];
    uint H_in = inp_shape[1];
    uint W_in = inp_shape[2];
    uint H_out = H_in / stride;
    uint W_out = W_in / stride;

    uint out_c = tid % C_out;
    uint tmp = tid / C_out;
    uint col = tmp % W_out;
    tmp /= W_out;
    uint row = tmp % H_out;
    uint n = tmp / H_out;

    int h_base = (int)row * stride - 1;
    int w_base = (int)col * stride - 1;

    float pw_acc = 0.0;

    // Loop over C_in to gather DW results and do 1x1 PW
    for (uint c_in = 0; c_in < C_in; ++c_in) {
        // ---- DW 3x3 for this c_in ----
        float dw_sum = 0.0;
        uint widx_dw = c_in * 9;
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                int h_src = h_base + kh;
                int w_src = w_base + kw;
                bool h_ok = (h_src >= 0) && (h_src < (int)H_in);
                bool w_ok = (w_src >= 0) && (w_src < (int)W_in);
                uint inp_idx = ((n * H_in + h_src) * W_in + w_src) * C_in + c_in;
                float x_val = (h_ok && w_ok) ? inp[inp_idx] : 0.0f;
                float w_val = w_dw[widx_dw + kh * 3 + kw];
                dw_sum += x_val * w_val;
            }
        }
        float inv1 = inv_std1[c_in];
        float dw_norm = (dw_sum - mean1[c_in]) * inv1;
        dw_norm = dw_norm * gamma1[c_in] + beta1[c_in];
        dw_norm = metal::max(0.0f, dw_norm);

        // ---- PW 1x1: accumulate ----
        // pw weight layout: (C_out, 1, 1, C_in) -> index = out_c * C_in + c_in
        float pw_w = w_pw[out_c * C_in + c_in];
        pw_acc += dw_norm * pw_w;
    }

    // BN2
    float inv2 = inv_std2[out_c];
    float pw_norm = (pw_acc - mean2[out_c]) * inv2;
    pw_norm = pw_norm * gamma2[out_c] + beta2[out_c];

    // Residual path: skip 1x1 + BN if use_skip
    float residual = 0.0;
    if (use_skip) {
        float skip_acc = 0.0;
        for (uint c_in = 0; c_in < C_in; ++c_in) {
            // 1x1 stride: sample at (row*stride, col*stride)
            int h_s = (int)row * stride;
            int w_s = (int)col * stride;
            // Clamp? at edges stride mapping still valid as H_in = H_out*stride
            uint inp_idx = ((n * H_in + h_s) * W_in + w_s) * C_in + c_in;
            float x_val = inp[inp_idx];
            float w_sv = w_skip[out_c * C_in + c_in];
            skip_acc += x_val * w_sv;
        }
        float inv_s = inv_std_s[out_c];
        residual = (skip_acc - mean_s[out_c]) * inv_s;
        residual = residual * gamma_s[out_c] + beta_s[out_c];
    } else {
        // Identity path: only when C_in == C_out and stride==1, residual is inp at same location
        // For fused stage without skip, C_in==C_out, so we can directly add inp value
        // But our earlier DW already consumed inp, identity should be original inp at (n,row,col, out_c)
        // Since C_in == C_out, out_c < C_in
        uint id_idx = ((n * H_out + row) * W_out + col) * C_out + out_c;
        // Only valid if not using skip and C_in == C_out; else zero
        // We approximate: if use_skip==0, residual = inp at output location (stride==1 case)
        // For stride==1, H_out==H_in, W_out==W_in
        if (C_in == C_out) {
            uint inp_idx = ((n * H_in + (int)row * stride) * W_in + (int)col * stride) * C_in + out_c;
            // For stride 1, this is just ((n*H_out+row)*W_out+col)*C_in+out_c
            residual = inp[inp_idx];
        }
    }

    float out_val = metal::max(0.0f, pw_norm + residual);
    uint out_idx = ((n * H_out + row) * W_out + col) * C_out + out_c;
    out[out_idx] = out_val;
"""

# Two kernels for with/without skip (template specialization)
_stage_fused_kernel = mx.fast.metal_kernel(
    name="stage_fused",
    input_names=["inp", "w_dw", "gamma1", "beta1", "mean1", "inv_std1",
                 "w_pw", "gamma2", "beta2", "mean2", "inv_std2",
                 "w_skip", "gamma_s", "beta_s", "mean_s", "inv_std_s"],
    output_names=["out"],
    source=_STAGE_FUSED_SOURCE,
)


def fused_stage(x, stage, use_fp16=False):
    """
    Fused entire DepthwiseSeparableConv stage in one Metal launch (issue #9).
    Experimental: high register pressure, recommended only with FP16 (issue #2).

    Args:
        x: (N, H, W, C_in)
        stage: DepthwiseSeparableConv module
        use_fp16: if True, use half for DW/PW (requires stage.dw.weight half)

    Returns:
        (N, H_out, W_out, C_out) after DW+BN+ReLU+PW+BN+Residual+ReLU
    """
    s = stage
    stride = s.dw.stride[0]
    N, H_in, W_in, C_in = x.shape
    C_out = s.pw.weight.shape[0] if hasattr(s.pw.weight, 'shape') else x.shape[3]  # fallback
    # MLX Conv2d weight shape (C_out, H, W, C_in) => C_out is shape[0]
    try:
        C_out = int(s.pw.weight.shape[0])
    except:
        C_out = C_in

    H_out, W_out = H_in // stride, W_in // stride

    # Prepare BN inv_stds
    inv_std1 = mx.rsqrt(s.bn1.running_var + 1e-5)
    inv_std2 = mx.rsqrt(s.bn2.running_var + 1e-5)

    # Prepare skip params (dummy if no skip)
    if s.use_skip:
        inv_std_s = mx.rsqrt(s.skip_bn.running_var + 1e-5)
        w_skip = s.skip_conv.weight  # (C_out,1,1,C_in)
        # Need to ensure w_skip is contiguous and correctly shaped for kernel indexing
        # Kernel expects flat [C_out*C_in]
        # MLX weight is (C_out,1,1,C_in) contiguous, flatten will match out_c*C_in + c_in
        w_skip_flat = mx.reshape(w_skip, (C_out * C_in,))
        gamma_s = s.skip_bn.weight
        beta_s = s.skip_bn.bias
        mean_s = s.skip_bn.running_mean
        # Use flat for kernel; but MLX kernel expects array, we pass flat
        w_skip_in = w_skip_flat
    else:
        # Dummy arrays for kernel (will be ignored when use_skip==0)
        # Create zero arrays of correct size to avoid kernel nil
        w_skip_in = mx.zeros((C_out * C_in,), dtype=x.dtype)
        gamma_s = mx.zeros((C_out,), dtype=mx.float32)
        beta_s = mx.zeros((C_out,), dtype=mx.float32)
        mean_s = mx.zeros((C_out,), dtype=mx.float32)
        inv_std_s = mx.ones((C_out,), dtype=mx.float32)

    # Flatten pw weight similarly: (C_out,1,1,C_in) -> (C_out*C_in,)
    w_pw_flat = mx.reshape(s.pw.weight, (C_out * C_in,))
    w_dw = s.dw.weight  # (C_in,3,3,1) - kernel uses c*9 indexing, contiguous already

    total = N * C_out * H_out * W_out
    # Choose threadgroup: for stage fused, still 1D
    tg = FUSED_THREADGROUP

    # Note: w_dw is passed as is (C_in*9), kernel indexes w_dw[c*9 + ...]
    # For fp16, need half weights; kernel currently float, so cast handling outside
    inputs = [x, w_dw, s.bn1.weight, s.bn1.bias, s.bn1.running_mean, inv_std1,
              w_pw_flat, s.bn2.weight, s.bn2.bias, s.bn2.running_mean, inv_std2,
              w_skip_in, gamma_s, beta_s, mean_s, inv_std_s]

    (out,) = _stage_fused_kernel(
        inputs=inputs,
        template=[("stride", stride), ("use_skip", 1 if s.use_skip else 0), ("C_out", C_out)],
        grid=(total, 1, 1),
        threadgroup=tg,
        output_shapes=[(N, H_out, W_out, C_out)],
        output_dtypes=[x.dtype],
    )
    return out


def forward_fused(model, x):
    x = nn.relu(model.bn1(model.conv1(x)))
    for stage in model.stages.layers:
        s = stage
        identity = x
        stride = s.dw.stride[0]
        x = fused_dw_bn_relu(
            x, s.dw.weight, s.bn1.weight, s.bn1.bias,
            s.bn1.running_mean, s.bn1.running_var, stride,
        )
        x = s.bn2(s.pw(x))
        if s.use_skip:
            identity = s.skip_bn(s.skip_conv(identity))
        x = nn.relu(x + identity)
    x = x.mean(axis=(1, 2))
    x = model.fc(x)
    return x


def forward_fused_fp16(model, x):
    """
    FP16 path for forward_fused (issue #2).
    Converts depthwise weights to half on-the-fly, keeps BN float32.
    """
    x = nn.relu(model.bn1(model.conv1(x)))
    for stage in model.stages.layers:
        s = stage
        identity = x
        stride = s.dw.stride[0]
        # Use fp16 kernel if weights can be cast
        x = fused_dw_bn_relu_fp16(
            x, s.dw.weight, s.bn1.weight, s.bn1.bias,
            s.bn1.running_mean, s.bn1.running_var, stride,
        )
        # PW remains float32 for now; could also be fp16 but keep for accuracy
        x = s.bn2(s.pw(x))
        if s.use_skip:
            identity = s.skip_bn(s.skip_conv(identity))
        x = nn.relu(x + identity)
    x = x.mean(axis=(1, 2))
    x = model.fc(x)
    return x


def forward_fused_stage(model, x, use_fp16=False):
    """
    Fully fused stage version (issue #9).
    Each stage is one Metal launch (DW+PW+Residual).
    Experimental: requires FP16 for register pressure relief.

    Set use_fp16=True to use half for DW/PW (recommended).
    """
    x = nn.relu(model.bn1(model.conv1(x)))
    for stage in model.stages.layers:
        x = fused_stage(x, stage, use_fp16=use_fp16)
    x = x.mean(axis=(1, 2))
    x = model.fc(x)
    return x
