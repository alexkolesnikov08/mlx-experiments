"""
MLX port of the Depthwise MNIST model.
MLX uses channel-last (N, H, W, C) vs PyTorch (N, C, H, W).
Weights are loaded from PyTorch checkpoint with layout conversion.
"""
import mlx.core as mx
import mlx.nn as nn
import torch
from pathlib import Path
from mlx.utils import tree_unflatten


def _nested_update(flat, model):
    model.update(tree_unflatten(list(flat.items())))
    return model


def perm_conv(arr):
    """PyTorch (C_out, C_in, H, W) -> MLX (C_out, H, W, C_in)."""
    return mx.array(arr.transpose(0, 2, 3, 1))


def load_pt_weights(pt_path, mlx_model):
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


def get_model(weights_path=None):
    model = DepthwiseMNIST()
    if weights_path is not None:
        model = load_pt_weights(weights_path, model)
    model.eval()
    return model


# ---- Fused Metal kernel for depthwise conv 3x3 + BN + ReLU ----

_DW_FUSED_SOURCE = """
uint nw = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;
uint col = thread_position_in_grid.z;

uint C_in = inp_shape[3];
uint H_in = inp_shape[1];
uint W_in = inp_shape[2];
uint n = nw / C_in;
uint c = nw % C_in;

uint H_out = H_in / stride;
uint W_out = W_in / stride;

int h_in = (int)row * stride - 1;
int w_in = (int)col * stride - 1;

float sum_val = 0.0;
uint widx = c * 9;

for (int kh = 0; kh < 3; kh++) {
    for (int kw = 0; kw < 3; kw++) {
        int h_src = h_in + kh;
        int w_src = w_in + kw;
        if (h_src >= 0 && h_src < (int)H_in && w_src >= 0 && w_src < (int)W_in) {
            uint inp_idx = ((n * H_in + h_src) * W_in + w_src) * C_in + c;
            float x_val = inp[inp_idx];
            float weight_val = w[widx + kh * 3 + kw];
            sum_val += x_val * weight_val;
        }
    }
}

float eps = 1e-5;
float normed = (sum_val - running_mean[c]) / metal::sqrt(running_var[c] + eps);
float activated = normed * gamma[c] + beta[c];
activated = metal::max(0.0f, activated);

uint out_idx = ((n * H_out + row) * W_out + col) * C_in + c;
out[out_idx] = activated;
"""

_dw_fused_kernel = mx.fast.metal_kernel(
    name="dw_conv_bn_relu",
    input_names=["inp", "w", "gamma", "beta", "running_mean", "running_var"],
    output_names=["out"],
    source=_DW_FUSED_SOURCE,
)


def fused_dw_bn_relu(x, w, gamma, beta, running_mean, running_var, stride):
    N, H, W, C = x.shape
    H_out, W_out = H // stride, W // stride
    (out,) = _dw_fused_kernel(
        inputs=[x, w, gamma, beta, running_mean, running_var],
        template=[("stride", stride)],
        grid=(N * C, H_out, W_out),
        threadgroup=(256, 1, 1),
        output_shapes=[(N, H_out, W_out, C)],
        output_dtypes=[mx.float32],
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
