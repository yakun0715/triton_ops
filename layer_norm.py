import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl
import triton.testing as tt


DEVICE = triton.runtime.driver.active.get_active_torch_device()

def pytorch_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    x: [**, C]   可以是多维输入
    weight: [C]
    bias: [C]
    eps: float
    return [**, C]
    """
    assert not x.requires_grad and not weight.requires_grad and not bias.requires_grad
    return F.layer_norm(x, x.shape[-1:], weight, bias, eps)

def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    x: [**, C]   可以是多维输入
    weight: [C]
    bias: [C]
    eps: float
    return [**, C]
    """
    assert not x.requires_grad and not weight.requires_grad and not bias.requires_grad
    origin_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    batch, dim = x.shape
    output = torch.empty_like(x, device=x.device)

    BLOCK_SIZE_FEAT = triton.next_power_of_2(dim)

    grid = lambda META: (triton.cdiv(batch, META["BLOCK_SIZE_BATCH"]),)
    layer_norm_kernel[grid](
        x,
        output,
        weight,
        bias,
        eps,
        batch,
        dim,
        *x.stride(),
        *output.stride(),
        BLOCK_SIZE_FEAT=BLOCK_SIZE_FEAT,
    )
    return output.reshape(origin_shape)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_BATCH": 1}, num_warps=4),
        triton.Config({"BLOCK_SIZE_BATCH": 2}, num_warps=4),
        triton.Config({"BLOCK_SIZE_BATCH": 4}, num_warps=4),
        triton.Config({"BLOCK_SIZE_BATCH": 8}, num_warps=4),
        triton.Config({"BLOCK_SIZE_BATCH": 16}, num_warps=4),
    ],
    key=["batch", "dim"],
)
@triton.jit
def layer_norm_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    eps,
    batch,
    dim,
    input_batch_stride,
    input_feat_stride,
    output_batch_stride,
    output_feat_stride,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """
    每个block处理多行数据
    input_ptr: [batch, dim]
    output_ptr: [batch, dim]
    weight_ptr: [dim]
    bias_ptr: [dim]
    eps: float
    batch: int
    dim: int
    input_batch_stride: int
    input_feat_stride: int
    output_batch_stride: int
    output_feat_stride: int
    BLOCK_SIZE_BATCH: int
    """
    batch_id = tl.program_id(0)
    batch_offset = batch_id * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

    batch_mask = batch_offset < batch
    feat_mask = feat_offset < dim

    input_ptrs = input_ptr + input_batch_stride * batch_offset[:, None] + input_feat_stride * feat_offset[None, :]
    input = tl.load(input_ptrs, mask=batch_mask[:, None] & feat_mask[None, :], other=0.0)

    # 只在有效的特征维度上做统计，忽略 padding 的 lane
    mean = tl.sum(input, axis=-1, keep_dims=True) / dim
    diff = (input - mean) * feat_mask[None, :].to(tl.float32)
    var = tl.sum(diff * diff, axis=-1, keep_dims=True) / dim
    output = (input - mean) / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + feat_offset, mask=feat_mask)
    bias = tl.load(bias_ptr + feat_offset, mask=feat_mask)    
    output = output * weight + bias

    tl.store(
        output_ptr
        + output_batch_stride * batch_offset[:, None]
        + output_feat_stride * feat_offset[None, :],
        output,
        mask=batch_mask[:, None] & feat_mask[None, :],
    )


def compare_result():
    eps = 1e-5

    # 多组不同形状的测试用例（最后一维视为 C）
    test_shapes = [
        (10, 16),          # 2D 小尺寸
        (32, 128),         # 2D 稍大
        (4, 8, 16),        # 3D
        (2, 3, 64),        # 3D 不同 C
        (1, 7, 513),       # 3D，非 2 的幂，测试 next_power_of_2
        (2, 3, 4, 5, 6),   # 5D，最后一维为 C=6
    ]

    for shape in test_shapes:
        x = torch.randn(*shape, device=DEVICE, requires_grad=False)
        dim = x.shape[-1]

        # 共享一份 weight / bias（都不需要梯度）
        weight = torch.randn(dim, device=DEVICE, requires_grad=False)
        bias = torch.randn(dim, device=DEVICE, requires_grad=False)

        # 计算两种实现的输出
        y_torch = pytorch_layer_norm(x, weight, bias, eps)
        y_triton = triton_layer_norm(x, weight, bias, eps)

        # 比较最大绝对误差和是否在一定误差范围内一致
        max_diff = (y_torch - y_triton).abs().max().item()
        allclose = torch.allclose(y_torch, y_triton, atol=1e-4, rtol=1e-4)

        print(f"shape={shape}, max abs diff: {max_diff}, allclose: {allclose}")

        # 如果希望严格一点，可以直接断言
        assert allclose, f"pytorch_layer_norm 和 triton_layer_norm 输出不一致，shape={shape}！"


@tt.perf_report(
    tt.Benchmark(
        x_names=["B"],  # batch size
        x_vals=[2**i for i in range(5, 13)],  # 32 .. 4096
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="layer-norm-performance",
        args={"C": 1024},  # feature dimension
    )
)
def benchmark(B, C, provider):
    eps = 1e-5
    x = torch.randn(B, C, device=DEVICE, dtype=torch.float32, requires_grad=False)
    weight = torch.randn(C, device=DEVICE, dtype=torch.float32, requires_grad=False)
    bias = torch.randn(C, device=DEVICE, dtype=torch.float32, requires_grad=False)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = tt.do_bench(
            lambda: pytorch_layer_norm(x, weight, bias, eps), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = tt.do_bench(
            lambda: triton_layer_norm(x, weight, bias, eps), quantiles=quantiles
        )

    # 粗略按 3 * |x| * sizeof(float) 计算带宽，主要看相对性能
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def main():
    compare_result()
    benchmark.run(print_data=True, show_plots=True)


if __name__ == "__main__":
    main()