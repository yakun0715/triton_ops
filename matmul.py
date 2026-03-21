import torch
from torch import nn
import triton
import triton.language as tl


@torch.no_grad()
def pytorch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [B, M, K]
    b: [B, K, N]
    return: [B, M, N]
    """
    assert len(a.shape) == 3 and len(b.shape) == 3
    assert a.shape[-1] == b.shape[-2]
    return torch.matmul(a, b)

def triton_matmul_split_k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [B, M, K]
    b: [B, K, N]
    return: [B, M, N]
    """
    assert len(a.shape) == 3 and len(b.shape) == 3
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[1]
    B, M, K = a.shape
    _, _, N = b.shape

    output = torch.empty((B, M, N), dtype=a.dtype, device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)
    triton_matmul_split_k_kernel[grid](a, b, output, M, N, K, *a.stride(), *b.stride(), *output.stride())
    return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4,
                      num_warps=4)
    ],
    key=["M", "N", "K"]
)
@triton.jit
def triton_matmul_split_k_kernel(
    a_ptr, b_ptr, o_ptr, M, N, K,
    a_stride_b, a_stride_m, a_stride_k,
    b_stride_b, b_stride_k, b_stride_n,
    o_stride_b, o_stride_m, o_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    a: [B, M, K]
    b: [B, K, N]
    o: [B, M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    a_ptr += pid_b * a_stride_b + pid_m * BLOCK_M * a_stride_m
    b_ptr += pid_b * b_stride_b + pid_n * BLOCK_N * b_stride_n
    o_ptr += pid_b * o_stride_b + pid_m * BLOCK_M * o_stride_m + pid_n * BLOCK_N * o_stride_n
    
    offset_m = tl.arange(0, BLOCK_M)
    mask_m = offset_m < (M - pid_m * BLOCK_M)
    offset_n = tl.arange(0, BLOCK_N)
    mask_n = offset_n < (N - pid_n * BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offset_k < K
        a = tl.load(a_ptr + a_stride_m * offset_m[:, None] + a_stride_k * offset_k[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptr + b_stride_k * offset_k[:, None] + b_stride_n * offset_n[None, :], mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        offset_k += BLOCK_K

    tl.store(o_ptr + o_stride_m * offset_m[:, None] + o_stride_n * offset_n[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul_split_k_group(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [B, M, K]
    b: [B, K, N]
    return: [B, M, N]
    """
    assert len(a.shape) == 3 and len(b.shape) == 3
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[1]
    B, M, K = a.shape
    _, _, N = b.shape

    output = torch.empty((B, M, N), dtype=a.dtype, device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), B)
    triton_matmul_split_k_group_kernel[grid](a, b, output, M, N, K, *a.stride(), *b.stride(), *output.stride())
    return output

# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 4}, num_stages=3, num_warps=8),
#         triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 4}, num_stages=3, num_warps=8),
#         triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4}, num_stages=3, num_warps=8),
#         triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
#     ],
#     key=["M", "N", "K"]
# )
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4)
    ],
    key=["M", "N", "K"]
)
@triton.jit
def triton_matmul_split_k_group_kernel(
    a_ptr, b_ptr, o_ptr, M, N, K,
    a_stride_b, a_stride_m, a_stride_k,
    b_stride_b, b_stride_k, b_stride_n,
    o_stride_b, o_stride_m, o_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr
):
    pid_b = tl.program_id(1)
    a_ptr += pid_b * a_stride_b
    b_ptr += pid_b * b_stride_b
    o_ptr += pid_b * o_stride_b
    pid = tl.program_id(0)
    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)
    num_block_group = GROUP_M * num_block_n
    group_id = num_block_n // num_block_group
    first_group_m = group_id * num_block_group

    num_block_m_in_group = min(GROUP_M, num_block_m - first_group_m)
    block_n = (pid % num_block_group) // num_block_m_in_group
    block_m = (pid % num_block_group) % num_block_m_in_group + first_group_m

    a_ptr += block_m * BLOCK_M * a_stride_m
    b_ptr += block_n * BLOCK_N * b_stride_n
    o_ptr += block_m * BLOCK_M * o_stride_m + block_n * BLOCK_N * o_stride_n
    
    offset_m = tl.arange(0, BLOCK_M)
    mask_m = offset_m < (M - block_m * BLOCK_M)
    offset_k = tl.arange(0, BLOCK_K)
    offset_n = tl.arange(0, BLOCK_N)
    mask_n = offset_n < (N - block_n * BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offset_k < K
        a = tl.load(a_ptr + a_stride_m * offset_m[:, None] + a_stride_k * offset_k[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptr + b_stride_k * offset_k[:, None] + b_stride_n * offset_n[None, :], mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        offset_k += BLOCK_K
    tl.store(o_ptr + o_stride_m * offset_m[:, None] + o_stride_n * offset_n[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])


DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_correctness():
    B, M, N, K = 2, 64, 48, 32
    a = torch.randn(B, M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(B, K, N, device=DEVICE, dtype=torch.float16)
    ref = pytorch_matmul(a, b)
    out_split_k = triton_matmul_split_k(a, b)
    out_split_k_group = triton_matmul_split_k_group(a, b)
    if torch.allclose(ref, out_split_k, atol=1e-3, rtol=1e-3):
        print("Triton split-k correctness check passed!")
    else:
        print("Triton split-k correctness check failed!")
    print("Max diff:", torch.max(torch.abs(ref - out_split_k)))

    if torch.allclose(ref, out_split_k_group, atol=1e-3, rtol=1e-3):
        print("Triton split-k-group correctness check passed!")
    else:
        print("Triton split-k-group correctness check failed!")
    print("Max diff:", torch.max(torch.abs(ref - out_split_k_group)))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["mnk"],
        x_vals=[2**i for i in range(5, 13)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "triton-split-k", "triton-split-k-group"],
        line_names=["Torch", "Triton-split-k", "Triton-split-k-group"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="TFLOPS",
        plot_name="batched-matmul-performance",
        args={"B": 4},
    )
)
def benchmark(B, mnk, provider):
    M = N = K = mnk
    a = torch.randn(B, M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(B, K, N, device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: pytorch_matmul(a, b), quantiles=quantiles
        )
    elif provider == "triton-split-k":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_matmul_split_k(a, b), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_matmul_split_k_group(a, b), quantiles=quantiles
        )
    flops = 2.0 * B * M * N * K
    tflops = lambda t: flops / (t * 1e-3) / 1e12
    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    test_correctness()
    benchmark.run(print_data=True, show_plots=True)
