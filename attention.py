import torch
from torch import nn
import triton
import triton.language as tl
import math

def triton_attention(q, k, v, scale=None, is_causal=False):
    """
    q: [b, n, s, d]
    k: [b, n, s, d]
    v: [b, n, s, d]
    return: [b, n, s, d]
    """
    assert q.shape == k.shape == v.shape
    b, n, s, d = q.shape
    o = torch.empty_like(q)
    scale = scale or 1.0 / math.sqrt(d)
    grid = lambda meta: (math.ceil(s / meta["b_r"]), b * n)
    triton_flash_attn_kernel[grid](
        q, k, v, o, scale, is_causal,
        b, n, s, d,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
    )

    # torch.cuda.synchronize()
    # torch.cuda.cudart().cudaProfilerStart()
    # triton_flash_attn_kernel[grid](
    #     q, k, v, o, scale, is_causal,
    #     b, n, s, d,
    #     *q.stride(), *k.stride(), *v.stride(), *o.stride(),
    # )
    # torch.cuda.synchronize()
    # torch.cuda.cudart().cudaProfilerStop()

    return o

@triton.autotune(
    configs=[
        # triton.Config({"b_r": 128, "b_c": 64}, num_stages=1, num_warps=4),
        triton.Config({"b_r": 128, "b_c": 64}, num_stages=1, num_warps=8),
        # triton.Config({"b_r": 64, "b_c": 64}, num_stages=1, num_warps=4),
        # triton.Config({"b_r": 64, "b_c": 64}, num_stages=1, num_warps=8),
        # triton.Config({"b_r": 32, "b_c": 64}, num_stages=1, num_warps=4),
    ],
    key=["b", "s", "n", "d"]
)
@triton.heuristics({
    "DIM": lambda args: triton.next_power_of_2(args["d"]),
    "EVEN_M": lambda args: (args["s"] % args["b_r"] == 0),
    "EVEN_N": lambda args: (args["s"] % args["b_c"] == 0),
    "EVEN_D": lambda args: (triton.next_power_of_2(args["d"]) == args["d"]),
})
@triton.jit
def triton_flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, scale, is_causal,
    b, n, s, d,
    q_b_stride, q_n_stride, q_s_stride, q_d_stride,
    k_b_stride, k_n_stride, k_s_stride, k_d_stride,
    v_b_stride, v_n_stride, v_s_stride, v_d_stride,
    o_b_stride, o_n_stride, o_s_stride, o_d_stride,
    b_r: tl.constexpr,  # block size for sequence dimension
    b_c: tl.constexpr,  # block size for head dimension
    DIM: tl.constexpr,
    EVEN_M: tl.constexpr,  # whether sequence length is divisible by b_r
    EVEN_N: tl.constexpr,  # whether head dimension is divisible by b_c
    EVEN_D: tl.constexpr,  # whether feature dimension is a power of 2
):
    b_id = tl.program_id(1) // (n)
    n_id = tl.program_id(1) % (n)
    q_ptr = q_ptr + b_id * q_b_stride + n_id * q_n_stride
    k_ptr = k_ptr + b_id * k_b_stride + n_id * k_n_stride
    v_ptr = v_ptr + b_id * v_b_stride + n_id * v_n_stride
    o_ptr = o_ptr + b_id * o_b_stride + n_id * o_n_stride

    pid = tl.program_id(0)

    q_ptr += pid * b_r * q_s_stride
    o_ptr += pid * b_r * o_s_stride
    offset_r = tl.arange(0, b_r)
    mask_r = (pid * b_r + offset_r) < s

    offset_c = tl.arange(0, b_c)
    
    offset_d = tl.arange(0, DIM)
    mask_d = offset_d < d
    mi = tl.zeros((b_r,), dtype=tl.float32) - float("inf")
    li = tl.zeros((b_r,), dtype=tl.float32)
    acc = tl.zeros((b_r, DIM), dtype=tl.float32)
    if EVEN_M:
        if EVEN_D:
            q = tl.load(q_ptr + q_s_stride * offset_r[:, None] + q_d_stride * offset_d[None, :])
        else:
            q = tl.load(q_ptr + q_s_stride * offset_r[:, None] + q_d_stride * offset_d[None, :], mask=mask_d[None, :], other=0.0)
    else:
        if EVEN_D:
            q = tl.load(q_ptr + q_s_stride * offset_r[:, None] + q_d_stride * offset_d[None, :], mask=mask_r[:, None], other=0.0)
        else:
            q = tl.load(q_ptr + q_s_stride * offset_r[:, None] + q_d_stride * offset_d[None, :], mask=mask_r[:, None] & mask_d[None, :], other=0.0)

    q = q * scale.to(tl.float16)
    # q = tl.load(q_ptr + q_s_stride * offset_r[:, None] + q_d_stride * offset_d[None, :], mask=mask_r[:, None] & mask_d[None, :], other=0.0)
    log2e: tl.constexpr = 1.44269504089
    for _ in range(0, tl.cdiv(s, b_c)):
        mask_c = offset_c < s
        if EVEN_N:
            if EVEN_D:
                # k = tl.load(k_ptr + k_s_stride * offset_c[:, None] + k_d_stride * offset_d[None, :])
                kt = tl.load(k_ptr + k_d_stride * offset_d[:, None] + k_s_stride * offset_c[None, :])
            else:
                # k = tl.load(k_ptr + k_s_stride * offset_c[:, None] + k_d_stride * offset_d[None, :], mask=mask_d[None, :], other=0.0)
                kt = tl.load(k_ptr + k_d_stride * offset_d[:, None] + k_s_stride * offset_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        else:
            if EVEN_D:
                # k = tl.load(k_ptr + k_s_stride * offset_c[:, None] + k_d_stride * offset_d[None, :], mask=mask_c[:, None], other=0.0)
                kt = tl.load(k_ptr + k_d_stride * offset_d[:, None] + k_s_stride * offset_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
            else:
                # k = tl.load(k_ptr + k_s_stride * offset_c[:, None] + k_d_stride * offset_d[None, :], mask=mask_c[:, None] & mask_d[None, :], other=0.0)
                kt = tl.load(k_ptr + k_d_stride * offset_d[:, None] + k_s_stride * offset_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
    
        # k = tl.load(k_ptr + k_s_stride * offset_c[:, None] + k_d_stride * offset_d[None, :], mask=mask_c[:, None] & mask_d[None, :], other=0.0)
        # score = tl.dot(q, tl.trans(k))
        score = tl.dot(q, kt)
        if is_causal:
            score += tl.where((pid * b_r + offset_r)[:, None] < offset_c[None, :], float("-inf"), 0)
        
        m_new = tl.maximum(mi, tl.max(score, axis=-1))
        p = tl.exp2((score - m_new[:, None]) * log2e)
        out_scale = tl.exp2((mi - m_new) * log2e)
        # p = tl.exp(score - m_new[:, None])
        # out_scale = tl.exp(mi - m_new)
        li = li * out_scale + tl.sum(p, axis=-1)

        if EVEN_N:
            if EVEN_D:
                v = tl.load(v_ptr + v_s_stride * offset_c[:, None] + v_d_stride * offset_d[None, :])
            else:
                v = tl.load(v_ptr + v_s_stride * offset_c[:, None] + v_d_stride * offset_d[None, :], mask=mask_d[None, :], other=0.0)
        else:
            if EVEN_D:
                v = tl.load(v_ptr + v_s_stride * offset_c[:, None] + v_d_stride * offset_d[None, :], mask=mask_c[:, None], other=0.0)
            else:
                v = tl.load(v_ptr + v_s_stride * offset_c[:, None] + v_d_stride * offset_d[None, :], mask=mask_c[:, None] & mask_d[None, :], other=0.0)
        # v = tl.load(v_ptr + v_s_stride * offset_c[:, None] + v_d_stride * offset_d[None, :], mask=mask_c[:, None] & mask_d[None, :], other=0.0)
        acc = acc * out_scale[:, None] + tl.dot(p.to(tl.float16), v)
        mi = m_new
        offset_c += b_c

    acc = acc / li[:, None]
    if EVEN_M:
        if EVEN_D:
            tl.store(o_ptr + o_s_stride * offset_r[:, None] + o_d_stride * offset_d[None, :], acc.to(tl.float16))
        else:
            tl.store(o_ptr + o_s_stride * offset_r[:, None] + o_d_stride * offset_d[None, :], acc.to(tl.float16), mask=mask_d[None, :])
    else:
        if EVEN_D:
            tl.store(o_ptr + o_s_stride * offset_r[:, None] + o_d_stride * offset_d[None, :], acc.to(tl.float16), mask=mask_r[:, None])
        else:
            tl.store(o_ptr + o_s_stride * offset_r[:, None] + o_d_stride * offset_d[None, :], acc.to(tl.float16), mask=mask_r[:, None] & mask_d[None, :])
    # tl.store(o_ptr + o_s_stride * offset_r[:, None] + o_d_stride * offset_d[None, :], acc.to(tl.float16), mask=mask_r[:, None] & mask_d[None, :])


DEVICE = triton.runtime.driver.active.get_active_torch_device()

@torch.no_grad()
def pytorch_attention(q, k, v, scale=None, is_causal=False):
    """
    q: [b, n, s, d]
    k: [b, n, s, d]
    v: [b, n, s, d]
    return: [b, n, s, d]
    """
    b, n, s, d = q.shape
    scale = scale or 1.0 / math.sqrt(d)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=is_causal,
    )
    return out


def test_correctness():
    b, n, s, d = 4, 32, 2048, 64
    q = torch.randn(b, n, s, d, device=DEVICE, dtype=torch.float16)
    k = torch.randn(b, n, s, d, device=DEVICE, dtype=torch.float16)
    v = torch.randn(b, n, s, d, device=DEVICE, dtype=torch.float16)
    # ref = pytorch_attention(q, k, v)
    out = triton_attention(q, k, v)
    # max_diff = torch.max(torch.abs(ref - out)).item()
    # print(f"Max diff: {max_diff}")
    # assert torch.allclose(ref, out, atol=1e-2, rtol=1e-2), \
    #     f"Correctness check FAILED! max diff: {max_diff}"
    # print("Correctness check passed!")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(5, 14)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="flash-attention-performance",
        args={"b": 4, "n": 32, "d": 64},
    )
)
def benchmark(b, n, d, seq_len, provider):
    q = torch.randn(b, n, seq_len, d, device=DEVICE, dtype=torch.float16)
    k = torch.randn(b, n, seq_len, d, device=DEVICE, dtype=torch.float16)
    v = torch.randn(b, n, seq_len, d, device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: pytorch_attention(q, k, v), quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_attention(q, k, v), quantiles=quantiles,
        )
    flops = 4.0 * b * n * seq_len * seq_len * d
    tflops = lambda t: flops / (t * 1e-3) / 1e12
    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    # test_correctness()
    benchmark.run(print_data=True, show_plots=True)