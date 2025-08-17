import triton
from triton import language as tl
import torch

# Conclusion: 1D tile is all-you-need in chained_reduction.
def chained_sum_2d(x, TILE_SIZE, num_warps):
    m, n = x.shape
    out = torch.empty((m,), dtype=x.dtype, device=x.device)
    TILE_M = triton.cdiv(8192, TILE_SIZE) # 
    grid = (triton.cdiv(m, TILE_M), 1, 1)
    chained_reduce_2d[grid](x, out, m, n, n, TILE_M, TILE_SIZE, num_warps=num_warps)
    return out


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@triton.jit
def chained_reduce_2d(
    input_ptr,
    out_ptr,
    batch_size,
    batch_stride,
    reduction_size,
    TILE_M: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    prev_multiple = prev_multiple_of(reduction_size, TILE_SIZE)
    acc = tl.zeros((TILE_M, TILE_SIZE,), dtype=input_ptr.type.element_ty)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    mask_m = m_offsets <= batch_size
    # coarse thread
    for n_start in tl.range(0, prev_multiple, TILE_SIZE, num_stages=1):
        n_offsets = n_start + tl.arange(0, TILE_SIZE)
        input = tl.load(
            input_ptr + m_offsets[:, None] * batch_stride + n_offsets[None, :],
            mask=mask_m[:, None],
        )
        acc += input
    # loop-peeling
    n_offsets = prev_multiple + tl.arange(0, TILE_SIZE)
    mask_n = n_offsets < reduction_size
    input = tl.load(
        input_ptr + m_offsets[:, None] * batch_stride + n_offsets[None, :], 
        mask=mask_m[:, None] & mask_n[None, :],
    )
    acc += input
    out = tl.sum(acc, axis=1)
    tl.store(out_ptr + m_offsets, out, mask=mask_m)


def test():
    m, n = 4096, 512 * 1024
    x = torch.randn((m, n), device="cuda:0")
    ref = torch.sum(x, dim=-1)

    for TILE_SIZE in (128, 256, 512, 1024, 2048, 4096):
        for num_warps in (2, 4, 8, 16, 32):
            hyp = chained_sum_2d(x, TILE_SIZE, num_warps)
            torch.testing.assert_close(hyp, ref)


def benchmark():
    m, n = 4096, 512 * 1024
    x = torch.randn((m, n), device="cuda:0")
    ref = torch.sum(x, dim=-1)

    io_amount = x.numel() * x.itemsize
    for num_warps in (2, 4, 8, 16, 32):
        for TILE_SIZE in (128, 256, 512, 1024, 2048, 4096, 8192):
            t = triton.testing.do_bench(
                lambda: chained_sum_2d(x, TILE_SIZE, num_warps),
                return_mode="median",
            )
            throughput = (io_amount * 1e-9) / (t * 1e-3)
            TILE_M = triton.cdiv(8192, TILE_SIZE)
            grid = triton.cdiv(m, TILE_M)
            wave_efficiency = (grid / 82) / triton.cdiv(grid, 82)
            print(
                f"TILE_SIZE: {TILE_SIZE}, num_warps: {num_warps}, wave_efficiency: {wave_efficiency:.3f}, throughput: {throughput:.3f} GB/s"
            )
        print()


if __name__ == "__main__":
    # test()
    benchmark()
