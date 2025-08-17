import triton
from triton import language as tl
import torch


def chained_sum(x, TILE_SIZE, num_warps):
    m, n = x.shape
    out = torch.empty((m,), dtype=x.dtype, device=x.device)
    max_grid_dim_x = 2**31 - 1
    if m <= max_grid_dim_x:
        chained_reduce_1d[grid](x, out, m, n, n, TILE_SIZE, num_warps=num_warps)
    else:
        # gsl kernel
        grid = (max_grid_dim_x, 1, 1)
        chained_reduce_1d_gsl[grid](x, out, m, n, n, TILE_SIZE, num_warps=num_warps)
    return out


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@triton.jit
def chained_reduce_1d(
    input_ptr,
    out_ptr,
    batch_size,
    batch_stride,
    reduction_size,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    prev_multiple = prev_multiple_of(reduction_size, TILE_SIZE)
    acc = tl.zeros((TILE_SIZE,), dtype=input_ptr.type.element_ty)
    # coarse thread
    for n_start in tl.range(0, prev_multiple, TILE_SIZE, num_stages=1):
        n_offsets = n_start + tl.arange(0, TILE_SIZE)
        input = tl.load(input_ptr + pid * batch_stride + n_offsets)
        acc += input
    # loop-peeling
    n_offsets = prev_multiple + tl.arange(0, TILE_SIZE)
    input = tl.load(
        input_ptr + pid * batch_stride + n_offsets, mask=n_offsets < reduction_size
    )
    acc += input
    out = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid, out)


@triton.jit
def chained_reduce_1d_gsl(
    input_ptr,
    out_ptr,
    batch_size,
    batch_stride,
    reduction_size,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    grid = tl.num_programs(0).to(tl.int64)
    prev_multiple = prev_multiple_of(reduction_size, TILE_SIZE)

    # gsl
    for m_id in tl.range(pid, batch_size, grid):
        acc = tl.zeros((TILE_SIZE,), dtype=input_ptr.type.element_ty)
        # coarse thread
        for n_start in tl.range(0, prev_multiple, TILE_SIZE, num_stages=1):
            n_offsets = n_start + tl.arange(0, TILE_SIZE)
            input = tl.load(input_ptr + m_id * batch_stride + n_offsets)
            acc += input
        # loop-peeling
        n_offsets = prev_multiple + tl.arange(0, TILE_SIZE)
        input = tl.load(
            input_ptr + m_id * batch_stride + n_offsets, mask=n_offsets < reduction_size
        )
        acc += input
        out = tl.sum(acc, axis=0)
        tl.store(out_ptr + m_id, out)


# ------------------------------------ test ------------------------------------
def test():
    m, n = 4096, 512 * 1024
    x = torch.randn((m, n), device="cuda:0")
    ref = torch.sum(x, dim=-1)

    for TILE_SIZE in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        for num_warps in (2, 4, 8, 16, 32):
            hyp = chained_sum(x, TILE_SIZE, num_warps)
            torch.testing.assert_close(hyp, ref)


# ------------------------------------ benchmark ------------------------------------
def benchmark():
    m, n = 4096, 32 * 1024
    x = torch.randn((m, n), device="cuda:0")
    ref = torch.sum(x, dim=-1)

    io_amount = x.numel() * x.itemsize
    for TILE_SIZE in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        if n < TILE_SIZE:
            continue
        for num_warps in (2, 4, 8, 16, 32):
            if TILE_SIZE < num_warps * 32:
                continue
            t = triton.testing.do_bench(
                lambda: chained_sum(x, TILE_SIZE, num_warps),
                return_mode="median",
            )
            throughput = (io_amount * 1e-9) / (t * 1e-3)
            unroll = TILE_SIZE / (num_warps * 32)
            print(
                f"TILE_SIZE: {TILE_SIZE}, num_warps: {num_warps}, unroll: {unroll}, throughput: {throughput:.3f} GB/s"
            )


if __name__ == "__main__":
    # test()
    benchmark()
