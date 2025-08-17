import triton
from triton import language as tl
import torch
# import flag_gems


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


# ------------------------ 1d persistent reduction kernel ------------------------
def persistent_sum_2d(x, num_warps):
    m, n = x.shape
    out = torch.empty((m,), dtype=x.dtype, device=x.device)
    # assert m <= 2147483647

    TILE_SIZE = triton.next_power_of_2(n)
    TILE_M = max(1, (num_warps * 32 * 4) // TILE_SIZE)

    grid_x = triton.cdiv(m, TILE_M)
    grid = (grid_x, 1, 1)
    # assert (TILE_M * TILE_SIZE) <= 128k or 1024k
    persistent_reduce_2d[grid](x, out, m, n, n, TILE_M, TILE_SIZE, num_warps=num_warps)
    return out


@triton.jit
def persistent_reduce_2d(
    input_ptr,
    out_ptr,
    batch_size,
    batch_stride,
    reduction_size,
    TILE_M: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, TILE_SIZE)
    mask = (m_offsets < batch_size)[:, None] & (n_offsets < reduction_size)[None, :]

    input = tl.load(
        input_ptr + m_offsets[:, None] * batch_stride + n_offsets[None, :],
        mask=mask,
    )
    out = tl.sum(input, axis=1)
    tl.store(out_ptr + m_offsets, out)


# ------------------------ 2d persistent reduction kernel gsl ------------------------
def persistent_sum_2d_gsl(x, num_warps):
    m, n = x.shape
    out = torch.empty((m,), dtype=x.dtype, device=x.device)
    # assert m <= 2147483647

    TILE_SIZE = triton.next_power_of_2(n)
    TILE_M = max(1, (num_warps * 32 * 4) // TILE_SIZE)

    occupancy = min(32, 48 // num_warps)  # reg and shm not considered here
    grid_x = min(triton.cdiv(m, TILE_M), 82 * occupancy)
    grid = (grid_x, 1, 1)
    # assert (TILE_M * TILE_SIZE) <= 128k or 1024k
    persistent_reduce_2d_gsl[grid](x, out, m, n, n, TILE_M, TILE_SIZE, num_warps=num_warps)
    return out


@triton.jit
def persistent_reduce_2d_gsl(
    input_ptr,
    out_ptr,
    batch_size,
    batch_stride,
    reduction_size,
    TILE_M: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    grid = tl.num_programs(0).to(tl.int64)

    m_start = pid * TILE_M
    for m_id in range(m_start, batch_size, grid * TILE_M):
        m_offsets = m_id + tl.arange(0, TILE_M)
        n_offsets = tl.arange(0, TILE_SIZE)
        mask = (m_offsets < batch_size)[:, None] & (n_offsets < reduction_size)[None, :]
        input = tl.load(
            input_ptr + m_offsets[:, None] * batch_stride + n_offsets[None, :],
            mask=mask,
        )
        out = tl.sum(input, axis=1)
        tl.store(out_ptr + m_offsets, out)

# ------------------------ 1d bsl persistent reduction kernel ------------------------
def persistent_sum_1d_bsl(x, num_warps):
    m, n = x.shape
    out = torch.empty((m,), dtype=x.dtype, device=x.device)
    # assert m <= 2147483647

    TILE_SIZE = triton.next_power_of_2(n)
    TILE_M = max(1, (num_warps * 32 * 4) // TILE_SIZE)

    grid_x = triton.cdiv(m, TILE_M)
    grid = (grid_x, 1, 1)
    # assert (TILE_M * TILE_SIZE) <= 128k or 1024k
    persistent_reduce_1d_bsl[grid](
        x, out, m, n, n, TILE_M, TILE_SIZE, num_warps=num_warps
    )
    return out


@triton.jit
def persistent_reduce_1d_bsl(
    input_ptr,
    out_ptr,
    batch_size,
    batch_stride,
    reduction_size,
    TILE_M: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    m_start = pid * TILE_M
    m_end = min(batch_size, m_start + TILE_M)
    n_offsets = tl.arange(0, TILE_SIZE)
    mask = n_offsets < reduction_size

    for m_id in tl.range(m_start, m_end, 1, loop_unroll_factor=TILE_M):
        input = tl.load(input_ptr + m_id * batch_stride + n_offsets, mask=mask)
        out = tl.sum(input, axis=0)
        tl.store(out_ptr + m_id, out)


# ------------------------ 1d persistent reduction kernel ------------------------
def persistent_sum(x, num_warps):
    m, n = x.shape
    out = torch.empty((m,), dtype=x.dtype, device=x.device)
    # assert m <= 2147483647
    grid = (m, 1, 1)
    TILE_SIZE = triton.next_power_of_2(n)
    # assert TILE_SIZE <= 128k or 1024k
    persistent_reduce_1d[grid](x, out, m, n, n, TILE_SIZE, num_warps=num_warps)
    return out


@triton.jit
def persistent_reduce_1d(
    input_ptr,
    out_ptr,
    batch_size,
    batch_stride,
    reduction_size,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    n_offsets = tl.arange(0, TILE_SIZE)
    mask = n_offsets < reduction_size

    input = tl.load(input_ptr + pid * batch_stride + n_offsets, mask=mask)
    out = tl.sum(input, axis=0)
    tl.store(out_ptr + pid, out)


# ---------------------- 1d persistent reduction kernel with gsl ---------------------
def persistent_sum_gsl(x, num_warps):
    m, n = x.shape
    out = torch.empty((m,), dtype=x.dtype, device=x.device)
    occupancy = min(16, 48 // num_warps)  # reg and shm not considered here
    grid_x = min(m, 82 * occupancy)
    grid = (grid_x, 1, 1)
    TILE_SIZE = triton.next_power_of_2(n)
    # assert TILE_SIZE <= 128k or 1024k
    persistent_reduce_1d_gsl[grid](
        x, out, m, n, n, grid_x, TILE_SIZE, num_warps=num_warps
    )
    return out


@triton.jit
def persistent_reduce_1d_gsl(
    input_ptr,
    out_ptr,
    batch_size,
    batch_stride,
    reduction_size,
    grid_size,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    # loop_unroll_factor is not supported in triton 3.1
    n_offsets = tl.arange(0, TILE_SIZE)
    mask = n_offsets < reduction_size
    for m_id in tl.range(pid, batch_size, grid_size):
        input = tl.load(input_ptr + m_id * batch_stride + n_offsets, mask=mask)
        out = tl.sum(input, axis=0)
        tl.store(out_ptr + m_id, out)


# ----------------- chained-reduce_1d-gsl -----------------
def chained_sum_gsl(x, TILE_SIZE, num_warps):
    m, n = x.shape
    out = torch.empty((m,), dtype=x.dtype, device=x.device)
    occupancy = min(16, 48 // num_warps)  # reg and shm not considered here
    grid = (min(m, 82 * occupancy), 1, 1)
    chained_reduce_1d_gsl[grid](x, out, m, n, n, TILE_SIZE, num_warps=num_warps)
    return out


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
    grid_size = tl.num_programs(0).to(tl.int64)

    prev_multiple = prev_multiple_of(reduction_size, TILE_SIZE)
    # loop_unroll_factor is not supported in triton 3.1
    for m_id in tl.range(pid, batch_size, grid_size, num_stages=1):
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


def benchmark():
    m, n = 512 * 1024, 128
    x = torch.randn((m, n), device="cuda:0", dtype=torch.float32)
    ref = torch.sum(x, dim=-1)

    # persistent-2d
    for num_warps in (4, 8, 16, 32):
        tile_size = triton.next_power_of_2(n)
        hyp = persistent_sum_2d(x, num_warps)
        #torch.testing.assert_close(hyp, ref)
        t = triton.testing.do_bench(
            lambda: persistent_sum_2d(x, num_warps), return_mode="median"
        )
        b = x.element_size() * x.numel() * 1e-9 / (t * 1e-3)
        print(
            f"[persistent-2d] tile_size: {tile_size}, num_warps: {num_warps}, time: {t:.3f} ms, bandwidth: {b:.3f} GB/s"
        )
    print("==")

    # persistent 2d -gsl
    for num_warps in (4, 8, 16, 32):
        tile_size = triton.next_power_of_2(n)
        hyp = persistent_sum_2d_gsl(x, num_warps)
        #torch.testing.assert_close(hyp, ref)
        t = triton.testing.do_bench(
            lambda: persistent_sum_2d_gsl(x, num_warps), return_mode="median"
        )
        b = x.element_size() * x.numel() * 1e-9 / (t * 1e-3)
        print(
            f"[persistent-2d-gsl] tile_size: {tile_size}, num_warps: {num_warps}, time: {t:.3f} ms, bandwidth: {b:.3f} GB/s"
        )
    print("==")

    # persistent-1d-bsl
    for num_warps in (1, 2, 4, 8, 16, 32):
        tile_size = triton.next_power_of_2(n)
        hyp = persistent_sum_1d_bsl(x, num_warps)
        #torch.testing.assert_close(hyp, ref)
        t = triton.testing.do_bench(
            lambda: persistent_sum_1d_bsl(x, num_warps), return_mode="median"
        )
        b = x.element_size() * x.numel() * 1e-9 / (t * 1e-3)
        print(
            f"[persistent-1d-bsl] tile_size: {tile_size}, num_warps: {num_warps}, time: {t:.3f} ms, bandwidth: {b:.3f} GB/s"
        )
    print("==")

    # persistent
    for num_warps in (4, 8, 16, 32):
        tile_size = triton.next_power_of_2(n)
        hyp = persistent_sum(x, num_warps)
        #torch.testing.assert_close(hyp, ref)
        t = triton.testing.do_bench(
            lambda: persistent_sum(x, num_warps), return_mode="median"
        )
        b = x.element_size() * x.numel() * 1e-9 / (t * 1e-3)
        print(
            f"[persistent] tile_size: {tile_size}, num_warps: {num_warps}, time: {t:.3f} ms, bandwidth: {b:.3f} GB/s"
        )
    print("==")

    # persistent gsl
    for num_warps in (4, 8, 16, 32):
        tile_size = triton.next_power_of_2(n)
        hyp = persistent_sum_gsl(x, num_warps)
        #torch.testing.assert_close(hyp, ref)
        t = triton.testing.do_bench(
            lambda: persistent_sum_gsl(x, num_warps), return_mode="median"
        )
        b = x.element_size() * x.numel() * 1e-9 / (t * 1e-3)
        print(
            f"[persistent-gsl] tile_size: {tile_size}, num_warps: {num_warps}, time: {t:.3f} ms, bandwidth: {b:.3f} GB/s"
        )
    print("==")

    for tile_size in (128, 256, 512, 1024, 4096, 8192, 16384):
        if tile_size > n:
            continue
        for num_warps in (4, 8, 16, 32):
            t = triton.testing.do_bench(
                lambda: chained_sum_gsl(x, tile_size, num_warps), return_mode="median"
            )
            b = x.element_size() * x.numel() * 1e-9 / (t * 1e-3)
            print(
                f"[chained-gsl] tile_size: {tile_size}, num_warps: {num_warps}, time: {t:.3f} ms, bandwidth: {b:.3f} GB/s"
            )
        print("==")

    # t = triton.testing.do_bench(
    #     lambda: flag_gems.sum_dim(x, [-1]), return_mode="median"
    # )
    # b = x.element_size() * x.numel() * 1e-9 / (t * 1e-3)
    # print(f"[flag_gems] time: {t:.3f} ms, bandwidth: {b:.3f} GB/s")

    t = triton.testing.do_bench(lambda: torch.sum(x, dim=-1), return_mode="median")
    b = x.element_size() * x.numel() * 1e-9 / (t * 1e-3)
    print(f"[torch] time: {t:.3f} ms, bandwidth: {b:.3f} GB/s")


if __name__ == "__main__":
    benchmark()
