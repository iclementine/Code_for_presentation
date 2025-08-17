import triton
from triton import language as tl
import torch

def persistent_sum(x, num_warps):
    m, n = x.shape
    out = torch.empty((m,), dtype=x.dtype, device=x.device)
    # assert m <= 2147483647

    TILE_SIZE = triton.next_power_of_2(n)
    TILE_M = max(1, (num_warps * 32 * 4) // TILE_SIZE)
    num_tasks = triton.cdiv(m, TILE_M)
    max_grid_dim_x = 2 ** 31 - 1
    if num_tasks <= max_grid_dim_x:
        grid = (num_tasks, 1, 1)
        persistent_reduce_2d[grid](x, out, m, n, n, TILE_M, TILE_SIZE, num_warps=num_warps)
    else:
        grid = (max_grid_dim_x, 1, 1)
        persistent_reduce_2d_gsl[grid](x, out, m, n, n, TILE_M, TILE_SIZE, num_warps=num_warps)
    return out

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