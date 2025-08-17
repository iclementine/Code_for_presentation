import triton
from triton import language as tl
import torch 

def chained_sum(x):
    n = x.numel()
    out = torch.empty((), dtype=x.dtype, device=x.device)
    grid =(1, 1, 1)
    TILE_SIZE = 4096
    num_warps = 8
    chained_reduce_1d[grid](x, out, n, TILE_SIZE, num_warps=num_warps)
    return out


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@triton.jit
def chained_reduce_1d(
    input_ptr,
    out_ptr,
    reduction_size,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    prev_multiple = prev_multiple_of(reduction_size, TILE_SIZE)
    acc = tl.zeros((TILE_SIZE,), dtype=input_ptr.type.element_ty)
    # coarse thread
    for n_start in tl.range(0, prev_multiple, TILE_SIZE, num_stages=1):
        n_offsets = n_start + tl.arange(0, TILE_SIZE)
        input = tl.load(input_ptr + n_offsets)
        acc += input
    # loop-peeling
    n_offsets = prev_multiple + tl.arange(0, TILE_SIZE)
    input = tl.load(
        input_ptr + n_offsets, mask=n_offsets < reduction_size
    )
    acc += input
    out = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid, out)


def full_reduction1(x, dtype=None):
    """
    per-block-state & flag, CTA-0 poll and adds the local sum
    """
    out_dtype = dtype or x.dtype
    out_dtype = torch.float32 if out_dtype == torch.bfloat16 else out_dtype
    out = torch.empty((), dtype=out_dtype, device=x.device)
    TILE_SIZE = 4096
    num_warps = 8
    N = x.numel()
    num_tiles = triton.cdiv(N, TILE_SIZE)
    max_ctas = 4096 # configurable
    num_ctas = min(num_tiles, max_ctas)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    grid = (num_ctas, 1, 1)
    # print(f"{grid=}")
    # print(f"{tiles_per_cta=}, {TILE_SIZE=}")
    partial_results = torch.empty((num_ctas,), dtype=out_dtype, device=x.device)
    flags = torch.zeros((num_ctas,), dtype=torch.int8, device=x.device)

    parallel_sum_per_cta_flag[grid](x, out, partial_results, flags, N, TILE_SIZE, tiles_per_cta, num_warps=num_warps)
    # print(f"debug: {torch.sum(partial_results)}")
    return out

@triton.jit
def parallel_sum_per_cta_flag(x_ptr, out_ptr, partial_results_ptr, flags_ptr, N, TILE_SIZE: tl.constexpr, TILES_PER_CTA: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    block_offset = pid * (TILES_PER_CTA * TILE_SIZE)
    block_end = min(block_offset + TILES_PER_CTA * TILE_SIZE, N)
    acc = tl.zeros((TILE_SIZE,), dtype=tl.float32)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        x = tl.load(x_ptr + offsets, mask=offsets < N).to(tl.float32)
        acc += x
    block_sum = tl.sum(acc, 0)
    tl.store(partial_results_ptr + pid, block_sum, cache_modifier=".cg")
    tl.store(flags_ptr + pid, 1, cache_modifier=".cg")

    if pid == 0:
        num_pids = tl.num_programs(0)
        global_sum = block_sum
        idx = 1
        while idx < num_pids:
            while not tl.load(flags_ptr + idx, volatile=True).to(tl.int1):
                pass
            global_sum += tl.load(partial_results_ptr + idx, volatile=True)
            idx += 1
        tl.store(out_ptr, global_sum)


def full_reduction2(x, dtype=None):
    """
    Atomic add results.
    """
    out_dtype = dtype or x.dtype
    out_dtype = torch.float32 if out_dtype == torch.bfloat16 else out_dtype
    out = torch.zeros((), dtype=out_dtype, device=x.device)
    TILE_SIZE = 4096
    num_warps = 8
    N = x.numel()
    num_tiles = triton.cdiv(N, TILE_SIZE)
    max_ctas = 82 * 4 # configurable
    num_ctas = min(num_tiles, max_ctas)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    grid = (num_ctas, 1, 1)
    # print(f"{grid=}")
    # print(f"{tiles_per_cta=}, {TILE_SIZE=}")
    # partial_results = torch.empty((num_ctas,), dtype=out_dtype, device=x.device)
    # flags = torch.zeros((num_ctas,), dtype=torch.int8, device=x.device)

    parallel_sum_atomic[grid](x, out, N, TILE_SIZE, tiles_per_cta, num_warps=num_warps)
    # print(f"debug: {torch.sum(partial_results)}")
    return out.to(x.dtype)

@triton.jit
def parallel_sum_atomic(x_ptr, out_ptr, N,  TILE_SIZE: tl.constexpr, TILES_PER_CTA: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    block_offset = pid * (TILES_PER_CTA * TILE_SIZE)
    block_end = min(block_offset + TILES_PER_CTA * TILE_SIZE, N)
    acc = tl.zeros((TILE_SIZE,), dtype=tl.float32)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        x = tl.load(x_ptr + offsets, mask=offsets < N).to(tl.float32)
        acc += x
    block_sum = tl.sum(acc, 0)
    tl.atomic_add(out_ptr, block_sum, sem="relaxed")


def full_reduction3(x, dtype=None):
    """
    a global counter for each CTA to add on
    local sums are then summed by CTA 0.
    """
    out_dtype = dtype or x.dtype
    out_dtype = torch.float32 if out_dtype == torch.bfloat16 else out_dtype
    out = torch.zeros((), dtype=out_dtype, device=x.device)
    TILE_SIZE = 4096
    num_warps = 8
    N = x.numel()
    num_tiles = triton.cdiv(N, TILE_SIZE)
    max_ctas = 82 * 4 # configurable
    TILE_SIZE2 = triton.next_power_of_2(max_ctas)
    num_ctas = min(num_tiles, max_ctas)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    grid = (num_ctas, 1, 1)
    # print(f"{grid=}")
    # print(f"{tiles_per_cta=}, {TILE_SIZE=}")
    partial_results = torch.empty((num_ctas,), dtype=out_dtype, device=x.device)
    flags = torch.zeros((), dtype=torch.int32, device=x.device)

    parallel_sum_global_flag[grid](x, out, partial_results, flags, N, TILE_SIZE, tiles_per_cta, TILE_SIZE2, num_warps=num_warps)
    # print(f"debug: {torch.sum(partial_results)}")
    return out

@triton.jit
def parallel_sum_global_flag(x_ptr, out_ptr, partial_results_ptr, flags_ptr, N, TILE_SIZE: tl.constexpr, TILES_PER_CTA: tl.constexpr, TILE_SIZE2: tl.constexpr,):
    pid = tl.program_id(0).to(tl.int64)
    block_offset = pid * (TILES_PER_CTA * TILE_SIZE)
    block_end = min(block_offset + TILES_PER_CTA * TILE_SIZE, N)
    acc = tl.zeros((TILE_SIZE,), dtype=tl.float32)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        x = tl.load(x_ptr + offsets, mask=offsets < N).to(tl.float32)
        acc += x
    block_sum = tl.sum(acc, 0)
    tl.store(partial_results_ptr + pid, block_sum, cache_modifier=".cg")
    tl.atomic_add(flags_ptr, 1, sem="release")

    if pid == 0:
        num_pids = tl.num_programs(0)
        while tl.atomic_add(flags_ptr, 0, sem="acquire") != num_pids:
            pass
        offsets = tl.arange(0, TILE_SIZE2)
        v = tl.load(partial_results_ptr + offsets, mask=offsets < num_pids)
        global_sum = tl.sum(v, 0)
        tl.store(out_ptr, global_sum)

def full_reduction4(x, dtype=None):
    """
    2 kernels
    """
    out_dtype = dtype or x.dtype
    out_dtype = torch.float32 if out_dtype == torch.bfloat16 else out_dtype
    out = torch.zeros((), dtype=out_dtype, device=x.device)
    TILE_SIZE = 4096
    num_warps = 8
    N = x.numel()
    num_tiles = triton.cdiv(N, TILE_SIZE)
    max_ctas = 82 * 4 # configurable
    TILE_SIZE2 = triton.next_power_of_2(max_ctas)
    num_ctas = min(num_tiles, max_ctas)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    grid = (num_ctas, 1, 1)
    # print(f"{grid=}")
    # print(f"{tiles_per_cta=}, {TILE_SIZE=}")
    partial_results = torch.empty((num_ctas,), dtype=out_dtype, device=x.device)

    parallel_sum_local_sum[grid](x, partial_results, N, TILE_SIZE, tiles_per_cta, num_warps=num_warps)
    parallel_sum_combine[(1, 1, 1)](partial_results, out, num_ctas, TILE_SIZE2, num_warps=num_warps)
    # print(f"debug: {torch.sum(partial_results)}")
    return out

@triton.jit
def parallel_sum_local_sum(x_ptr, partial_results_ptr, N,  TILE_SIZE: tl.constexpr, TILES_PER_CTA: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    block_offset = pid * (TILES_PER_CTA * TILE_SIZE)
    block_end = min(block_offset + TILES_PER_CTA * TILE_SIZE, N)
    acc = tl.zeros((TILE_SIZE,), dtype=tl.float32)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        x = tl.load(x_ptr + offsets, mask=offsets < N).to(tl.float32)
        acc += x
    block_sum = tl.sum(acc, 0)
    tl.store(partial_results_ptr + pid, block_sum, cache_modifier=".cg")

@triton.jit
def parallel_sum_combine(partial_results_ptr, out_ptr, N, TILE_SIZE: tl.constexpr):
    offsets = tl.arange(0, TILE_SIZE)
    x = tl.load(partial_results_ptr + offsets, mask=offsets < N)
    out = tl.sum(x, 0)
    tl.store(out_ptr, out)

   

if __name__ == "__main__":
    import flag_gems
    x = torch.randn(1024 * 1024, dtype=torch.float32, device="cuda")
    out = torch.sum(x)
    out2 = flag_gems.sum(x)
    out3 = full_reduction1(x)
    out4 = full_reduction2(x)
    out5 = full_reduction3(x)
    out6 = full_reduction4(x)
    out7 = chained_sum(x)
    print(out)
    print(out2)
    print(out3)
    print(out4)
    print(out5)
    print(out6)
    print(out7)

    io_amount = x.numel() * x.itemsize
    t1 = triton.testing.do_bench(lambda: torch.sum(x))
    t2 = triton.testing.do_bench(lambda: flag_gems.sum(x))
    t3 = triton.testing.do_bench(lambda: full_reduction1(x))
    t4 = triton.testing.do_bench(lambda: full_reduction2(x))
    t5 = triton.testing.do_bench(lambda: full_reduction3(x))
    t6 = triton.testing.do_bench(lambda: full_reduction4(x))
    t7 = triton.testing.do_bench(lambda: chained_sum(x))
    
    t1 = io_amount * 1e-9 / (t1 * 1e-3)
    t2 = io_amount * 1e-9 / (t2 * 1e-3)
    t3 = io_amount * 1e-9 / (t3 * 1e-3)
    t4 = io_amount * 1e-9 / (t4 * 1e-3)
    t5 = io_amount * 1e-9 / (t5 * 1e-3)
    t6 = io_amount * 1e-9 / (t6 * 1e-3)
    t7 = io_amount * 1e-9 / (t7 * 1e-3)
    print(f"torch: {t1:.3f} GB/s")
    print(f"flag_gems: {t2:.3f} GB/s")
    print(f"full_reduction1 [per-cta-flag]: {t3:.3f} GB/s")
    print(f"full_reduction2 [atomic_add]: {t4:.3f} GB/s")
    print(f"full_reduction3 [global-flag]: {t5:.3f} GB/s")
    print(f"full_reduction3 [2-kernel]: {t6:.3f} GB/s")
    print(f"chained reduction: {t7:.3f} GB/s")

    



    
    