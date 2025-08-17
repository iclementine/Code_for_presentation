import triton
from triton import language as tl
import torch

# ------------------- Reduce then Scan (3 kernel implementation) -------------------
def reduce_then_scan(x):
    assert x.ndim == 1
    N = x.numel()
    TILE_SIZE = 4096
    num_warps = 8
    num_tiles = triton.cdiv(N, TILE_SIZE)
    max_ctas = 82 * 4
    num_ctas = min(num_tiles, max_ctas)
    ROOT_SCAN_TILE_SIZE = triton.next_power_of_2(num_ctas)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    block_sums = torch.empty((num_ctas,), dtype=x.dtype, device=x.device)
    block_inclusive_prefix = torch.empty((num_ctas,), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)

    # 3-kernel implementartion
    reduce_then_scan_block_sum_kernel[(num_ctas, 1, 1)](
        x, block_sums, N, tiles_per_cta, TILE_SIZE, num_warps=num_warps)
    reduce_then_scan_root_scan_kernel[(1, 1, 1)](
        block_sums, block_inclusive_prefix, num_ctas, ROOT_SCAN_TILE_SIZE, num_warps=num_warps)
    reduce_then_scan_block_scan_kernel[(num_ctas, 1, 1)](
        x, block_inclusive_prefix, out, N, tiles_per_cta, TILE_SIZE, num_warps=num_warps)
    return out

@triton.jit
def reduce_then_scan_block_sum_kernel(in_ptr, block_sum_ptr, N, tiles_per_cta, TILE_SIZE: tl.constexpr, ):
    """The same kernel as the block sum in parallel reduce"""
    pid = tl.program_id(0).to(tl.int64)
    block_offset = pid * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)
    acc = tl.zeros((TILE_SIZE,), dtype=in_ptr.type.element_ty)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        x = tl.load(in_ptr + offsets, mask=offsets < N)
        acc += x
    block_sum = tl.sum(acc, 0)
    tl.store(block_sum_ptr + pid, block_sum, cache_modifier=".cg")

@triton.jit
def reduce_then_scan_root_scan_kernel(in_ptr, out_ptr, N, TILE_SIZE: tl.constexpr):
    """Almost The same kernel as the persistent scan kernel"""
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out = tl.cumsum(x, 0)
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def reduce_then_scan_block_scan_kernel(in_ptr, previous_sum_ptr, out_ptr, N, tiles_per_cta, TILE_SIZE: tl.constexpr):
    # we can also loop here if you insist
    pid = tl.program_id(0).to(tl.int64)
    block_offset = pid * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)

    prefix = tl.load(previous_sum_ptr + pid - 1, mask=pid > 0, other=0)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        mask = offsets < N
        x = tl.load(in_ptr + offsets, mask=mask)
        tile_scan = prefix + tl.cumsum(x, 0)
        prefix += tl.sum(x, 0)
        tl.store(out_ptr + offsets, tile_scan, mask=mask, cache_modifier=".cg")

# ------------------- scan_then_propagate (3 kernel implementation) -------------------
def scan_then_propagate(x):
    assert x.ndim == 1
    N = x.numel()
    TILE_SIZE = 4096
    num_warps = 8
    num_tiles = triton.cdiv(N, TILE_SIZE)
    max_ctas = 82 * 4
    num_ctas = min(num_tiles, max_ctas)
    ROOT_SCAN_TILE_SIZE = triton.next_power_of_2(num_ctas)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    block_sums = torch.empty((num_ctas,), dtype=x.dtype, device=x.device)
    block_inclusive_prefix = torch.empty((num_ctas,), dtype=x.dtype, device=x.device)
    block_scan = torch.empty_like(x)

    # 3-kernel implementartion
    scan_then_propagate_block_scan_kernel[(num_ctas, 1, 1)](
        x, block_sums, block_scan, N, tiles_per_cta, TILE_SIZE, num_warps=num_warps)
    reduce_then_scan_root_scan_kernel[(1, 1, 1)](
        block_sums, block_inclusive_prefix, num_ctas, ROOT_SCAN_TILE_SIZE, num_warps=num_warps)
    scan_then_propagate_propagate_kernel[(num_ctas, 1, 1)](
        block_scan, block_inclusive_prefix, N, tiles_per_cta, TILE_SIZE, num_warps=num_warps)
    return block_scan

@triton.jit
def scan_then_propagate_block_scan_kernel(in_ptr, block_sum_ptr, block_scan_ptr, N, tiles_per_cta, TILE_SIZE: tl.constexpr, ):
    """The same kernel as the block sum in parallel reduce"""
    pid = tl.program_id(0).to(tl.int64)
    block_offset = pid * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)
    prefix = tl.zeros((), dtype=in_ptr.type.element_ty)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        mask = offsets < N
        x = tl.load(in_ptr + offsets, mask=mask)
        tile_scan = prefix + tl.cumsum(x, 0)
        prefix +=  tl.sum(x, 0)
        tl.store(block_scan_ptr + offsets, tile_scan, mask=mask)
    tl.store(block_sum_ptr + pid, prefix, cache_modifier=".cg")

@triton.jit
def scan_then_propagate_propagate_kernel(io_ptr, previous_sum_ptr, N, tiles_per_cta, TILE_SIZE: tl.constexpr):
    # we can also loop here if you insist
    pid = tl.program_id(0).to(tl.int64)
    block_offset = pid * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)

    prefix = tl.load(previous_sum_ptr + pid - 1, mask=pid > 0, other=0)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        mask = offsets < N
        x = tl.load(io_ptr + offsets, mask=mask)
        tile_scan = prefix + x
        tl.store(io_ptr + offsets, tile_scan, mask=mask, cache_modifier=".cg")

# ------------------- chained with a global state(it is just a demo, it is not a good idea) -------------------
def chained_scan_global_state(x):
    """Stream scan"""
    assert x.ndim == 1
    N = x.numel()
    TILE_SIZE = 16384
    num_warps = 8
    num_tiles = triton.cdiv(N, TILE_SIZE)
    num_ctas = num_tiles

    out = torch.empty_like(x)
    flag = torch.zeros((), dtype=torch.int32, device="cuda:0")
    previous_sum = torch.zeros((), dtype=x.dtype, device="cuda:0")
    grid = (num_ctas, 1, 1)
    chained_scan_kernel_global_state[grid](x, out, flag, previous_sum, N, TILE_SIZE, num_warps=num_warps)
    return out

@triton.jit
def chained_scan_kernel_global_state(in_ptr, out_ptr, flag_ptr, previous_sum_ptr, N, TILE_N: tl.constexpr):
    pid = tl.program_id(0)
    n_offsets = pid * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + n_offsets, mask=mask, other=0)
    local_cumsum = tl.cumsum(x, 0)
    local_sum = tl.sum(x, 0)
    # It is serial, so relaxed also works
    while (tl.atomic_add(flag_ptr, 0, sem="relaxed") != pid):
        pass
    previous_sum = tl.atomic_add(previous_sum_ptr, local_sum, sem="relaxed")
    tl.atomic_add(flag_ptr, 1, sem="relaxed")
    tl.store(out_ptr + n_offsets, previous_sum + local_cumsum, mask=mask)

# ------------------- stream scan -------------------
def stream_scan(x):
    """Stream scan"""
    assert x.ndim == 1
    N = x.numel()
    TILE_SIZE = 16384
    num_warps = 8
    num_tiles = triton.cdiv(N, TILE_SIZE)
    num_ctas = num_tiles

    out = torch.empty_like(x)
    flag = torch.zeros((num_ctas,), dtype=torch.int32, device="cuda:0")
    prefix = torch.zeros((num_ctas,), dtype=x.dtype, device="cuda:0")
    grid = (num_ctas, 1, 1)
    stream_scan_kernel[grid](x, out, flag, prefix, N, TILE_SIZE, num_warps=num_warps)
    return out

@triton.jit
def stream_scan_kernel(in_ptr, out_ptr, flag_ptr, prefix_ptr, N, TILE_N: tl.constexpr):
    pid = tl.program_id(0)

    n_offsets = pid * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + n_offsets, mask=mask, other=0)
    local_sum = tl.sum(x, 0)

    if pid > 0:
        while tl.atomic_add(flag_ptr + pid - 1, 0, sem="acquire") != 1:
            pass
        prefix = tl.load(prefix_ptr + pid - 1)
    else:
        prefix = tl.zeros_like(local_sum)

    updated_prefix = prefix + local_sum
    tl.store(prefix_ptr + pid, updated_prefix)
    tl.atomic_xchg(flag_ptr + pid, 1, sem="release")

    global_cumsum = prefix + tl.cumsum(x, 0)
    tl.store(out_ptr + n_offsets, global_cumsum, mask=mask)

# ------------------- stream scan -------------------
def stream_scan_no_barrier(x):
    """Stream scan"""
    assert x.ndim == 1
    N = x.numel()
    TILE_SIZE = 16384
    num_warps = 8
    num_tiles = triton.cdiv(N, TILE_SIZE)
    num_ctas = num_tiles

    out = torch.empty_like(x)
    block_states = torch.zeros((num_ctas,), dtype=torch.uint64, device="cuda:0")
    grid = (num_ctas, 1, 1)
    stream_scan_no_barrier_kernel[grid](x, out, block_states, N, TILE_SIZE, num_warps=num_warps)
    return out

@triton.jit
def stream_scan_no_barrier_kernel(in_ptr, out_ptr, block_states_ptr, N, TILE_N: tl.constexpr):
    pid = tl.program_id(0)
    n_offsets = pid * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + n_offsets, mask=mask, other=0)
    local_sum = tl.sum(x, 0)
    done_flag = tl.full((), 1, dtype=tl.uint64)

    if pid > 0:
        s = tl.load(block_states_ptr + pid - 1, volatile=True)
        while s == 0:
            s = tl.load(block_states_ptr + pid - 1, volatile=True)
        previous_sum = (s & 0xFFFFFFFF).to(tl.uint32).to(x.type.element_ty, bitcast=True)
        inclusive_sum = previous_sum + local_sum
        s = (done_flag << 32) | inclusive_sum.to(tl.uint32, bitcast=True)
        tl.store(block_states_ptr + pid, s, cache_modifier=".cg")
    else:
        s = (done_flag << 32) | local_sum.to(tl.uint32, bitcast=True)
        tl.store(block_states_ptr + pid, s, cache_modifier=".cg")
        previous_sum = tl.zeros_like(local_sum)

    global_cumsum = previous_sum + tl.cumsum(x, 0)
    tl.store(out_ptr + n_offsets, global_cumsum, mask=mask)

# ------------------------ single_pass_scan_decoupled_lookback ------------------------
def single_pass_scan_decoupled_lookback(x):
    assert x.ndim == 1
    N = x.numel()

    TILE_SIZE = 16384
    num_warps = 8
    num_tiles_n = triton.cdiv(N, TILE_SIZE)
    grid = (num_tiles_n, 1, 1)
    out = torch.empty_like(x)
    flag = torch.zeros((num_tiles_n,), dtype=torch.int32, device="cuda:0")
    aggregate = torch.empty((num_tiles_n,), dtype=x.dtype, device="cuda:0")
    previous_sum = torch.zeros((num_tiles_n,), dtype=x.dtype, device="cuda:0")
    
    single_pass_scan_decoupled_lookback_kernel[grid](
        x, out, flag, previous_sum, aggregate, N, TILE_SIZE, num_warps=num_warps)
    return out

@triton.jit
def single_pass_scan_decoupled_lookback_kernel(
    in_ptr, out_ptr, flag_ptr, inclusive_prefix_ptr, aggregate_ptr, N, TILE_N: tl.constexpr):
    pid_n = tl.program_id(0)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + n_offsets, mask=mask, other=0)
    
    # store local_sum to aggregate
    local_sum = tl.sum(x, 0)
    tl.store(aggregate_ptr + pid_n, local_sum)
    tl.atomic_xchg(flag_ptr + pid_n, 1, sem="release")

    exclusive_prefix = tl.zeros_like(local_sum)
    # decoupled-lookback
    i = pid_n - 1
    while i >= 0:
        flag = tl.atomic_add(flag_ptr + i, 0, sem="acquire")
        while flag == 0:
            flag = tl.atomic_add(flag_ptr + i, 0, sem="acquire")
        if flag == 1:
            this_aggregate = tl.load(aggregate_ptr + i)
            exclusive_prefix += this_aggregate
            i -= 1
        else: # flag == 2
            inclusive_prefix = tl.load(inclusive_prefix_ptr + i)
            exclusive_prefix += inclusive_prefix
            i = -1
    tl.store(inclusive_prefix_ptr + pid_n, exclusive_prefix + local_sum)
    tl.atomic_xchg(flag_ptr + pid_n, 2, sem="release")

    local_cumsum = tl.cumsum(x, 0)
    tl.store(out_ptr + n_offsets, exclusive_prefix + local_cumsum, mask=mask)


def single_pass_scan_decoupled_lookback_no_barrier(x):
    assert x.ndim == 1
    N = x.numel()

    TILE_SIZE = 16384
    num_warps = 8
    num_tiles_n = triton.cdiv(N, TILE_SIZE)
    grid = (num_tiles_n, 1, 1)
    out = torch.empty_like(x)
    state = torch.zeros((num_tiles_n,), dtype=torch.uint64, device="cuda:0")
    single_pass_scan_decoupled_lookback_no_barrier_kernel[grid](
        x, out, state, N, TILE_SIZE, num_warps=num_warps)
    return out

@triton.jit
def single_pass_scan_decoupled_lookback_no_barrier_kernel(
    in_ptr, out_ptr, state_ptr, N, TILE_N: tl.constexpr):
    pid = tl.program_id(0)
    flag_a = tl.full((), 1, dtype=tl.uint64) << 32
    flag_p = tl.full((), 2, dtype=tl.uint64) << 32

    n_offsets = pid * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + n_offsets, mask=mask, other=0)
    
    # store local_sum to aggregate
    local_sum = tl.sum(x, 0)
    tl.store(state_ptr + pid, flag_a | local_sum.to(tl.uint32, bitcast=True), cache_modifier=".cg") 

    exclusive_prefix = tl.zeros_like(local_sum)
    # decoupled-lookback
    i = pid - 1
    while i >= 0:
        state = tl.load(state_ptr + i, volatile=True)
        while state == 0:
            state = tl.load(state_ptr + i, volatile=True)
        value = (state & 0xFFFFFFFF).to(tl.uint32).to(x.type.element_ty, bitcast=True)
        exclusive_prefix += value
        if state & flag_a:
            i -= 1
        else:
            i = -1
    inclusive_prefix = exclusive_prefix + local_sum
    tl.store(state_ptr + pid, flag_p | inclusive_prefix.to(tl.uint32, bitcast=True), cache_modifier=".cg") 

    local_cumsum = tl.cumsum(x, 0)
    tl.store(out_ptr + n_offsets, exclusive_prefix + local_cumsum, mask=mask)

def test_reduce_then_scan():
    x = torch.randint(0, 2, (16 * 1024 * 1024,), dtype=torch.int64, device="cuda")
    ref = torch.cumsum(x, 0)
    hyp = reduce_then_scan(x)
    torch.testing.assert_close(hyp, ref)

def test_scan_then_propagate():
    x = torch.randint(0, 2, (16 * 1024 * 1024,), dtype=torch.int64, device="cuda")
    ref = torch.cumsum(x, 0)
    hyp = reduce_then_scan(x)
    torch.testing.assert_close(hyp, ref)

def test_chained_scan_global_status():
    x = torch.randint(0, 2, (16 * 1024 * 1024,), dtype=torch.int64, device="cuda")
    ref = torch.cumsum(x, 0)
    hyp = chained_scan_global_state(x)
    torch.testing.assert_close(hyp, ref)

def test_stream_scan():
    x = torch.randint(0, 2, (16 * 1024 * 1024,), dtype=torch.int64, device="cuda")
    ref = torch.cumsum(x, 0)
    hyp = stream_scan(x)
    torch.testing.assert_close(hyp, ref)

def test_stream_scan_no_barrier():
    x = torch.randint(0, 2, (16 * 1024 * 1024,), dtype=torch.int32, device="cuda")
    ref = torch.cumsum(x, 0, dtype=x.dtype)
    hyp = stream_scan_no_barrier(x)
    torch.testing.assert_close(hyp, ref)

def test_single_pass_scan_decoupled_lookback():
    x = torch.randint(0, 2, (16 * 1024 * 1024,), dtype=torch.int32, device="cuda")
    ref = torch.cumsum(x, 0, dtype=x.dtype)
    hyp = single_pass_scan_decoupled_lookback(x)
    torch.testing.assert_close(hyp, ref)

def test_single_pass_scan_decoupled_lookback_no_barrier():
    x = torch.randint(0, 2, (16 * 1024 * 1024,), dtype=torch.int32, device="cuda")
    ref = torch.cumsum(x, 0, dtype=x.dtype)
    hyp = single_pass_scan_decoupled_lookback_no_barrier(x)
    torch.testing.assert_close(hyp, ref)

import flag_gems

def benchmark():
    x = torch.randint(0, 2, (1024 * 1024 * 1024,), dtype=torch.int32, device="cuda")
    io_amount = x.numel() * x.itemsize * 2
    t1 = triton.testing.do_bench(lambda: torch.cumsum(x, 0, dtype=torch.int32), return_mode="median")
    t2 = triton.testing.do_bench(lambda: flag_gems.cumsum(x, 0), return_mode="median")
    t3 = triton.testing.do_bench(lambda: reduce_then_scan(x), return_mode="median")
    t4 = triton.testing.do_bench(lambda: scan_then_propagate(x), return_mode="median")
    t5 = triton.testing.do_bench(lambda: chained_scan_global_state(x), return_mode="median")
    t6 = triton.testing.do_bench(lambda: stream_scan(x), return_mode="median")
    t7 = triton.testing.do_bench(lambda: stream_scan_no_barrier(x), return_mode="median")
    t8 = triton.testing.do_bench(lambda: single_pass_scan_decoupled_lookback(x), return_mode="median")
    t9 = triton.testing.do_bench(lambda: single_pass_scan_decoupled_lookback_no_barrier(x), return_mode="median")

    tp1 = io_amount * 1e-9 / (t1 * 1e-3)
    tp2 = io_amount * 1e-9 / (t2 * 1e-3)
    tp3 = io_amount * 1e-9 / (t3 * 1e-3)
    tp4 = io_amount * 1e-9 / (t4 * 1e-3)
    tp5 = io_amount * 1e-9 / (t5 * 1e-3)
    tp6 = io_amount * 1e-9 / (t6 * 1e-3)
    tp7 = io_amount * 1e-9 / (t7 * 1e-3)
    tp8 = io_amount * 1e-9 / (t8 * 1e-3)
    tp9 = io_amount * 1e-9 / (t9 * 1e-3)

    print(f"[torch] {tp1:.3f} GB/s")
    print(f"[flag_gems] {tp2:.3f} GB/s")
    print(f"[reduce-then-scan] {tp3:.3f} GB/s")
    print(f"[scan_then_propagate] {tp4:.3f} GB/s")
    print(f"[chained_scan_global_state] {tp5:.3f} GB/s")
    print(f"[stream_scan] {tp6:.3f} GB/s")
    print(f"[stream_scan_no_barrier] {tp7:.3f} GB/s")
    print(f"[single_pass_scan_decoupled_lookback] {tp8:.3f} GB/s")
    print(f"[single_pass_scan_decoupled_lookback_no_barrier] {tp9:.3f} GB/s")

if __name__ == "__main__":
    benchmark()
