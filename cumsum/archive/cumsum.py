import triton
from triton import language as tl
import torch

# ------------------------ persistent ------------------------
@triton.jit
def persistent_cumsum_kernel(in_ptr, out_ptr, N, TILE_N: tl.constexpr):
    # loads a line a compute cumsum with triton lang's tile primitive
    # n reads & n writes
    tl.assume(TILE_N >= N)
    pid = tl.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N
    offsets = pid * N + n_offsets
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out = tl.cumsum(x, 0)
    tl.store(out_ptr + offsets, out, mask=mask)

# ------------------------ serial ------------------------
@triton.jit
def serial_cumsum_kernel(in_ptr, out_ptr, N, TILE_N: tl.constexpr, STAGES: tl.constexpr):
    # chained-scan, loop by tile and accumulate by using 
    pid = tl.program_id(0)
    previous_sum = tl.zeros((), dtype=out_ptr.type.element_ty)
    for start_n in tl.range(0, N, TILE_N, num_stages=STAGES): # num_stages affects only mm related loop
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N 
        offsets = pid * N + n_offsets
        x = tl.load(in_ptr + offsets, mask=mask, other=0)
        partial_cumsum = previous_sum + tl.cumsum(x, 0)
        previous_sum += tl.sum(x, 0)
        tl.store(out_ptr + offsets, partial_cumsum, mask=mask)

def serial_cumsum(x):
    m, n = x.shape
    out = torch.empty_like(x)
    TILE_N = 4096
    grid = (m, 1, 1)
    serial_cumsum_kernel[grid](x, out, n, TILE_N, STAGES=3, num_warps=8)
    return out

# ------------------------ reduce-then-scan  ------------------------
@triton.jit
def partial_sum_kernel(in_ptr, out_ptr, N, OUT_N, TILE_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    offsets = pid_m * N + n_offsets
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out = tl.sum(x, 0)
    tl.store(out_ptr + pid_m * OUT_N + pid_n, out)

@triton.jit
def partial_sum_serial_kernel(in_ptr, out_ptr, N, OUT_N, k, TILE_N: tl.constexpr, STAGE:tl.constexpr):
    # k: tiles_per_cta in the loop
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    cols_per_cta = TILE_N * k
    acc = tl.zeros((TILE_N, ), dtype=in_ptr.type.element_ty)
    for i in tl.range(0, k, num_stages=STAGE):
        n_offsets = pid_n * cols_per_cta + i * TILE_N + tl.arange(0, TILE_N)
        mask = n_offsets < N
        offsets = pid_m * N + n_offsets
        x = tl.load(in_ptr + offsets, mask=mask, other=0)
        acc += x
    out = tl.sum(acc, 0)
    tl.store(out_ptr + pid_m * OUT_N + pid_n, out)


@triton.jit
def cumsum_split_kernel(in_ptr, previous_sum_ptr, out_ptr, N, PSUM_N, TILE_N: tl.constexpr):
    # we can also loop here if you insist
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    previous_sum = tl.load(previous_sum_ptr + pid_m * PSUM_N + pid_n - 1, mask=pid_n > 0, other=0)

    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    offsets = pid_m * N + n_offsets
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    local_cumsum = tl.cumsum(x, 0)
    global_cumsum = previous_sum + local_cumsum
    tl.store(out_ptr + offsets, global_cumsum, mask=mask)

@triton.jit
def cumsum_split_serial_kernel(in_ptr, previous_sum_ptr, out_ptr, N, PSUM_N, k, TILE_N: tl.constexpr, STAGE: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    cols_per_cta = TILE_N * k
    previous_sum = tl.load(previous_sum_ptr + pid_m * PSUM_N + pid_n - 1, mask=pid_n > 0, other=0)
    for i in tl.range(0, k, num_stages=STAGE):
        n_offsets = pid_n * cols_per_cta + i * TILE_N + tl.arange(0, TILE_N)
        mask = n_offsets < N
        offsets = pid_m * N + n_offsets
        x = tl.load(in_ptr + offsets, mask=mask, other=0)
        local_cumsum = previous_sum + tl.cumsum(x, 0)
        previous_sum += tl.sum(x, 0)
        tl.store(out_ptr + offsets, local_cumsum, mask=mask)

# ------------------------ chained-scan ------------------------
@triton.jit
def chained_scan_kernel(in_ptr, out_ptr, flag_ptr, previous_sum_ptr, N, TILE_N: tl.constexpr):
    # in_ptr: (*, N)
    # out_ptr: (*, N)
    # flag_ptr: (*,)
    # previous_sum_ptr: (*,)
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    offsets = pid_m * N + n_offsets
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    local_cumsum = tl.cumsum(x, 0)
    local_sum = tl.sum(x, 0)
    while (tl.atomic_add(flag_ptr + pid_m, 0, sem="relaxed") != pid_n):
        pass
    previous_sum = tl.load(previous_sum_ptr + pid_m)
    tl.atomic_add(previous_sum_ptr + pid_m, local_sum, sem="relaxed")
    tl.atomic_add(flag_ptr + pid_m, 1, sem="relaxed")
    tl.store(out_ptr + offsets, previous_sum + local_cumsum, mask=mask)


def chained_scan(x):
    m, n = x.shape
    out = torch.empty_like(x)
    flag = torch.zeros((m,), dtype=torch.uint64, device="cuda:0")
    previous_sum = torch.zeros((m,), dtype=x.dtype, device="cuda:0")
    TILE_N = 4096
    grid = (triton.cdiv(n, TILE_N), m, 1)
    chained_scan_kernel[grid](x, out, flag, previous_sum, n, TILE_N, num_warps=8)
    return out

    
# ------------------------ chained-scan-decoupled-lookback ------------------------
@triton.jit
def chained_scan_decoupled_lookback_kernel(
    in_ptr, out_ptr, flag_ptr, inclusive_prefix_ptr, aggregate_ptr, N, OUT_N, TILE_N: tl.constexpr):
    # in_ptr: (*, N)
    # out_ptr: (*, N)
    # flag_ptr: (*, OUT_N)
    # aggregate_ptr: (*, OUT_N)
    # inclusive_prefix_ptr: (*, OUT_N)

    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    offsets = pid_m * N + n_offsets
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    
    # store local_sum to aggregate
    local_sum = tl.sum(x, 0)
    tl.store(aggregate_ptr + pid_m * OUT_N + pid_n, local_sum)
    tl.atomic_xchg(flag_ptr + pid_m * OUT_N + pid_n, 1, sem="release")

    exclusive_prefix = tl.zeros((), dtype=x.dtype)
    # decoupled-lookback
    i = pid_n - 1
    done = False
    while (not done) and (i >= 0):
        flag = tl.atomic_add(flag_ptr + pid_m * OUT_N + i, 0, sem="acquire")
        while flag == 0:
            flag = tl.atomic_add(flag_ptr + pid_m * OUT_N + i, 0, sem="acquire")
        if flag == 1:
            this_aggregate = tl.load(aggregate_ptr + pid_m * OUT_N + i)
            exclusive_prefix += this_aggregate
            i -= 1
        else: # flag == 2
            inclusive_prefix = tl.load(inclusive_prefix_ptr + pid_m * OUT_N + i)
            exclusive_prefix += inclusive_prefix
            done = True
    tl.store(inclusive_prefix_ptr + pid_m * OUT_N + pid_n, exclusive_prefix + local_sum)
    tl.atomic_xchg(flag_ptr + pid_m * OUT_N + pid_n, 2, sem="release")

    local_cumsum = tl.cumsum(x, 0)
    tl.store(out_ptr + offsets, exclusive_prefix + local_cumsum, mask=mask)


def chained_scan_decoupled_lookback(x):
    m, n = x.shape
    TILE_N = 4096
    num_tiles_n = triton.cdiv(n, TILE_N)
    grid = (num_tiles_n, m, 1)
    out = torch.empty_like(x)
    flag = torch.zeros((m, num_tiles_n), dtype=torch.uint64, device="cuda:0")
    aggregate = torch.empty((m, num_tiles_n), dtype=x.dtype, device="cuda:0")
    previous_sum = torch.zeros((m, num_tiles_n), dtype=x.dtype, device="cuda:0")
    
    chained_scan_decoupled_lookback_kernel[grid](
        x, out, flag, previous_sum, aggregate, n, num_tiles_n, TILE_N, num_warps=8)
    return out

if __name__ == "__main__":
    x = torch.randint(0, 2, (1, 4096 * 1024, ), dtype=torch.int64, device="cuda:0")
    y = torch.cumsum(x, 1)

    # torch
    t = triton.testing.do_bench(lambda: torch.cumsum(x, 1), warmup=2, rep=10)
    print(f"torch: {t:.3f}ms, {2 * x.nbytes / (t * 1e6):.3f}GB/s")

    # serial
    for _ in range(1000):
        out = serial_cumsum(x)
        torch.testing.assert_close(out, y)
    t = triton.testing.do_bench(lambda: serial_cumsum(x), warmup=2, rep=10)
    print(f"serial: {t:.3f}ms, {2 * x.nbytes / (t * 1e6):.3f}GB/s")

    # chained 
    for _ in range(1000):
        out = chained_scan(x)
        torch.testing.assert_close(out, y)
    t = triton.testing.do_bench(lambda: chained_scan(x), warmup=2, rep=10)
    print(f"chained: {t:.3f}ms, {2 * x.nbytes / (t * 1e6):.3f}GB/s")

    # chained-decoupled-lookback
    for _ in range(1000):
        out = chained_scan_decoupled_lookback(x)
        torch.testing.assert_close(out, y)
    t = triton.testing.do_bench(lambda: chained_scan_decoupled_lookback(x), warmup=2, rep=10)
    print(f"chained-decoupled-lookback: {t:.3f}ms, {2 * x.nbytes / (t * 1e6):.3f}GB/s")

    from cumsum_of_bool_pack_no_atomic import chained_scan_decoupled_lookback as axx
    # chained-decoupled-lookback
    for _ in range(1000):
        out = axx(x)
        torch.testing.assert_close(out, y)
    t = triton.testing.do_bench(lambda: axx(x), warmup=2, rep=10)
    print(f"chained-decoupled-lookback-pack: {t:.3f}ms, {2 * x.nbytes / (t * 1e6):.3f}GB/s")

    # flag_gems scan then fan
    import flag_gems
    for _ in range(1000):
        out = flag_gems.cumsum(x, 1)
        torch.testing.assert_close(out, y)
    t = triton.testing.do_bench(lambda: flag_gems.cumsum(x, 1), warmup=2, rep=10)
    print(f"flag_gems: {t:.3f}ms, {2 * x.nbytes / (t * 1e6):.3f}GB/s")