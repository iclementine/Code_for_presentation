import triton
from triton import language as tl
import torch

@triton.jit
def cumsum_persistent_kernel(in_ptr, out_ptr, M, N, TILE_N: tl.constexpr):
    pid = tl.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N
    offsets = pid * N + n_offsets
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out = tl.cumsum(x, 0)
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def cumsum_loop_kernel(in_ptr, out_ptr, tmp_ptr, M, N, TILE_N: tl.constexpr):
    pid = tl.program_id(0)

    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        offsets = pid * N + n_offsets
        x = tl.load(in_ptr + offsets, mask=mask, other=0)
        previous_sum = tl.load(tmp_ptr + pid) # shape()
        partial_cumsum = previous_sum + tl.cumsum(x, 0)

        tl.store(out_ptr + offsets, partial_cumsum, mask=mask)
        tmp_mask = tl.arange(0, TILE_N) == (TILE_N - 1)
        tl.store(tmp_ptr + pid - TILE_N + 1 + tl.arange(0, TILE_N), partial_cumsum, mask=tmp_mask)


@triton.jit
def partial_sum_kernel(in_ptr, out_ptr, M, N, TILE_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    offsets = pid_m * N + n_offsets
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out = tl.sum(x, 0)
    num_tiles_n = tl.num_programs(0)
    tl.store(out_ptr + pid_m * num_tiles_n + pid_n, out)

@triton.jit
def cumsum_split_kernel(in_ptr, previous_sum_ptr, out_ptr, M, N, TILE_N: tl.constexpr):
    # we can also loop here if you insist
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    num_tiles_n = tl.num_programs(0)
    previous_sum = tl.load(previous_sum_ptr + pid_m * num_tiles_n + pid_n - 1, mask=pid_n > 0, other=0)

    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    offsets = pid_m * N + n_offsets
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    local_cumsum = tl.cumsum(x, 0)
    global_cumsum = previous_sum + local_cumsum
    tl.store(out_ptr + offsets, global_cumsum, mask=mask)

def cumsum(x):
    # cumsum along axis -1 for 2d tensors
    assert x.is_contiguous()
    m, n = x.shape
    if n <= 4096:
        out = torch.empty_like(x)
        grid = (m, 1, 1)
        TILE_N = triton.next_power_of_2(n)
        cumsum_persistent_kernel[grid](x, out, m, n, TILE_N)
        return out
    elif n <= 256 * 1024:
        out = torch.empty_like(x)
        tmp = torch.zeros((m,), dtype=x.dtype, device=x.device)
        grid = (m, 1, 1)
        TILE_N = 4096
        cumsum_loop_kernel[grid](x, out, tmp, m, n, TILE_N)
        return out
    else:
        TILE_N = 4096
        num_tiles_n = triton.cdiv(n, TILE_N)
        partial_sums = torch.empty(m, num_tiles_n, dtype=x.dtype, device=x.device)
        grid = (num_tiles_n, m, 1)
        partial_sum_kernel[grid](x, partial_sums, m, n, TILE_N)

        # inplace
        cumsum_persistent_kernel[(m, 1, 1)](
            partial_sums, partial_sums, m, num_tiles_n, triton.next_power_of_2(num_tiles_n)
        )

        out = torch.zeros_like(x)
        grid = (num_tiles_n, m, 1)
        cumsum_split_kernel[grid](x, partial_sums, out, m, n, TILE_N)
        return out

# reduce-then-scan


torch.random.manual_seed(1089)
def test_cumsum_persistent():
    x = torch.randint(0, 5, (10, 4096), dtype=torch.int64, device="cuda")
    ref = torch.cumsum(x, 1)
    hyp = cumsum(x)
    torch.testing.assert_close(hyp, ref)


def test_cumsum_loop():
    x = torch.randint(0, 5, (10, 256 * 1024), dtype=torch.int64, device="cuda")
    ref = torch.cumsum(x, 1)
    hyp = cumsum(x)
    torch.testing.assert_close(hyp, ref)


def test_cumsum_split():
    x = torch.randint(0, 5, (10, 1024 * 1024), dtype=torch.int64, device="cuda")
    ref = torch.cumsum(x, 1)
    hyp = cumsum(x)
    torch.testing.assert_close(hyp, ref)

import numpy as np

def benchmark_cumsum():
    for m in [4096]:
        for n in np.linspace(512, 16384, 5).tolist() + np.linspace(32 * 1024, 128 * 1024, 5).tolist():
            n = int(n)
            x = torch.randint(0, 1, (m, n), dtype=torch.int64, device="cuda")
            def throughput(t):
                return 2 * m * n * 4 * 1e-9 / (t * 1e-3)
            t1 = triton.testing.do_bench(lambda: cumsum(x), return_mode="median")
            t2 = triton.testing.do_bench(lambda: torch.cumsum(x, dim=-1), return_mode="median")
            throughput1 = throughput(t1)
            throughput2 = throughput(t2)
            print(f"{x.shape}\tmy: {throughput1:.2f} GB/s\t{throughput2:.2f} GB/s")

if __name__ == "__main__":
    benchmark_cumsum()