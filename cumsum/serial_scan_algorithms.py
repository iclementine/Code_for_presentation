import triton
from triton import language as tl
import torch

def cumsum_persistent(x):
    # cumsum along axis -1 for 2d tensors
    assert x.is_contiguous()
    n = x.shape[-1]
    m = x.numel() // n
    out = torch.empty_like(x)
    grid = (m, 1, 1)
    TILE_N = triton.next_power_of_2(n)
    cumsum_persistent_kernel[grid](x, out, m, n, TILE_N)
    return out
    
@triton.jit
def cumsum_persistent_kernel(in_ptr, out_ptr, M, N, TILE_N: tl.constexpr):
    # 1d grid on batch dimension
    pid = tl.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N
    offsets = pid * N + n_offsets
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out = tl.cumsum(x, 0)
    tl.store(out_ptr + offsets, out, mask=mask)

def cumsum_sequential(x):
    assert x.is_contiguous()
    n = x.shape[-1]
    m = x.numel() // n
    out = torch.empty_like(x)
    grid = (m, 1, 1)
    TILE_N = 4096
    cumsum_loop_kernel[grid](x, out, m, n, TILE_N)
    return out

@triton.jit
def cumsum_loop_kernel(in_ptr, out_ptr, M, N, TILE_N: tl.constexpr):
    # serial: 1d grid on batch dimension
    pid = tl.program_id(0)
    previous_sum = tl.zeros((), dtype=out_ptr.type.element_ty)
    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        offsets = pid * N + n_offsets
        x = tl.load(in_ptr + offsets, mask=mask, other=0)
        partial_cumsum = previous_sum + tl.cumsum(x, 0)
        previous_sum += tl.sum(x, 0)
        tl.store(out_ptr + offsets, partial_cumsum, mask=mask)

def test_cumsum_persistent():
    x = torch.randint(0, 5, (10, 4096), dtype=torch.int64, device="cuda")
    ref = torch.cumsum(x, 1)
    hyp = cumsum_persistent(x)
    torch.testing.assert_close(hyp, ref)


def test_cumsum_loop():
    x = torch.randint(0, 5, (10, 256 * 1024), dtype=torch.int64, device="cuda")
    ref = torch.cumsum(x, 1)
    hyp = cumsum_sequential(x)
    torch.testing.assert_close(hyp, ref)

"""
`pip install pytest-repeat`
and `pytest --count n ...`
to run it repeatedly to ensure that the result is always correct.
"""

import flag_gems
def benchmark():
    x = torch.randint(0, 2, (16 * 1024 * 1024,), dtype=torch.int32, device="cuda")
    io_amount = x.numel() * x.itemsize * 2
    t1 = triton.testing.do_bench(lambda: torch.cumsum(x, 0, dtype=torch.int32), return_mode="median")
    t2 = triton.testing.do_bench(lambda: flag_gems.cumsum(x, 0), return_mode="median")
    t3 = triton.testing.do_bench(lambda: cumsum_sequential(x), return_mode="median")


    tp1 = io_amount * 1e-9 / (t1 * 1e-3)
    tp2 = io_amount * 1e-9 / (t2 * 1e-3)
    tp3 = io_amount * 1e-9 / (t3 * 1e-3)

    print(f"[torch] {tp1:.3f} GB/s")
    print(f"[flag_gems] {tp2:.3f} GB/s")
    print(f"[sequantial-scan] {tp3:.3f} GB/s")


if __name__ == "__main__":
    benchmark()