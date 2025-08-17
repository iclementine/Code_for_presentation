import torch
import triton
from triton import language as tl

@triton.jit
def chained_scan_decoupled_lookback_kernel(
    in_ptr, out_ptr, flag_ptr, N, OUT_N, TILE_N: tl.constexpr):
    # in_ptr: (*, N)
    # out_ptr: (*, N)
    # flag_ptr: (*, OUT_N)
    # aggregate_ptr: (*, OUT_N)
    # inclusive_prefix_ptr: (*, OUT_N)

    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1<< 31
    v_mask: tl.constexpr = (1 << 30) - 1

    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    offsets = pid_m * N + n_offsets
    # if x's dtype is smaller than 32bit, things get wrong
    # so you must cast it to int32 or uint32
    x = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.uint32)
    
    # store local_sum to aggregate
    local_sum = tl.sum(x, 0) # boll ->  uint32
    pack = local_sum | aggregate_mask
    tl.atomic_xchg(flag_ptr + pid_m * OUT_N + pid_n, pack, sem="release")

    
    exclusive_prefix = tl.zeros((), tl.uint32)
    # decoupled-lookback
    i = pid_n - 1
    while i >= 0:
        pack_ = tl.atomic_add(flag_ptr + pid_m * OUT_N + i, 0, sem="acquire") # uint32
        while pack_ == 0:
            pack_ = tl.atomic_add(flag_ptr + pid_m * OUT_N + i, 0, sem="acquire") # uint32
        v = pack_ & v_mask
        exclusive_prefix += v
        if pack_ & inclusive_prefix_mask:
            i = -1
        else:
            i -= 1
    inclusive_prefix = tl.cast(exclusive_prefix + local_sum, tl.uint32, bitcast=True)
    pack2 = inclusive_prefix | inclusive_prefix_mask
    tl.atomic_xchg(flag_ptr + pid_m * OUT_N + pid_n, pack2, sem="release")

    local_cumsum = tl.cumsum(x, 0)
    tl.store(out_ptr + offsets, exclusive_prefix + local_cumsum, mask=mask)


def chained_scan_decoupled_lookback(x):
    m, n = x.shape
    TILE_N = 4096
    num_tiles_n = triton.cdiv(n, TILE_N)
    grid = (num_tiles_n, m, 1)
    out = torch.empty_like(x, dtype=torch.int64)

    # only use 32bit status
    status = torch.zeros((m, num_tiles_n), dtype=torch.uint32, device="cuda:0")
    chained_scan_decoupled_lookback_kernel[grid](
        x, out, status, n, num_tiles_n, TILE_N, num_warps=8)
    return out

if __name__ == "__main__":
    x = torch.randint(0, 2, (1, 4096 * 1024), dtype=torch.bool, device="cuda:0")
    y = torch.cumsum(x, 1)
    print(f"ref: {y}")

    for _ in range(1000):
        y2 = chained_scan_decoupled_lookback(x)
        print(f"hyp: {y2}")
        torch.testing.assert_close(y2, y, atol=0, rtol=0)
