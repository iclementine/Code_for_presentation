"""
The k-bit global histogram kernel in triton.
"""
import torch
import triton
from triton import language as tl
from triton.language.core import _unwrap_if_constexpr

@tl.constexpr
def get_int_t(num_bits: tl.constexpr, signed: tl.constexpr) -> tl.dtype:
    num_bits = _unwrap_if_constexpr(num_bits)
    signed = _unwrap_if_constexpr(signed)
    return tl.core.get_int_dtype(num_bits, signed)

@tl.constexpr
def one_zeros(num_bits: tl.constexpr) -> int:
    num_bits = _unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)

@tl.constexpr
def zero_ones(num_bits: tl.constexpr) -> int:
    num_bits = _unwrap_if_constexpr(num_bits)
    return (1 << (num_bits - 1)) - 1

@triton.jit
def uint_to_uint(x, descending: tl.constexpr=False):
    out = ~x if descending else x
    return out

@triton.jit
def int_to_uint(x, descending: tl.constexpr=False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    udtype = get_int_t(num_bits, False)
    ux = tl.cast(x, udtype, bitcast=True)
    if descending:
        # 0111111....1
        bit_mask: tl.constexpr = zero_ones(num_bits)
        out = ux ^ bit_mask
    else:
        # 1000000...0
        sign_bit_mask: tl.constexpr = one_zeros(num_bits)
        out = ux ^ sign_bit_mask
    return out

@triton.jit
def floating_to_uint(x, descending: tl.constexpr=False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    sdtype = get_int_t(num_bits, True)
    udtype = get_int_t(num_bits, False)
    sx = x.to(sdtype, bitcast=True)
    ux = x.to(udtype, bitcast=True)
    
    sign_bit_mask: tl.constexpr = one_zeros(num_bits)
    # mind the dtype, right_shift for signed is arithmetic right shift
    mask = sign_bit_mask | (sx >> (num_bits - 1)).to(udtype, bitcast=True)
    # 1000000000...0 for positive
    # 1111111111...1 for negative
    if descending:
        out = ux ^ (~mask)
    else:
        out = ux ^ mask
    return out.to(udtype, bitcast=True)

@triton.jit
def convert_to_uint_preverse_order(x: tl.tensor, descending:tl.constexpr=False):
    if x.dtype.is_floating():
        out = floating_to_uint(x, descending)
    elif x.dtype.is_int_signed():
        out = int_to_uint(x, descending)
    elif x.dtype.is_int_unsigned():
        out = uint_to_uint(x, descending)
    return out

@triton.jit
def compute_global_hist_kernel(arr_ptr, out_ptr, 
                        num_passes, n, tiles_n_per_cta,
                        TILE_N: tl.constexpr, TILE_R: tl.constexpr, 
                        num_bits_per_pass: tl.constexpr,
                        descending: tl.constexpr):
    # arr_ptr: (m, n)
    # out_ptr: (m, n_passes, r), where r = 2 ** k_bits is the number of bins
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    r: tl.constexpr = 2 ** num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1 # a.k.a. 2 ** k_bits - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = CTA_TILE_N * pid_n
    cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)
        
    for p in range(0, num_passes): # parallel
        bit_offset = p * num_bits_per_pass
        for r_start in range(0, r, TILE_R): # parallel
            bin_indices = r_start + tl.arange(0, TILE_R)
            acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int64)
            for n_start in range(cta_n_start, cta_n_end, TILE_N): # sequantial
                n_offsets = n_start + tl.arange(0, TILE_N) # (TILE_N, )
                mask = n_offsets < cta_n_end
                arr = tl.load(arr_ptr + pid_m * n + n_offsets, mask=mask) 
                arr = convert_to_uint_preverse_order(arr, descending)
                key = (arr >> bit_offset) & bfe_mask # (TILE_N, )
                matches = tl.where(mask, (bin_indices[:, None] == key), False)#  (TILE_R, TILE_N)
                acc += matches
            local_sum = tl.sum(acc, axis=1)
            tl.atomic_add(out_ptr + pid_m * num_passes * r + p * r + bin_indices, local_sum, sem="relaxed")


def compute_global_hist(arr: torch.Tensor, k_bits: int=8, descending: bool=False) -> torch.Tensor:
    m, n = arr.shape
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    TILE_N = 1024
    tiles_n_per_cta = 8
    CTA_TILE_N = tiles_n_per_cta * TILE_N

    num_bins = 2 ** k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    TILE_R = 16

    grid_n = triton.cdiv(n, CTA_TILE_N)
    grid_for_global_hist = (grid_n, m, 1)

    global_hist = torch.zeros((m, n_passes, num_bins), device=arr.device, dtype=torch.int64)
    compute_global_hist_kernel[grid_for_global_hist](
        arr, global_hist, n_passes, n, tiles_n_per_cta, 
        TILE_N, TILE_R, k_bits, descending)
    return global_hist

if __name__ == "__main__":
    import numpy as np
    x = torch.randint(-10000, 100000, (8, 512 * 1024), dtype=torch.int64, device="cuda")
    x_ref = x.cpu().numpy()

    import numpy_k_bit_global_hist
    global_hist_ref = numpy_k_bit_global_hist.compute_global_hist(x_ref, dim=-1, k_bits=4, descending=False)
    global_hist_hyp = compute_global_hist(x, k_bits=4, descending=False)
    global_hist_hyp = global_hist_hyp.cpu().numpy()

    print(f"global_hist_ref:\n {global_hist_ref[0]}")
    print(f"global_hist_hyp:\n {global_hist_hyp[0]}")
    np.testing.assert_equal(
        global_hist_hyp,
        global_hist_ref)
    print("Pass")
