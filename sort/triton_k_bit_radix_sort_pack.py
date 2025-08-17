"""
Try a solution without memeory barrier.
"""
import torch
import triton
from triton import language as tl

from triton_k_bit_global_hist import convert_to_uint_preverse_order, compute_global_hist_kernel
from utils import generate_matrix

@triton.jit
def sweep(arr_ptr, associate_arr_ptr, # inputs: (key & value)
          out_ptr, associate_out_ptr, # outputs: (key & value)
          excumsum_bins_ptr, status_ptr, # aux input and status 
          n_passes, pass_id, bit_offset, N, OUT_N, 
          TILE_N: tl.constexpr, TILE_R: tl.constexpr, 
          k_bits: tl.constexpr, descending: tl.constexpr):
    # r: num_bins = 2 ** k_bits
    # OUT_N: grid_n = cdiv(N, )

    # arr_ptr: (m, N)
    # out_ptr: (m, N)
    # excumsum_bins_ptr: (m, n_passes, r)
    # flag_ptr: (m, r, OUT_N)

    # grid: (m, grid_r, grid_n)

    # load data
    pid_n = tl.program_id(0) # TODO: use a global counter to avoid dead-lock
    pid_r = tl.program_id(1)
    pid_m = tl.program_id(2)

    # bit masks
    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1 << 31
    v_mask: tl.constexpr = (1 << 30) - 1
    bfe_mask: tl.constexpr = (1 << k_bits) - 1 # a.k.a. 2 ** k_bits - 1 

    # initialize flag to zero-local sum is not ready
    r: tl.constexpr = 2 ** k_bits
    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    # cumsum for a bin_index
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N) # (TILE_N, )
    mask = n_offsets < N
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
    arr_u = convert_to_uint_preverse_order(arr, descending)
    key = (arr_u >> bit_offset) & bfe_mask # (TILE_N, )

    # since triton can only use scalar as condition, loop by bin_index
    # status must be pre zero-initialized, or else we have to initialize it
    for bin_index in range(cta_r_start, cta_r_end):
        matches = tl.where(mask, key == bin_index, False) # (TILE_N, ) bool
        # cta level cumsum per bin
        # CAUTION: tl.sum in triton 3.2 does not promote type
        local_sum = tl.sum(matches.to(tl.uint32), axis=0)
        pack0 = aggregate_mask | local_sum
        status_offset = pid_m * (r * OUT_N) + bin_index * OUT_N + pid_n
        tl.store(status_ptr + status_offset, pack0, cache_modifier=".cg")

        # decoupled lookback
        exclusive_prefix = tl.zeros((), dtype=tl.uint32)
        i_lookback = pid_n - 1
        while i_lookback >= 0:
            flag_offset_i = pid_m * (r * OUT_N) + bin_index * OUT_N + i_lookback
            pack1 = tl.load(status_ptr + flag_offset_i, volatile=True) # uin32
            while pack1 == 0:
                pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            exclusive_prefix += (pack1 & v_mask)
            if (pack1 & aggregate_mask) == aggregate_mask:
                i_lookback -= 1
            else:
                i_lookback = -1
        pack2 =  inclusive_prefix_mask | (exclusive_prefix + local_sum)
        tl.store(status_ptr + status_offset, pack2, cache_modifier='.cg')

        local_ex_cumsum = tl.cumsum(matches.to(tl.uint32), axis=0) - matches # (TILE_N, )
        ex_cumsum_in_bin = exclusive_prefix + local_ex_cumsum # global ex_cumsum_in_bin (TILE_N, ) 

        # ex_cumsum_bins (m, n_passes, r)
        ex_cumsum_bins = tl.load(excumsum_bins_ptr + pid_m * (n_passes * r) + pass_id * r + bin_index) # scalar
        pos = ex_cumsum_bins + ex_cumsum_in_bin #(TILE_N, )

        # scatter
        tl.store(out_ptr + pid_m * N + pos, arr, mask=matches)
        if associate_arr_ptr is not None:
            associate_arr = tl.load(associate_arr_ptr + pid_m * N + n_offsets, mask=mask)
            tl.store(associate_out_ptr + pid_m * N + pos, associate_arr, mask=matches)



def radix_sort(arr, k_bits=8, descending=False):
    m, n = arr.shape
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
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

    global_hist = torch.zeros((m, n_passes, num_bins), device=arr.device, dtype=torch.int32)
    compute_global_hist_kernel[grid_for_global_hist](arr, global_hist, n_passes, n, tiles_n_per_cta, TILE_N, TILE_R, k_bits, descending)
    ex_cumsum_bins = torch.cumsum(global_hist, -1) - global_hist
    ex_cumsum_bins = ex_cumsum_bins.to(torch.uint32)

    # sort
    arr_in = torch.clone(arr)
    indices_in = torch.arange(0, n, dtype=torch.int64, device=arr_in.device).broadcast_to((m, n)).contiguous()
    arr_out = torch.empty_like(arr)
    indices_out = torch.empty_like(indices_in)

    TILE_R = 8
    grid_r = triton.cdiv(num_bins, TILE_R)
    TILE_N = 2048
    grid_n = triton.cdiv(n, TILE_N)    
    grid_for_sweep = (grid_n, grid_r, m)

    status = torch.empty((m, num_bins, grid_n), device=arr.device, dtype=torch.uint32)

    for i in range(0, n_passes):
        bit_offset = i * k_bits
        status.zero_() # must zero init it.
        sweep[grid_for_sweep](arr_in, indices_in, arr_out, indices_out, 
                              ex_cumsum_bins, status, 
                              n_passes, i, bit_offset, n, grid_n, 
                              TILE_N, TILE_R, k_bits, descending)
        # print(f"< sorted last {bit_offset + k_bits:>2d} bits: {arr_out}")
        arr_in, arr_out = arr_out, arr_in
        indices_in, indices_out = indices_out, indices_in
        
    return arr_in, indices_in




def test(dtype, descending):
    print(f"======= START: {dtype=}, {descending=} =======")
    m = 8
    n = 128 * 1024
    x = generate_matrix(m, n, dtype)

    K = 1 if dtype == torch.bool else 4
    # since torch sort & bitwise operation has no kernel for many uint dtypes
    # we move it to the cpu for computing a reference
    try:
        out = torch.sort(x, dim=-1, stable=True, descending=descending)
    except:
        x_ref = x.to(torch.device("cpu"))
        out = torch.sort(x_ref, dim=-1, stable=True, descending=descending)

    for _ in range(100): # test it repeatedly to ensure it is correct
        out2 = radix_sort(x, k_bits=K, descending=descending)
        d = x.device
        torch.testing.assert_close(out2[0], out[0].to(d), atol=0, rtol=0)
        torch.testing.assert_close(out2[1], out[1].to(d), atol=0, rtol=0)



def bench(dtype):
    m = 1
    n = 1024 * 1024
    K = 1 if dtype == torch.bool else 4
    x = generate_matrix(m, n, dtype)
    sorted1, indices1 = torch.sort(x, stable=True, dim=-1)
    sorted2, indices2 = radix_sort(x, k_bits=K)
    torch.testing.assert_close(sorted2, sorted1, rtol=0, atol=0)
    torch.testing.assert_close(indices2, indices1, rtol=0, atol=0)

    t1 = triton.testing.do_bench(lambda: torch.sort(x, stable=True, dim=-1))
    t2 = triton.testing.do_bench(lambda: radix_sort(x, k_bits=K))
    print(f"torch: {t1:.3f} ms, triton: {t2:.3f} ms")

if __name__ == "__main__":
    # test(torch.bool, False)
    # test(torch.bool, True)

    # test(torch.int8, False)
    # test(torch.int8, True)  

    # test(torch.int16, False)
    # test(torch.int16, True)  

    # test(torch.int32, False)
    # test(torch.int32, True)

    # test(torch.int64, False)
    # test(torch.int64, True)

    # test(torch.uint8, False)
    # test(torch.uint8, True)  

    # test(torch.uint16, False)
    # test(torch.uint16, True)  

    # test(torch.uint32, False)
    # test(torch.uint32, True)  

    # test(torch.uint64, False)
    # test(torch.uint64, True)  


    # test(torch.bfloat16, False)
    # test(torch.bfloat16, True)

    # test(torch.float16, False)
    # test(torch.float16, True)
    
    # test(torch.float32, False)
    # test(torch.float32, True)

    # test(torch.float64, False)
    # test(torch.float64, True)

    print("benchmarking: ")
    bench(torch.int32)
    bench(torch.int64)
    bench(torch.float32)
    