"""
Radix sort for a matrix of unsigned integers

- mateix sort, adding a batch dimension makes it hard to understand the indexing code
- only support unsigned integers, which is radix sort intended for (support for other dtypes in next episodes)
- returing indices
- returing global histogram of bins for each pass
- no descending
- only stable sort
"""
import numpy as np

def sweep(arr, i, k_bits):
    # inplace sort
    # print(f"sorting against [{i}, {i + k_bits}) bits")
    m, n = arr.shape
    num_bins = 2 ** k_bits # r = num_bins
    bfe_mask = 2 ** k_bits - 1
    bins = np.arange(0, num_bins, dtype=np.uint64)  #(r, )

    key = (arr >> i) & bfe_mask # (m, n)
    matches = bins[:, None] == np.expand_dims(key, 1)  #(m, r, n)
    ex_cumsum_in_bin = np.cumulative_sum(matches, axis=-1, include_initial=True, dtype=np.uint64) #(m, r, n+1)
    hist = ex_cumsum_in_bin[:, :, -1] # (m, r), actually a sum of matches along axis 1
    ex_cumsum_in_bin = ex_cumsum_in_bin[:, :, :-1] # (m, r, n) this can be skipped

    ex_cumsum_bins = np.cumulative_sum(hist, axis=1, include_initial=True) # (m, r+1)
    ex_cumsum_bins = ex_cumsum_bins[:, :-1] # (m, r)
    # scatter
    n_index = np.arange(n, dtype=np.uint64) 
    m_index = np.arange(m, dtype=np.uint64)
    # pos is the index for each value in the sorted sequence, shape(m, n)
    pos = np.take_along_axis(ex_cumsum_bins, key, 1) \
        + np.take_along_axis(ex_cumsum_in_bin, np.expand_dims(key, 1), 1).squeeze(1)
    return pos, hist

def radix_sort(arr: np.ndarray, k_bits: int=4):
    m, n  = arr.shape
    dtype = arr.dtype
    arr_copy = np.copy(arr)
    num_bits = 1 if dtype == np.bool else arr.itemsize * 8
    n_passes = (num_bits + k_bits - 1) // k_bits
    num_bins = 2 ** k_bits
    global_hist = np.empty((m, n_passes, num_bins), dtype=np.uint64)
    indices = np.arange(n, dtype=np.int64)
    indices = np.tile(indices, (m, 1)) #(m, n)
    for i in range(n_passes):
        bit_offset = i * k_bits
        pos, hist = sweep(arr_copy, bit_offset, k_bits)
        global_hist[:, i, :] = hist # record the global_hist
        np.put_along_axis(arr_copy, pos, arr_copy, 1)
        np.put_along_axis(indices, pos, indices, 1)
    return arr_copy, indices, global_hist


if __name__ == "__main__":
    x = np.random.randint(0, 10000, (5, 1024)).astype(np.int64)
    sorted_ref = np.sort(x, axis=1, stable=True)
    indices_ref = np.argsort(x, axis=1, stable=True)

    out = radix_sort(x, k_bits=4)
    print("========= sorted_values =========")
    print(f"expected: \n{sorted_ref}")
    print(f"actual:\n{out[0]}")
    np.testing.assert_equal(out[0], sorted_ref)

    print("========= indices =========")
    print(f"expected: \n{indices_ref}")
    print(f"actual: :\n{out[1]}")
    np.testing.assert_equal(out[1], indices_ref)


    print("========= global hist of radix sort =========")
    print(out[2])

    print("Pass")

