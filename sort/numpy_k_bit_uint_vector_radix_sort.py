"""
Radix sort for a vector of unsigned integers

- only vector sort to be simple in indexing
- only support unsigned integers, which is radix sort intended for (support for other dtypes in next episodes)
- no indices, or key-value sort by key
- no descending
- only stable sort
"""
import numpy as np


def sweep(arr, i, k_bits):
    # print(f"sorting against [{i}, {i + k_bits}) bits")
    N = arr.size
    bfe_mask = 2 ** k_bits - 1
    key = (arr >> i) & bfe_mask
    bins = np.arange(0, 2 ** k_bits, dtype=np.uint64) #(r, )
    matches = bins[:, None] == key  #(r, N)
    ex_cumsum_in_bin = np.cumulative_sum(matches, axis=1, include_initial=True, dtype=np.uint64) #(r, N+1)
    hist = ex_cumsum_in_bin[:, -1] # (r, ), actually a sum of matches along axis 1
    # ex_cumsum_in_bin = ex_cumsum_in_bin[:, :-1] # (r, N) this can be skipped
    ex_cumsum_bins = np.cumulative_sum(hist, axis=0, include_initial=True)
    # scatter
    index = np.arange(N, dtype=np.uint64) 
    pos = ex_cumsum_bins[key] + ex_cumsum_in_bin[key, index]
    arr[pos] = arr

def radix_sort(arr, k_bits=4):
    arr_copy = np.copy(arr)
    num_bits = arr.itemsize * 8
    for i in range(0, num_bits, k_bits):
        sweep(arr_copy, i, k_bits)
    return arr_copy

def test_for_uint64():
    x = np.random.randint(0, 100000, 256 * 1024, dtype=np.uint64)
    np.testing.assert_allclose(radix_sort(x), np.sort(x))

def test_for_uint32():
    x = np.random.randint(0, 100000, 256 * 1024, dtype=np.uint32)
    np.testing.assert_allclose(radix_sort(x), np.sort(x))

if __name__ == "__main__":
    x = np.random.randint(0, 100, 8, dtype=np.uint32)
    radix_sort(x, 2)

