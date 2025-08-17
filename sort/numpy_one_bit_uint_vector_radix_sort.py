"""
Radix sort for a vector of unsigned integers

- only vector sort to be simple in indexing
- only support unsigned integers, which is radix sort intended for (support for other dtypes in next episodes)
- no indices, or key-value sort by key
- no descending
- only stable sort
- one-bit per pass
"""
import numpy as np


def sweep(arr, i):
    N = arr.size
    index = np.arange(N, dtype=np.uint64)
    key = arr
    
    b = ((key >> i) & 1)
    e = 1 - b
    f = np.cumulative_sum(e, include_initial=True)[:-1]
    total_zeros = e[-1] + f[-1]
    p = np.where(b, index - f + total_zeros, f)
    arr[p] = arr

def radix_sort(arr):
    arr_copy = np.copy(arr)
    num_bits = arr.itemsize * 8
    for i in range(num_bits):
        sweep(arr_copy, i)
    return arr_copy
    
if __name__ == "__main__":
    x = np.random.randint(0, 100000, 256 * 1024, dtype=np.uint64)
    np.testing.assert_allclose(radix_sort(x), np.sort(x))
    print("Pass for uint64")

    x = np.random.randint(0, 100000, 256 * 1024, dtype=np.uint32)
    np.testing.assert_allclose(radix_sort(x), np.sort(x))
    print("Pass for uint32")
