"""
Standalone function to compute histogram of digits per k-bits.
"""
import numpy as np

def get_int_t(num_bits: int, signed: bool = False) -> type:
    prefix: str = "int" if signed else "uint"
    typename: str = f"{prefix}{num_bits}"
    assert hasattr(np, typename), f"illegal typename {typename}"
    return getattr(np, typename)

def floating_to_uint(x: np.ndarray, descending: bool = False) -> np.ndarray:
    dtype: np.dtype = x.dtype
    num_bits: int = dtype.itemsize * 8
    sdtype: np.dtype = get_int_t(num_bits, True)
    udtype: np.dtype = get_int_t(num_bits, False)
    sx = x.view(sdtype)
    ux = x.view(udtype)
    
    sign_bit_mask = 1 << (num_bits - 1)
    mask = sign_bit_mask | (sx >> (num_bits - 1)).to(udtype, bitcast=True)
    # 1000000000...0 for positive
    # 1111111111...1 for negative
    if descending:
        out = ux ^ (~mask)
    else:
        out = ux ^ mask
    return out.view(udtype)

def int_to_uint(x: np.ndarray, descending: bool = False) -> np.ndarray:
    dtype: np.dtype = x.dtype
    num_bits: int = dtype.itemsize * 8 
    udtype: np.dtype = get_int_t(num_bits, False)
    ux = x.view(udtype) # torch bitcast
    if descending:
        # 0111111....1
        bit_mask = (1 << (num_bits - 1)) - 1
        out = ux ^ bit_mask
    else:
        # 1000000...0
        sign_bit_mask = 1 << (num_bits - 1)
        out = ux ^ sign_bit_mask
    return out

def uint_to_uint(x: np.ndarray, descending: bool = False) -> np.ndarray:
    out = ~x if descending else x
    return out

def convert_to_uint_preverse_order(x: np.ndarray, descending: bool = False) -> np.ndarray:
    if x.dtype.kind == "c":
        raise TypeError("Not supported dtype: {x.dtype}")
    elif x.dtype.kind == "f":
        out = floating_to_uint(x, descending)
    elif x.dtype.kind == "i":
        out = int_to_uint(x, descending)
    elif x.dtype.kind in ("b", "u"): # unsigned integer
        out = uint_to_uint(x, descending)
    return out

def compute_global_hist(arr: np.ndarray, dim: int=-1, k_bits: int=8, descending: bool=False):
    dtype: np.dtype = arr.dtype
    num_bits: int = 1 if dtype == np.bool else dtype.itemsize * 8
    n_passes: int = (num_bits + k_bits - 1) // k_bits
    num_bins: int = 2 ** k_bits
    bfe_mask = (1 << k_bits) - 1 # 2 ** k_bits - 1
    m, n = arr.shape
    global_hist = np.zeros((m, n_passes, num_bins), dtype=np.int32)
    arr_u = convert_to_uint_preverse_order(arr, descending)

    bins = np.arange(num_bins, device=arr.device)
    for p in range(n_passes):
        bit_offset = p * k_bits
        key = (arr_u >> bit_offset) & bfe_mask
        matches = (bins[:, None] == np.expand_dims(key, 1)) # (m, r, n)
        h = np.sum(matches, -1) #(m, r)
        global_hist[:, p, :] = h
    return global_hist


if __name__ == "__main__":
    x = np.expand_dims(np.arange(-10000, 2 ** 12), 0).astype(np.int32) * 4 + 1
    K = 3
    hist = compute_global_hist(x, dim=-1, k_bits=K, descending=False)
    print(np.arange(0, 2 **K))
    print("======" * (2 ** K))
    print(hist[0])

