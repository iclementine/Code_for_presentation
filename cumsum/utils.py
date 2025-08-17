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